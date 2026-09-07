# T-083 功能与性能测例讲解

> 更新时间：2026-09-07T22:36:49+08:00
> 准备状态：源码合同已审核，GPU/NPU 均未执行；静态校验不是运行结果。

本批从 5 个旧候选中准备 3 个可运行社区合同，其余逐项保留在文末。冻结 PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。
NPU 全程采用 `triton_experimental`，导入 torch/torch_npu 前选后端。先原生 GPU，阻断后才评审最小适配；显式产品关闭免测。

## GPU 一键执行

```bash
cd /data/z50063656/tmp
bash /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh --task T-083 --gpu 2 --wait-gpu
```

固定回传入口为 `/data/z50063656/tmp/t083-reference-results/latest-text-handoff.json`；超限自动生成分片，上传至 `results/incoming/T-083/`，全部分片与 manifest 同批上传。
原生 TestCollectivesInductor 使用 world_size=1；此处一张卡即可，NCCL 要可用。不能把它解释为两卡通信性能。性能 worker 必须由 torchrun 启动 2 个 rank，真实 NCCL/HCCL；拒绝 world_size=1、fake PG、卡数不足。

原生单进程基类位于`torch/testing/_internal/common_distributed.py:1777`，使用真实设备默认通信后端，固定`MASTER_PORT=12355`。同时启动另一份同类suite可能端口冲突；本批顺序执行，冲突时保留原生错误再审核最小端口适配。

## 采集与运行边界

本批保留原生unittest直接入口，使用原生TORCH_COMPILE_DEBUG采集，没有启用joint入口观察器。参数化入口逐项固定，缺例/skip/xfail不算成功；结合world_size=1解释通信证据。

直接结构测试调用链是 `社区 test → make_fx → joint_graph_passes → 图节点断言`，不执行设备kernel。编译测试调用链是 `社区 test → torch.compile → AOTAutograd/joint_graph_passes → post_grad_passes → lowering → scheduler/codegen →（存在候选时）autotune`。图改写在lowering/autotune之前；无kernel的视图消除不能伪称内核加速。

## post-grad独立all-gather分桶及依赖保护

单元：`AU-post-grad-bucket-all-gathers`。源码入口：`torch/_inductor/fx_passes/post_grad.py:384`，`bucket_all_gather`。

被调实现：`torch/_inductor/fx_passes/bucketing.py:394`，`bucket_all_gather(gm, bucket_cap_mb_by_bucket_idx, mode)`。

### 功能测例

- `test/distributed/test_inductor_collectives.py::TestCollectivesInductor.test_all_gather_bucket_copy_cat_fusion`：world_size=1真实CUDA/NCCL原生codegen例；3个[64]输入合并copy(cat)仅1个Triton kernel，原例不作数值比较。
- `test/distributed/test_inductor_collectives.py::TestCollectivesInductor.test_all_gather_bucket_path`：第二allgather依赖第一输出，bucket_all_gathers_fx=all仍须2个wait。codegen断言，不冒充数值。

以下为社区函数的核心摘录；文件位置在代码块内。命名简化处只用于阅读，reference 实际运行完整 upstream 方法。

```python
# test/distributed/test_inductor_collectives.py::TestCollectivesInductor.test_all_gather_bucket_copy_cat_fusion
def func(ag_0, ag_1, ag_2, *, group_size):
    group = dist.distributed_c10d._get_default_group().group_name
    outs = [torch.ops._c10d_functional.all_gather_into_tensor(x, group_size, group)
            for x in (ag_0, ag_1, ag_2)]
    return tuple(torch.ops._c10d_functional.wait_tensor(x) for x in outs)
```

意图与验证：post-grad独立all-gather分桶及依赖保护。只切bucket_all_gathers_fx none/all，其他两类bucket=none，overlap=false；记录collective调用次数3→1及FX。World_size=1功能证据不授予2rank性能门禁。

GPU：等待原生运行，不预判命中。NPU：等待 `triton_experimental` 同合同验证；原始断言、图、设备实现差异将写回 result 与适配/修复报告。

### 性能测例

复用copy_cat_fusion中3个[64] fp32输入/3个all_gather→wait；性能将group_size从1扩展2并按rank输入校验真实NCCL/HCCL；明确分布式适配。
3个独立collective的端到端通信子图，包含打包/collective/解包；不含DDP完整模型。每样本跨rank barrier在计时外，取rank最大值。

OFF/ON：只切bucket_all_gathers_fx none/all，其他两类bucket=none，overlap=false；记录collective调用次数3→1及FX。World_size=1功能证据不授予2rank性能门禁。

无专属社区目标benchmark；从上述社区功能原图派生。保留正确性、目标图变化、生成代码、编译耗时、host/设备Event p50/p99、显存峰值。OFF1→ON1→ON2→OFF2→OFF3→ON3，各自独立进程，10次预热、100样本。不得先计时后补命中证据。

## post-grad独立all-reduce分桶

单元：`AU-post-grad-bucket-all-reduce`。源码入口：`torch/_inductor/fx_passes/post_grad.py:367`，`bucket_all_reduce`。

被调实现：`torch/_inductor/fx_passes/bucketing.py:759`，`bucket_all_reduce(...)`。

### 功能测例

- `test/distributed/test_inductor_collectives.py::TestCollectivesInductor.test_all_reduce_bucket`：原生world_size=1，bucket_mode=all，代码1个all_reduce_，compiled/eager结果相等。

以下为社区函数的核心摘录；文件位置在代码块内。命名简化处只用于阅读，reference 实际运行完整 upstream 方法。

```python
# test/distributed/test_inductor_collectives.py::TestCollectivesInductor.test_all_reduce_bucket
def func(x, w, ar_0, ar_1, group):
    y = torch.mm(x, w)
    a = torch.ops._c10d_functional.all_reduce(ar_0, 'sum', group)
    b = torch.ops._c10d_functional.all_reduce(ar_1, 'sum', group)
    return y, torch.ops._c10d_functional.wait_tensor(a), torch.ops._c10d_functional.wait_tensor(b)
```

意图与验证：post-grad独立all-reduce分桶。只切bucket_all_reduces_fx none/all；另两类bucket=none，overlap=false；编译后计数2→1，所有rank正确且没有fallback/graph break

GPU：等待原生运行，不预判命中。NPU：等待 `triton_experimental` 同合同验证；原始断言、图、设备实现差异将写回 result 与适配/修复报告。

### 性能测例

原生x[4,384],w[384,512],ar0[384,512],ar1[384,256] fp32；保留MM与两个all_reduce及wait输出。性能2rank不删无关MM。
社区MM+collective完整函数端到端，非全训练模型；跨rank最大host/Event与每rankmemory。

OFF/ON：只切bucket_all_reduces_fx none/all；另两类bucket=none，overlap=false；编译后计数2→1，所有rank正确且没有fallback/graph break

无专属社区目标benchmark；从上述社区功能原图派生。保留正确性、目标图变化、生成代码、编译耗时、host/设备Event p50/p99、显存峰值。OFF1→ON1→ON2→OFF2→OFF3→ON3，各自独立进程，10次预热、100样本。不得先计时后补命中证据。

## post-grad reduce-scatter分桶与输出cast

单元：`AU-post-grad-bucket-reduce-scatters`。源码入口：`torch/_inductor/fx_passes/post_grad.py:349`，`bucket_reduce_scatter`。

被调实现：`torch/_inductor/fx_passes/bucketing.py:412`，`bucket_reduce_scatter(...)`。

### 功能测例

- `test/distributed/test_inductor_collectives.py::TestCollectivesInductor.test_reduce_scatter_bucket`：原生default/custom_ops各执行func/func2，1个reduce_scatter调用且compiled/eager相等。

以下为社区函数的核心摘录；文件位置在代码块内。命名简化处只用于阅读，reference 实际运行完整 upstream 方法。

```python
# test/distributed/test_inductor_collectives.py::TestCollectivesInductor.test_reduce_scatter_bucket
a = rs_0.to(torch.bfloat16)
b = rs_1.to(torch.bfloat16)
a = torch.ops._c10d_functional.reduce_scatter_tensor(a, 'sum', world_size, group)
b = torch.ops._c10d_functional.reduce_scatter_tensor(b, 'sum', world_size, group)
return torch.mm(x, w), wait(a), wait(b)
```

意图与验证：post-grad reduce-scatter分桶与输出cast。只切bucket_reduce_scatters_fx none/all与固定bucket_mode=default；另两类bucket=none，overlap=false；2→1collective及正确性后计时

GPU：等待原生运行，不预判命中。NPU：等待 `triton_experimental` 同合同验证；原始断言、图、设备实现差异将写回 result 与适配/修复报告。

### 性能测例

x[4,384],w[384,512] fp32；rs0[384,512],rs1[384,256] fp32先转bf16，sum reduce_scatter后wait；基准default mode/func；func2与custom_ops保留功能回归。
社区MM+cast+collective整个函数，真实world_size=2的端到端子图。float32输出cast变体另行测量才可推广。

OFF/ON：只切bucket_reduce_scatters_fx none/all与固定bucket_mode=default；另两类bucket=none，overlap=false；2→1collective及正确性后计时

无专属社区目标benchmark；从上述社区功能原图派生。保留正确性、目标图变化、生成代码、编译耗时、host/设备Event p50/p99、显存峰值。OFF1→ON1→ON2→OFF2→OFF3→ON3，各自独立进程，10次预热、100样本。不得先计时后补命中证据。

## Deferred 候选与下一步

- `AU-post-grad-chain-random-ops-ordering`：fallback_random触发排序链；未找到针对_chain_random_ops_for_ordering的社区直接断言。随机数正确性/排序需独立合同。
- `AU-post-grad-decomp-comms`：已找到test/distributed/test_decomp_comms.py::TestDecompGramMatrixAllGather，直接helper+FakeTensorMode+fake PG world_size=2只证明结构。需真实多rank数值/稳定性/通信量证据，不能以fake测试宣称GPU计算通过。

Deferred 项不放入 acceptance_units，不自动重排后续 T 编号，也不进入 GPU suite 分母。其功能/性能阻断都在 manifest 留档，获得新证据后再审核。

## 性能执行准备边界

性能命令、原件绑定字段和2rank要求见[性能复核门禁](PREPARED_PERFORMANCE_GATE.md)。

`runners/t081_t083_performance_worker.py` 提供目标级工作负载、OFF/ON控制、数值/图门禁和原始计时采集。`scripts/run_prepared_performance.py` 检查每单元经复核的功能门禁记录后启动6个独立进程；当前没有该记录，因此拒绝正式性能测量。
门禁记录必须绑定 acceptance_unit_id、backend、PyTorch commit、功能报告sha256、GPU与NPU通过状态、显式disable处置、fallback/graph-break以及待测输入。未提供/不一致时停在准备状态。
