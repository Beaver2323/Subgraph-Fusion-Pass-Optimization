# T-084 功能与性能测例讲解

> 更新时间：2026-09-10 06:55:00 CST（UTC+08:00）
> 当前状态：1/1 单元正式闭环；GPU 原生 reference、NPU 真实两 rank HCCL 功能和六臂性能均完成。4 个候选继续延期。

冻结 PyTorch revision：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。GPU 先运行社区原生
CUDA/NCCL 测例；NPU 动态功能、修复验证和性能只允许 `triton_experimental`，且必须在导入
`torch`/`torch_npu` 前选择后端、OFF/ON 使用新进程。

## 实测闭环结论

NPU `triton_experimental` 的 ON 图把两个 `reduce_scatter_tensor` 收敛为一个，两个 rank
均与 eager 一致，无 graph break/CPU fallback。六臂（每臂 10 次预热、100 次采样）结果：host
p50/p99 改善 `16.52%/40.47%`，NPU Event p50/p99 改善 `20.91%/22.00%`；allocated
峰值从 `11,264 B` 增至 `12,288 B`，reserved 均为 `2 MiB`。结论为
`PERF_IMPROVED`，保留现状，但不外推成完整 FSDP 训练收益。

功能、性能、FX/IR/generated code 和适配调用链集中见
[T-084～T-086 NPU 闭环报告](../report/t084_t086_npu_function_performance_and_fix_20260910.md)。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-084 \
  --gpu 2 \
  --wait-gpu
```

默认允许与已有计算进程共享指定卡；需要启动时独占可额外传 `--exclusive`。固定回传入口：

```text
/data/z50063656/tmp/t084-reference-results/latest-text-handoff.json
```

文件超过网页限制时，入口会输出 `handoff_upload_mode=split` 和分片目录。将同一轮次
`manifest.json` 与全部 `part-*.json` 上传至 `results/incoming/T-084/`。

## 审核结论

原草案的五项并非都能直接成为性能验收单元：

| 草案候选 | 社区功能证据复核 | 性能对照 | 处置 |
| --- | --- | --- | --- |
| `dedup_reduce_scatters` | 直接 CUDA/NCCL 编译、codegen 与数值断言 | 配置 `false/true` 是合法目标 OFF/ON | 本批执行 |
| `decompose_auto_functionalized` | 主测例为 CPU 自定义算子，未找到直接 CUDA 合同 | 无开关 | 延期 |
| `decompose_map_to_while_loop` | `test_control_flow.py::MapTests` 是直接 CUDA 数值合同 | 无开关；OFF 遗留不可 lower 的 `map_impl` | 延期 |
| `decompose_scan_to_while_loop` | `test_control_flow.py::ScanTests` 是直接 CUDA 数值合同 | 无开关；OFF 遗留不可 lower 的 `scan` | 延期 |
| `decompose_triton_kernel_wrapper_functional` | 有直接 CUDA 编译及 mutation HOP 断言 | 无开关；OFF 不是等价可运行路径 | 延期 |

这里的“延期”不代表功能无价值。后三个 decomposition 是进入 lowering 前必须完成的结构规范化，
更接近正确性/可编译性 pass。没有合法同后端 OFF 时，不能通过临时删除 pass 或手写另一个模型来制造
性能收益；待项目确认“必需结构分解的无 OFF 免测/编译成本”政策后再重开。

## AU-post-grad-dedup-reduce-scatters

### Pattern 与意图

入口位于 `torch/_inductor/fx_passes/post_grad.py:341`，只有配置开启才进入实际 helper：

```python
# torch/_inductor/fx_passes/post_grad.py:341
if config.dedup_reduce_scatters:
    from torch._inductor.fx_passes.fsdp import dedup_fsdp_reduce_scatter

    GraphTransformObserver(gm, "dedup_reduce_scatters").apply_gm_pass(
        dedup_fsdp_reduce_scatter
    )
```

实际改写位于 `torch/_inductor/fx_passes/fsdp.py:167`。利用 reduce-scatter 对 `sum/avg` 的线性性质：

```python
# torch/_inductor/fx_passes/fsdp.py:167
# before: wait(RS(a)) + wait(RS(b))
# after:  wait(RS(a + b))
def dedup_fsdp_reduce_scatter(gm):
    dedup_rs_pass = _get_dedup_rs_pass()
    while dedup_rs_pass.apply(gm):
        pass
```

意图是把两次相同通信参数、相同 dtype 且结果只参与该加法的 reduce-scatter 合为一次。handler 的
extra check 会拒绝不同 group/group_size/reduce op、非线性 reduce、不同 dtype 和 wait 多用户。
改写发生在 lowering、scheduler/codegen 和 autotune 之前。

### 功能测例

执行社区原生：

```text
test/distributed/test_inductor_collectives.py::
TestCollectivesInductor.test_dedup_reduce_scatter
```

核心图保持社区源码，不在 tracker 内重写：

```python
# test/distributed/test_inductor_collectives.py:2272
def func(rs_0, rs_1, tag, ranks, group_size):
    group = torch.distributed.distributed_c10d._get_default_group().group_name
    a = torch.ops._c10d_functional.reduce_scatter_tensor(
        rs_0, "avg", group_size, group
    )
    b = torch.ops._c10d_functional.reduce_scatter_tensor(
        rs_1, "avg", group_size, group
    )
    return wait_tensor(a) + wait_tensor(b)
```

输入为两个 `[4, 128]` 连续 FP32 CUDA tensor。测试显式开启 `dedup_reduce_scatters=True`，断言生成
代码只有一次 `reduce_scatter_tensor.default`，并比较 compiled/eager 结果。

GPU reference 的 community harness 是真实 CUDA/NCCL，但固定 `world_size=1`。它足以确认原生入口、
pattern 改写和数值没有回归；不能据此宣称跨 rank 通信正确或有性能收益。GPU 输出必须同时保留
FX before/after 和原生编译 artifacts，skip/xfail/无测试均失败。

NPU 阶段不能直接套用 CUDA/NCCL 类。原生阻断后允许的最小适配仅包括：设备改为 NPU、进程组改为
真实 HCCL、`world_size` 扩展到 2；函数、shape、dtype、`avg`、OFF/ON 配置和数值 oracle 不变。
进入测试前必须确认实际注册后端为 `triton_experimental`。

### 性能测例

冻结 revision 的 `benchmarks/`、`test/inductor/`、`test/distributed/` 中未找到该目标的同后端
OFF/ON 社区 benchmark。因此性能 worker 是从上述社区功能图派生，不是社区原生 benchmark，也不是
FSDP 训练端到端：

- 输入仍为两个 `[4,128]` 连续 FP32 tensor，仍使用 `avg`；
- 唯一必要扩展是 `world_size=2`，用真实 NCCL/HCCL 才能观察通信差异；
- OFF 只设置 `dedup_reduce_scatters=False`，预期 2 次 collective；
- ON 只设置 `dedup_reduce_scatters=True`，预期 2→1；
- `reorder_for_compute_comm_overlap=False` 固定不变，不关闭全局 matcher；
- 每个 rank 都与 eager 比较，任一 rank 错误、fake PG、ON 未改写或 OFF 误改写均拒绝计时。

正式顺序为 `OFF1 → ON1 → ON2 → OFF2 → OFF3 → ON3`，每臂新进程，10 次预热、100 个样本。
记录 host 同步 wall time、设备 Event、编译耗时、allocated/reserved、逐样本数组；每个样本取两 rank
最大值，再对三轮同模式统计取中位数。该范围是完整 collective 子图，但不包含训练器、参数分片、
通信计算重叠或完整 FSDP step。

GPU reference 通过后，先运行 NPU 功能阶段并人工复核原件、签发 gate；没有 gate 时性能 launcher
会拒绝执行。worker 与 launcher 当前只完成零设备静态验证，不能标记为动态性能已测。

## 延期候选的真实调用关系

四项延期 pass 都在 `post_grad_passes` 尾部、reinplace 之后执行：

```text
torch.compile
  → AOTAutograd
  → post_grad_passes
  → reinplace_inplaceable_ops
  → decompose_triton_kernel_wrapper_functional
  → decompose_auto_functionalized
  → decompose_scan_to_while_loop
  → decompose_map_to_while_loop
  → lowering / scheduler / codegen / autotune
```

`map`/`scan` 的原索引指向 `MutationTests.test_while_loop`，该测试只覆盖用户 Triton kernel 内部的
while 语句，不触发 `higher_order.map_impl` 或 `higher_order.scan`，因此已纠正。真实社区合同分别位于
`test/inductor/test_control_flow.py::MapTests` 与 `ScanTests`。本次保留证据但不把它们塞进一个没有
合法 OFF 的性能分母。
