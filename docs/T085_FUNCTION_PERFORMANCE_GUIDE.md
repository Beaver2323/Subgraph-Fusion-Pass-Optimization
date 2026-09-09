# T-085 功能与性能测例讲解

> 更新时间：2026-09-09 00:43:01 CST（UTC+08:00）
> 状态：零设备准备完成，等待真实 GPU 原生执行；没有 GPU/NPU 通过结论。

T-085 从 5 个 provisional 候选中选出 3 个可执行合同。冻结 PyTorch commit 为
`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。GPU 先运行完整社区原生方法；NPU 后续必须在导入
`torch`/`torch_npu` 前选择 `triton_experimental`，并用新进程隔离 OFF/ON。本文所有“预期”都不是实测结论。

## GPU 一键执行

T-085 包含一个真实 2-rank NCCL 社区例，需要同时暴露两张 GPU：

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-085 \
  --gpus 2,3 \
  --wait-gpu
```

只做零设备检查：

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-085 \
  --validate-only
```

成功后回传 `/data/z50063656/tmp/t085-reference-results/latest-text-handoff.json`。若自动生成分片，上传
`latest/text-handoff-parts/manifest.json` 和全部 `part-*.json` 到 `results/incoming/T-085/`。

## 运行和判定顺序

调用链为：

```text
社区 unittest
  → torch.compile / make_fx
  → AOTAutograd
  → post_grad_passes
  → 目标改写
  → lowering / scheduler / codegen
  → 设备执行
```

目标改写发生在 lowering 与 autotune 前。一次 unittest 通过仍不足以声明目标 pass 生效；复核必须同时看社区
断言、目标计数或结构、FX 前后、生成代码、数值以及 skip/xfail。性能必须等功能 gate 签署后按
`OFF1 → ON1 → ON2 → OFF2 → OFF3 → ON3` 独立进程执行。

## AU-post-grad-overlap-scheduling-device-put-sync

### Pattern 与意图

该单元不是宣称整个 overlap scheduler 已被覆盖，而是收窄到社区测试直接断言的安全改写：只要图中存在
异步 GPU→CPU `device_put`，调度器就把异步标志改成同步，避免后续 `tolist()` 在数据传输完成前读取。

```python
# torch/_inductor/fx_passes/post_grad.py:431-451
if config.aten_distributed_optimizations.enable_overlap_scheduling:
    GraphTransformObserver(gm, "overlap_scheduling").apply_graph_pass(
        lambda graph: schedule_overlap_bucketing_from_inductor_configs(
            graph.owning_module
        )
    )

# torch/_inductor/fx_passes/overlap_scheduling.py:40-78
def make_all_device_put_sync(gm):
    # 发现 non_blocking=True 后，将相应 device_put 改成 False。
    ...
```

### 功能测例

社区入口：
`test/distributed/test_inductor_collectives.py::TestSyncDecisionCrossRanks.test_overlap_scheduling_device_put_sync`。

核心图来自 expert-parallel：先 all-to-all 交换 token 数，计算 splits，一个异步、一个同步传到 CPU，转 list 后再
all-to-all。原生例使用 2-rank CUDA/NCCL，逐节点断言 `device_put` 最终同步，并比较 eager/compiled 输出。

```python
# test/distributed/test_inductor_collectives.py:3748-3795（核心摘录）
input_splits = num_tokens_per_expert.view(group_size, -1).sum(dim=1)
output_splits = num_tokens_per_expert_group.view(group_size, -1).sum(dim=1)
cpu_input_splits = input_splits.to("cpu", non_blocking=True)
cpu_output_splits = output_splits.to("cpu", non_blocking=False)
routed_output = all_to_all_single(
    routed_input,
    cpu_output_splits.tolist(),
    cpu_input_splits.tolist(),
    group_name,
)
```

GPU 预期：真实 2-rank NCCL、至少一个异步 `device_put` 被改为同步、数值相等。NPU 尚未实测；原生方法
硬编码 `torch.cuda`/NCCL，必须先留下原生阻断，再评审只替换设备和 PG 为 NPU/HCCL 的最小适配，后端仍是
`triton_experimental`。

### 性能测例

未找到社区专属 benchmark。worker 保持同一 512 tokens、hidden 128、top-k 2、2-rank all-to-all 子图，OFF/ON
只切目标 scheduler。该 pass 的直接目的包含正确性保护，ON 把异步传输改同步可能变慢；因此性能用于量化安全
成本，不把“加速”作为功能验收条件。host 同步时间是主指标，device Event 不覆盖 CPU `tolist()` 等待，只作补充。

## AU-post-grad-partitioned-scatter-optimization

### Pattern 与意图

高争用 `index_put(..., accumulate=True)` 会让许多线程原子累加同一槽位。该 pass 把输出扩展为多个 partition，
先分散原子写，再归约 partition，以临时显存换取较低争用；它必须经过 accumulate、设备、shape、争用率和显存门禁。

```python
# torch/_inductor/fx_passes/post_grad.py:253-256
if config.partitioned_scatter_enabled:
    partitioned_scatter_optimization_pass(gm.graph)

# torch/_inductor/fx_passes/reduced_atomic_contention.py:621-640
def partitioned_scatter_optimization_pass(graph):
    if not config.partitioned_scatter_enabled:
        return graph
    ctx = ScatterPassContext()
    _scan_candidates(graph, ctx)
    ctx.memory = _build_scatter_memory_state(graph)
    return _build_pattern_pass(ctx).apply(graph)
```

### 功能测例

本批执行三个真实 GPU 社区方法：

- `test_pr_reference_accuracy`：三个高争用累加 scatter，数值容差通过且 `applied >= 3`；
- `test_skip_accumulate_false`：`accumulate=False` 不命中，bitwise 相等；
- `test_pass_disabled`：显式 OFF 时不命中，结果仍正确。

GPU 预期只证明社区主动开启后的功能正确性。NPU 尚未实测；由于默认配置是“HIP 开、其他后端关”，这对 NPU
属于 generic guard 省略而不是已经批准的 NPU 产品开启，需先完成 `triton_experimental` capability review。

### 性能测例

社区确有目标级 benchmark：
`TestPartitionedScatterOpt.test_perf_atomic_contention`，使用 3 个 `N=1,000,000, D=100, n=501` 的 fp32
scatter，索引仅落在 8 个槽。源码明确 1.5× 门槛来自 ROCm MI300X：

```python
# test/inductor/test_scatter_optimization.py:700-755
config.partitioned_scatter_enabled = False
baseline_ms = benchmarker.benchmark_gpu(lambda: baseline_fn(*inputs))
config.partitioned_scatter_enabled = True
partitioned_ms = benchmarker.benchmark_gpu(lambda: partitioned_fn(*inputs))
self.assertGreater(baseline_ms / partitioned_ms, 1.5)
```

CUDA 默认显式关闭，本批不强开做 CUDA 性能，也不把 ROCm 门槛外推到 NPU。NPU 在能力评审、显存门禁和产品决策
签署前保持 `capability-pending`，不得用 `partitioned_scatter_force` 制造 ON 路径。

## AU-post-grad-pointless-cumsum

### Pattern 与意图

对由单个常量填充的张量，沿某维做 cumsum 的第 i 个元素可直接写成 `(i + 1) * fill_value`。改写把 scan 变成
`arange × constant` 的点式表达，并显式处理整数/布尔输入提升和输出 dtype。

```python
# torch/_inductor/fx_passes/post_grad.py:1039-1067
def pointless_cumsum_replacement(match, shape, fill_value, device, dtype, dim):
    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        dtype = torch.int64
    acc_dtype = torch.int64 if integral_out else torch.float64
    idx = torch.arange(1, shape[dim] + 1, device=device, dtype=acc_dtype)
    return (idx * fill_value).view(inter_shape).expand(shape).to(out_dtype)
```

### 功能测例

社区 `test_pointless_cumsum` 包含 11 个函数，覆盖整数、布尔、FP16/FP32、显式 FP32/FP64 输出以及不同维度。
当 `HAS_GPU` 时文件末尾将默认设备设为 `GPU_TYPE`，所以这些无显式 device 的构造器仍在真实 GPU 上执行。
每个分支都要求 generated code 不含 `aten.cumsum`、结果与原函数相等、pattern count 至少为 1。

GPU 和 NPU 当前都尚未运行。NPU 必须确认目标 handler 而不能只用全局 `pattern_matcher_count`；后端固定
`triton_experimental`。

### 性能测例

未找到社区专属 benchmark。目标级 worker 复用社区最大既有 `fn10`：`full([5000], 1.0, fp16)` 后
`cumsum(dtype=fp32)`。OFF 只禁用 `pointless_cumsum_replacement` 注册项，ON 恢复目标项；不关闭整个
pattern matcher。测量覆盖完整常量构造+cumsum编译函数，是 microbenchmark，不代表模型端到端收益，也不能
推广到输入张量 cumsum。

## Deferred 候选

- `AU-post-grad-fix-auto-functionalized-dtype-views`：社区例的 custom op 只注册 CPU/Meta，输入也为 CPU。CPU
  通过不能冻结 GPU 合同；后续若重开，要保留 storage alias、dtype view 和 mutation 语义设计最小 GPU op。
- `AU-post-grad-fuse-ddp-communication`：直接 bucketing/codegen 断言运行 CPU+gloo；真实 GPU compile 例没有直接
  断言目标 pass。需补 2-rank CUDA/NCCL 的目标结构、数值和不与 T-083 重复计数的边界后再重开。

## 后续 NPU 门禁

GPU handoff 经人工复核后，才进入 NPU。每个 NPU 进程在导入前设置：

```bash
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
```

原生阻断、最小适配、修复前后 `fx_graph_readable.py`、`fx_graph_transformed.py`、`ir_pre_fusion.txt`、
`ir_post_fusion.txt`、`output_code.py` 都应落入对应 issue。功能通过后，pointless-cumsum 和 overlap 才能签性能
gate；partitioned-scatter必须先有独立 capability 决策。
