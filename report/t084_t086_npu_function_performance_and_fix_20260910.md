# T-084～T-086 NPU 功能、性能、适配与修复报告

> 更新时间：2026-09-10 07:33:34 CST（UTC+08:00）
> PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> NPU 后端：`triton_experimental`
> 设备：Ascend 910B2；分布式单元使用真实 2-rank HCCL
> 结论：T-084 `1/1`、T-085 `3/3`、T-086 `1/1` 正式闭环。
> 社区对齐结论采用独立 sidecar 保存，并以路径和 SHA256 绑定功能原件；不会改写 performance gate 已签名的功能 JSON。

## 1. 总览

| 任务 / 单元 | GPU reference | NPU 功能 | 性能 verdict | 产品处置 |
| --- | --- | --- | --- | --- |
| T-084 dedup reduce-scatter | A100 原生社区例通过 | 真实 2-rank，2→1，数值正确 | `PERF_IMPROVED` | 保留现状 |
| T-085 overlap device_put | A100 真实 2-rank NCCL | HCCL 中 async→sync，数值正确 | `PERF_MIXED` | 保留安全改写，扩大稳定性复核 |
| T-085 partitioned scatter | A100 三正/负/OFF合同通过 | 3 次改写、NPU 显存门禁、负例正确 | `PERF_REGRESSED` | 保持默认关闭 |
| T-085 pointless cumsum | A100 11 个 dtype 合同通过 | 测试态开启可命中且正确 | `PERF_REGRESSED` | 新增 NPU 默认关闭 gate，真机验证 |
| T-086 reinplace index_put | A100 正负例通过 | 正例重入原位、活跃输入负例受保护 | `PERF_MIXED` | 保留功能，扩大尾延迟复核 |

“正式闭环”表示 GPU 合同、NPU 真机功能、目标结构、性能处置和产品动作均已有证据；它不表示每个
候选都有收益，也不表示延期候选已经完成。

逐单元社区对齐 sidecar 位于：

- [T-084 community alignment](../results/current/T-084/community_alignment/)
- [T-085 community alignment](../results/current/T-085/community_alignment/)
- [T-086 community alignment](../results/current/T-086/community_alignment/)

## 2. 共同执行合同

所有 NPU worker 从 `/home/z50063656/tmp` 启动，并在导入 `torch`/`torch_npu` 前选择
`TORCHINDUCTOR_NPU_BACKEND=triton_experimental`。功能 OFF/ON 与性能
`OFF1 → ON1 → ON2 → OFF2 → OFF3 → ON3` 均使用 fresh process。性能每臂预热 10 次、采样
100 次，同时保留 host 同步时间、NPU Event、内存、FX、IR、`output_code.py` 和源码哈希。

执行层次为：

```text
community-derived callable
  → torch.compile / Dynamo
  → AOTAutograd
  → post_grad_passes
  → 本报告的目标改写
  → lowering
  → scheduler / codegen
  → NPU 执行
```

因此“改写发生”证明 pass 生效，但必须再结合 eager/compiled 数值、负例、fallback 和最终生成代码；
仅有全局 pattern counter 不足以闭环。

## 3. T-084：dedup reduce-scatter

### 3.1 Pattern 与意图

线性关系 `RS(a) + RS(b) = RS(a + b)` 允许先在输入侧相加，再执行一次 reduce-scatter：

```python
# torch/_inductor/fx_passes/fsdp.py:167
def dedup_fsdp_reduce_scatter(gm):
    dedup_rs_pass = _get_dedup_rs_pass()
    while dedup_rs_pass.apply(gm):
        pass
```

社区 GPU 例是单 rank 的真实 CUDA/NCCL 结构合同；NPU adapter 只把设备和进程组替换为 NPU/HCCL，
并扩展到真实两个 rank，没有更换输入语义或直接调用 helper 冒充 post-grad 命中。

### 3.2 NPU 行为与性能

OFF 生成代码有 2 次 `reduce_scatter_tensor.default`，ON 为 1 次；两 rank 数值正确，无 graph
break/CPU fallback。host p50/p99 改善 `16.52%/40.47%`，Event p50/p99 改善
`20.91%/22.00%`。allocated 多 `1 KiB`，reserved 不变。正式判为 `PERF_IMPROVED`。

原件入口：

- 功能：[dedup-reduce-scatter.json](../results/current/T-084/functional/dedup-reduce-scatter.json)
- 性能：[performance_summary.json](../results/current/T-084/performance_summary.json)
- OFF/ON 代码：[codegen/](../results/current/T-084/performance/codegen/dedup-reduce-scatter/)

## 4. T-085：overlap device_put

### 4.1 Pattern 与真实问题

该安全改写在调度前把异步 GPU/NPU→CPU `device_put` 改为同步，避免紧接着的 `tolist()` 读到未完成
传输：

```python
# torch/_inductor/fx_passes/overlap_scheduling.py:40
def make_all_device_put_sync(gm):
    # 找到 non_blocking=True 的 device_put，并改为 False。
    ...
```

NPU 首次进入分析估时器时，调用栈错误进入 CUDA 当前设备查询：

```text
post_grad_passes
  → schedule_overlap_bucketing_from_inductor_configs
  → schedule_overlap_bucketing
  → OverlapScheduler.__init__
  → gather_node_runtime_estimations
  → estimate_fused_node_costs
  → get_transfer_time
  → triton.testing.get_dram_gbps
  → torch.cuda.current_device()
```

这不是 pattern 语义错误，而是通用估时器把“带宽来源”写成 CUDA 专用。最小适配保留上游字节计数，
只在已选择 experimental NPU 后端时读取显式 NPU 带宽：

```python
# torch_npu/_inductor/triton_experimental/runtime_estimation.py:43
def patch_runtime_estimation_for_npu():
    ...
    bandwidth = float(config.overlap_dram_gbps)
    return (read_bytes + write_bytes) / bandwidth
```

CPU/CUDA 和显式 `gpu_type` 路径继续调用上游实现。定向单测覆盖幂等、NPU替换和非NPU委托。

### 4.2 NPU 行为与性能

真实 2-rank HCCL 中 ON 确认 scheduler 被调用且至少一个 device_put 同步化，OFF 不调用目标
scheduler；两 rank 输出正确。聚合 Event p50 改善 `2.38%`、p99 改善 `71.75%`，但 OFF/ON
轮间 p50 spread 达 `3.04×/2.51×`，环境噪声使尾部改善不可稳定归因。正式判为
`PERF_MIXED`，保留安全改写，不宣称性能收益。

原件：[功能](../results/current/T-085/functional/overlap-device-put.json)；
[性能](../results/current/T-085/performance_summary.json)；
[代码](../results/current/T-085/performance/codegen/overlap-device-put/)。

## 5. T-085：partitioned scatter

### 5.1 Pattern、社区 benchmark 与最小适配

高争用 `index_put(accumulate=True)` 被扩展到多个 partition buffer，再归约 partition，以显存换取
较少原子冲突：

```python
# torch/_inductor/fx_passes/reduced_atomic_contention.py:621
def partitioned_scatter_optimization_pass(graph):
    if not config.partitioned_scatter_enabled:
        return graph
    _scan_candidates(graph, ctx)
    ctx.memory = _build_scatter_memory_state(graph)
    return _build_pattern_pass(ctx).apply(graph)
```

本单元直接复用社区 `test_perf_atomic_contention` 的 `N=1,000,000, D=100, n=501`、三个 fp32
scatter 和索引范围 `[0, 8)`；ROCm 的 1.5× 门槛不外推到 NPU。上游显存门禁调用
`torch.cuda.mem_get_info()`，NPU 若直接启用会失去正确的硬预算，因此做如下后端局部适配：

```python
# torch_npu/_inductor/triton_experimental/runtime_estimation.py:91
def patch_partitioned_scatter_memory_for_npu():
    _, total_device = torch.npu.mem_get_info()
    allowed_peak = max(0, total_device - floor_bytes)
    profile = scatter.build_memory_profile(graph, is_releasable)
    return scatter.ScatterMemoryState(..., allowed_peak_bytes=allowed_peak)
```

### 5.2 NPU 行为、对齐边界与性能

ON 发生 3 次改写，实际读取 NPU 总显存 `65,452,113,920 B`，扣除 `1.5 GB` floor 后形成预算；
`accumulate=False` 不命中。功能与社区“主动开启”合同一致，但社区只为 HIP 默认开启，所以这是
`PARTIALLY_ALIGNED`，不是把 NPU 默认配置伪装成 CUDA/ROCm 相同。

Event p50/p99 回退 `7.68%/7.10%`；peak allocated 从 `473,164,800 B` 增至
`493,790,720 B`，reserved 从 `488,636,416 B` 增至 `551,550,976 B`。OFF 为 1 个 Triton
kernel，ON 为 3 个。正式判为 `PERF_REGRESSED`，保持默认关闭。

原件：[功能](../results/current/T-085/functional/partitioned-scatter.json)；
[OFF/ON 代码](../results/current/T-085/performance/codegen/partitioned-scatter/)。

## 6. T-085：pointless cumsum

### 6.1 Pattern 与功能

常量张量沿一维的 cumsum 可写成 `arange(1, n+1) * fill_value`：

```python
# torch/_inductor/fx_passes/post_grad.py:1039
def pointless_cumsum_replacement(match, shape, fill_value, device, dtype, dim):
    idx = torch.arange(1, shape[dim] + 1, device=device, dtype=acc_dtype)
    return (idx * fill_value).view(inter_shape).expand(shape).to(out_dtype)
```

GPU 社区例覆盖 11 个 dtype/输出 dtype 变体。NPU 测试态开启时，同一 handler 命中并保持数值；这
证明 capability，不代表值得默认开启。

### 6.2 性能回退与产品修复

社区最大既有 shape `full([5000], 1.0, fp16) → cumsum(dtype=fp32)` 上，OFF 是一个 Triton full
kernel 加原生 `aten.cumsum`，ON 变成两个 Triton kernel。Event p50/p99 回退
`90.83%/93.33%`，host p50/p99 回退 `89.73%/133.51%`。

产品修复不是删掉上游 pattern，而是只给 NPU 输出加可逆 live gate：

```python
# torch_npu/_inductor/triton_experimental/fx_passes.py:694
def _install_pointless_cumsum_gate():
    ...
    if ncfg.disable_pointless_cumsum and is_npu:
        return False
    return original_check(match)

# torch_npu/_inductor/triton_experimental/config.py:147
disable_pointless_cumsum: bool = True
```

gate 在 `overrides.py` 初始化；CPU/CUDA 不受影响。定向单测 2/2 通过，随后真机产品态验证
`device_execution=true`、`handler_calls=0`、`native_cumsum_calls=1`。性能数据是在加入 gate
之前测得的候选能力数据；产品态验证只证明关闭生效，两者不可混成一组 OFF/ON。

原件：[功能](../results/current/T-085/functional/pointless-cumsum.json)；
[产品 gate 真机验证](../results/current/T-085/fixes/pointless-cumsum/product_gate_result.json)；
[OFF/ON 代码](../results/current/T-085/performance/codegen/pointless-cumsum/)。

## 7. T-086：reinplace index_put

### 7.1 Pattern 与边界

functionalization 后，只有原输入不再被观察且 alias/mutation 安全时，才可把 functional
`index_put + copy_` 恢复成 `index_put_`：

```python
# torch/_inductor/fx_passes/reinplace.py:1062
def reinplace_inplaceable_ops(fake_tensor_updater, graph):
    canonicalize_view_scatter_ops(graph)
    fake_tensor_updater.incremental_update()
    reinplace_inplaceable_ops_core(graph)
    decompose_generalized_scatter(graph)
```

NPU 正例的 before 为 `index_put=1, index_put_=0, copy_=1`，after 为
`index_put=0, index_put_=1, copy_=0`；同时返回原输入与结果的负例保持 functional，输入未污染。
最终 `index_put_` 走 torch_npu 已注册 `IndexPutFallback` extern lowering，不是 CPU fallback；
所以“发生 FX 重入原位”与“生成 Triton scatter kernel”是两个不同层级。

### 7.2 性能

host/Event p50 改善 `17.85%/17.83%`，allocated 少 `4,608 B`；但 host p99 回退
`82.63%`，Event p99 回退 `4.66%`。正式判为 `PERF_MIXED`，保留正确性/内存方面的安全改写，
后续以模型级尾延迟复核决定是否需 shape/场景 gate。

原件：[功能](../results/current/T-086/functional/reinplace-index-put.json)；
[性能](../results/current/T-086/performance_summary.json)；
[OFF/ON 代码](../results/current/T-086/performance/codegen/reinplace-index-put/)。

## 8. 安装态、源码态与后续边界

本轮 `runtime_estimation.py` 和 pointless-cumsum gate 位于 torch_npu 源码树；tracker 的 source
overlay 使测试进程加载这些改动，并记录 loaded-source SHA256。它们不是“当前已安装 wheel 已经永久修改”的
证据。正式产品合入仍需在 torch_npu 仓独立 review、提交并以相同源码重建/安装验证。

延期的 10 个 provisional 候选仍因 CPU-only、缺直接社区合同、仅负例或没有合法同后端 OFF 而不计入
本批分母。下一批应从 T-087 开始，不应把这些延期项写成 T-084～T-086 已完成。
