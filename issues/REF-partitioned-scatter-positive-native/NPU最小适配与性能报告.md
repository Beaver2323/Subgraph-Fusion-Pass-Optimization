# REF-partitioned-scatter-positive-native NPU 最小适配与性能报告

> 更新时间：2026-09-10 06:55:00 CST（UTC+08:00）

## 问题与处置

上游 partitioned-scatter 默认面向 HIP，显存硬门禁使用 `torch.cuda.mem_get_info()`。generic device
guard 省略 NPU 不视为产品明确关闭；能力评审后，仅将显存查询替换为真实
`torch.npu.mem_get_info()`，其余 memory profile、1.5 GB floor、候选扫描和分区计算沿用上游。

```python
# torch_npu/_inductor/triton_experimental/runtime_estimation.py:91
_, total_device = torch.npu.mem_get_info()
allowed_peak = max(0, total_device - floor_bytes)
profile = scatter.build_memory_profile(graph, is_releasable)
```

## 证据与结论

- [功能 OFF/ON、负例与显存探针](../../results/current/T-085/functional/partitioned-scatter.json)
- [功能 FX/IR/output_code](../../results/current/T-085/functional/artifacts/partitioned-scatter/)
- [性能 OFF/ON 代码](../../results/current/T-085/performance/codegen/partitioned-scatter/)
- [完整闭环报告](../../report/t084_t086_npu_function_performance_and_fix_20260910.md)

直接复用社区百万行 benchmark；ON 发生三次分区改写但 Event p50/p99 回退
`7.68%/7.10%`，且显存峰值增加。能力可用不等于应默认开启，最终保持默认关闭。
