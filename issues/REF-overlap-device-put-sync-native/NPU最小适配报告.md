# REF-overlap-device-put-sync-native NPU 最小适配报告

> 更新时间：2026-09-10 06:55:00 CST（UTC+08:00）

## 问题与处置

社区图迁移到真实 2-rank NPU/HCCL 后，overlap 分析估时调用 CUDA 专用带宽探针：

```text
post_grad_passes → schedule_overlap_bucketing_from_inductor_configs
→ OverlapScheduler → estimate_fused_node_costs → get_transfer_time
→ triton.testing.get_dram_gbps → torch.cuda.current_device()
```

最小适配位于
`torch_npu/_inductor/triton_experimental/runtime_estimation.py:patch_runtime_estimation_for_npu`：只对
已选择的 NPU backend 替换带宽来源，继续复用上游读写字节计算；CPU/CUDA 路径委托原函数。

## 证据

- [功能 OFF/ON 与源码哈希](../../results/current/T-085/functional/overlap-device-put.json)
- [功能 FX/IR/output_code](../../results/current/T-085/functional/artifacts/overlap-device-put/)
- [性能 OFF/ON 代码](../../results/current/T-085/performance/codegen/overlap-device-put/)
- [完整闭环报告](../../report/t084_t086_npu_function_performance_and_fix_20260910.md)

功能对齐社区安全合同；性能轮间高波动，正式为 `PERF_MIXED`，不宣称收益。
