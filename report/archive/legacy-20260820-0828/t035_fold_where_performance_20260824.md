# T-035：fold_where 单 pass NPU 性能

## 结论

`fold_where` 在本轮 fp16 contiguous 代表 shape 上功能可用，但端到端性能未达到项目的
10% 收益门槛，verdict 为 `supported-neutral`。candidate p50/p99 分别改善 1.16%/3.12%，
task 数与峰值显存不变；因此保留既有 pass，但不为该场景手写 Triton，也不把它计为性能
优化成功。

## 测试合同

- 输入：bool contiguous `mask=(2048,2048)`、fp16 contiguous `x=(2048,2048)`。
- 图：直接输出 `where(mask,x,x)`。
- baseline：只跳过 `fold_where`，图门禁 where `1→1`、clone `0→0`。
- candidate：执行当前产品 pass，图门禁 where `1→0`、clone `0→1`。
- 顺序：`B1,C1,C2,B2,B3,C3`；每 worker 使用 fresh process/cache，warmup 10、
  runs 100；B1/C1 另采 warmup 1、active 10 的 NPU profile。
- 环境：PyTorch `2.14.0a0+git8e86e0a`、torch_npu
  `2.14.0a0+git83cc452` source-built wheel、CANN 9.0.1、Ascend910B2、物理 NPU 1。
  设备在批次前后均无外部进程。

## 三轮汇总

| 指标 | baseline | candidate | 改善 |
|---|---:|---:|---:|
| mean 中位数 | `0.249119 ms` | `0.246128 ms` | `1.20%` |
| stdev 中位数 | `0.005972 ms` | `0.005860 ms` | — |
| p50 中位数 | `0.247985 ms` | `0.245115 ms` | `1.16%` |
| p99 中位数 | `0.274870 ms` | `0.266300 ms` | `3.12%` |
| additional allocated peak | `8,389,120 B` | `8,389,120 B` | `0 B` |
| additional reserved peak | `0 B` | `0 B` | `0 B` |

三轮 p50 分别为：baseline `0.261525/0.247985/0.244900 ms`，candidate
`0.234040/0.245115/0.245830 ms`。全部 worker 的测量前后 max/mean absolute error 为 0，
且 shape、dtype、stride、相对输入的 storage alias、对象身份和 `requires_grad` 与 eager
一致。

## NPU task 归因

首轮 10 个 active step 中，两侧都是每 step 1 个 Triton task：

- baseline：`triton_poi_fused_where_0`，10 个 task 的 device duration 合计
  `97.14 μs`，均值 `9.714 μs/task`。
- candidate：`triton_poi_fused_0`，合计 `71.44 μs`，均值 `7.144 μs/task`。

candidate 的 device kernel 时间下降约 26.46%，但没有减少 task 数或输出分配；在约
`0.25 ms` 的同步端到端测量里，host/launch 固定开销掩盖了大部分 kernel 收益。项目验收看
端到端 p50/p99 与资源，而不是只看 kernel 自身，因此最终仍为 `supported-neutral`。

baseline 编译路径反复出现 Triton 的 `tl.where` int8 condition 弃用 warning；它没有导致
功能、图门禁或 worker 状态失败，但应作为未来 Triton lowering 兼容性事项独立处理，不能
算作本 pass 的性能收益或 blocker。

原始 worker、profile 与聚合结果位于
`results/t035_fold_where_performance_20260824/`。
