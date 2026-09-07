# T-051：B4 attention pattern 13 配对性能报告

## 结论

`_sfdp_pattern_13_half_inference` 在 `(32,128,64)` fp16 三维 BMM attention 上精确触发并落到
单个 `aclnnFlashAttentionScore`。它没有达到 10% 稳态延迟收益门槛，但显著减少 task、编译时间
和峰值内存，因此记 `supported-neutral-resource-beneficial`，不是 `supported-beneficial`。

本轮没有修改产品源码，也没有新增 Triton kernel。已有 vendor attention 是可用实现；短 shape
上的稳态耗时被 host/launch 固定开销主导，不能用资源收益冒充端到端 latency 收益。

## 隔离与正确性

pattern 13 是 inference-only 三维 BMM family。测试函数保持上游结构：

```text
bmm(Q, K.transpose) -> softmax -> dropout(training=False) -> bmm(..., V)
```

baseline 只关闭 `_sfdp_pattern_13_half_inference` 的一个 entry。短 smoke 与六个正式 worker 均
确认没有其他 attention matcher 接管：baseline exact/总 fusion counter 为 0，generated code 为
2 个 BMM + 1 个 Triton softmax；candidate counter 为 1，generated code 和 profiler 为单个 NPU
FlashAttention。

六轮输出 shape `(32,128,64)`、dtype fp16、finite 均正确，`atol=rtol=0.02` 全部通过。baseline
最大绝对误差为 `0.001953125`，candidate 为 `0.01318359375`；后者低于预登记容差，但也说明
vendor fusion 与逐步数学路径的浮点舍入顺序不同，后续 dtype/shape 扩展仍需保留精度门禁。

## 三轮配对结果

执行顺序为 `B1,C1,C2,B2,B3,C3`；每个 fresh process 使用 warmup 10、runs 100，两侧统一使用
T-022 audit-only launcher/header wrapper。

| 指标（三轮中位） | baseline：2 BMM + Triton softmax | candidate：1 FlashAttention | 变化 |
|---|---:|---:|---:|
| P50 | 0.335175 ms | 0.331850 ms | 改善 0.99% |
| P99 | 0.363240 ms | 0.363420 ms | 回退 0.05% |
| mean | 0.338153 ms | 0.333533 ms | 改善 1.37% |
| 首次 compile+run | 34,134.41 ms | 2,944.29 ms | 改善 91.37% |
| additional allocated peak | 204,472,832 B | 25,955,328 B | 减少 87.31% |
| additional reserved peak | 224,395,264 B | 27,262,976 B | 减少 87.85% |
| device task/step | 3 | 1 | 减少 66.67% |

P50 主门槛未通过；P99 回退小于 5%，allocated peak 未增加，task 和内存明确减少。按项目既有
分层，最终 verdict 为 `supported-neutral-resource-beneficial`。

## 如何理解 pattern 1 与 pattern 13 的不同结果

两者最终都调用一个 vendor FlashAttention，但 pass-off baseline 不同、输入语义也不同：

- pattern 1 baseline 包含 scale/isfinite/cast 等 2 个 BMM + 2 个 Triton task，P50 改善 46.70%；
- pattern 13 baseline 只有 2 个 BMM + 1 个 softmax task，且 3D 结构的测试规模较短，P50 只改善
  0.99%。

因此“同一个目标 kernel”不保证每个 pattern family 的端到端收益相同。pass verdict 必须按
family、shape、dtype 和 baseline 单独测，不能从 pattern 1 外推 pattern 13。

## 证据

- 隔离 smoke：`results/t051_b4_attention_pattern13_smoke_20260826/{B,C}/result.json`
- 正式 workers：`results/t051_b4_attention_pattern13_performance_20260826/{B1,B2,B3,C1,C2,C3}/result.json`
- 聚合：`results/t051_b4_attention_pattern13_performance_20260826/aggregate/aggregate.json`
- 启动命令首次把 `set -e` 放在 `env.sh` 之前而提前退出，未创建 worker、未占用 NPU；属于中性
  审计命令错误，不进入 pass 结论。

## 下一步

继续 B4 的能力分流：先定位 pattern 5/21/29 为什么在 matcher 后被 NPU dispatcher re-expand，
再选择其中一个做 paired。pattern 18/28 的辅助 logical-not/clone 也应单独 profile；当前不为
pattern 13 写 Triton 替身。
