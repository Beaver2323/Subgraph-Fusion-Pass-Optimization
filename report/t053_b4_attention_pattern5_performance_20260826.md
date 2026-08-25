# T-053：B4 attention pattern 5 math re-expansion 配对性能报告

## 结论

`_sfdp_pattern_5_half_inference` 在 `(4,8,128,64)` fp16 Q/K/V 和
`(1,1,128,128)` fp16 additive mask 上功能正确，但当前 NPU 默认 pass 明显变慢，最终记为
`supported-performance-regressed`。

baseline 保留原始 attention 子图，Inductor 生成 `2×BMM + 1×Triton`；candidate 的 exact
matcher 和 `fuse_attention` counter 都为 1，但 SDPA dispatcher 因 float mask 不能进入 vendor
branch，重新展开成 `2×BMM + 6×Triton`。这不是算子不支持或数值失败，而是“通用融合先发生、
设备端又安全分解”造成的重复转换和额外 kernel。

## 隔离与正确性

baseline 只把 `_sfdp_pattern_5_half_inference` 的一个 entry 临时置为 false。短 smoke 和六个
正式 worker 都确认没有等价 matcher 接管：

- baseline：patched entry 1、exact/总 fusion counter `0/0`；
- candidate：patched entry 0、exact/总 fusion counter `1/1`；
- 两侧最终都含 2 个 `aclnnBatchMatMul`，且都没有 `npu_fusion_attention_v3`；
- 六轮输出 shape/dtype/finite 正确，`atol=rtol=0.02` 全部通过，最大绝对误差均为
  `0.0029296875`。

因此本轮量到的是 pattern 5 rewrite 与随后 NPU math decomposition 的净成本，不是 vendor
FlashAttention 和普通 attention 的比较。

## 三轮配对结果

执行顺序 `B1,C1,C2,B2,B3,C3`；每个 worker 使用独立 fresh cache、warmup 10、runs 100，
两侧统一使用 T-022 audit-only launcher/header wrapper。

| 指标（三轮中位） | baseline：原图 math | candidate：SDPA 后重展开 | 变化 |
|---|---:|---:|---:|
| P50 | 0.381385 ms | 0.775105 ms | 回退 103.23% |
| P99 | 0.409750 ms | 0.824400 ms | 回退 101.20% |
| mean | 0.383198 ms | 0.779040 ms | 回退 103.30% |
| 首次 compile+run | 41,807.41 ms | 100,682.28 ms | 回退 140.82% |
| additional allocated peak | 204,472,832 B | 205,527,552 B | 增加 1,054,720 B（0.52%） |
| additional reserved peak | 224,395,264 B | 224,395,264 B | 不变 |
| device task/step | 3 | 8 | 增加 166.67% |

P50、P99 和 allocated peak 三个预登记 gate 全部失败。三轮 baseline P50 为
`0.375850/0.387835/0.381385 ms`，candidate 为
`0.815880/0.766440/0.775105 ms`，交错顺序下方向完全一致。

## 为什么 pass-on 会多五个 Triton task

原图允许 scheduler 把 scale、mask add 和 softmax 合并成一个 reduction kernel。通用 pattern
先把它替换成 SDPA 后，torch_npu dispatcher 为保持任意 additive bias 语义调用 math fallback；
decomposition 又显式产生 Q/K/V cast/scale/transpose、safe-softmax 辅助和输出 cast。当前生成代码
因此包含：

1. Q scale/cast；
2. K scale/cast/transpose；
3. 第一个 BMM；
4. safe-softmax pointwise；
5. safe-softmax reduction；
6. V cast；
7. 第二个 BMM；
8. 输出 cast。

原图只有两个 BMM 和一个融合 reduction。这里最有价值的优化不是再写一份完整 Triton
attention，而是避免对无法进入高效设备 kernel 的输入做无收益 SDPA rewrite。

## 处理方向

- 首选：在 torch_npu default backend 对已验证的
  `_sfdp_pattern_5_half_inference` + NPU scope 增加保守 capability/performance guard，使它保持
  原图；CPU/CUDA/XPU 和其他 attention family 不变。
- 后续：只有能证明某个 mask 子域可无损映射到 bool，或 vendor `pse` 可完整表达 additive bias
  时，才单独开放 vendor path，并覆盖广播、有限 bias、0、`-inf`、NaN、dtype、dynamic 与
  backward。
- 不采用：把任意 float mask 直接转 bool；这会改变“加性偏置”为“允许/屏蔽”，属于语义错误。
- 暂不采用：手写完整 Triton attention。当前原图已经是更少 task、更低延迟和更低内存的
  Triton math baseline，复制完整 attention 缺少收益依据。

## 证据与中性尝试

- 第一次 smoke 的 baseline 已完成 compile/profile，但 debug 子目录先创建了 worker 目录，审计
  脚本最终再次 `mkdir` 导致 `FileExistsError`，未写 `result.json`；candidate 未启动。目录
  `results/t053_b4_attention_pattern5_isolation_smoke_20260826/B0` 保留，不计产品失败。
- 修正脚本为入口拒绝已存在输出、结束时允许本轮 debug 已创建父目录后，在
  `results/t053_b4_attention_pattern5_isolation_smoke_retry_20260826/{B0,C0}` 完成有效隔离。
- 正式 workers：
  `results/t053_b4_attention_pattern5_performance_20260826/{B1,B2,B3,C1,C2,C3}/result.json`。
- 聚合：
  `results/t053_b4_attention_pattern5_performance_20260826/aggregate/aggregate.json`。

本轮没有修改产品源码，NPU1 测试前后均无其他进程。
