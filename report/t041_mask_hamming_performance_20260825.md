# T-041：mask arithmetic 与 sign-Hamming 单 pass paired 性能

## 结论

T-040 三条功能通过的 pass 共执行 4 个 representative case、24/24 个 fresh-process
worker。所有 worker 的目标图门禁和测量前后完整 tensor contract 通过。结果不是“三条都因
删节点而更快”，而是明显分流：

- `masked_add_compose_pass`：p50 改善 3.71%，task/显存不变，`supported-neutral`。
- `sign_diff_hamming_fuse_pass`：p50 改善 3.64%，task/显存不变，`supported-neutral`。
- `bool_cast_mul_to_where_pass` direct：p50 回退 0.69%，p99 回退 19.02%，device profile
  也变慢，`supported-performance-regressed`。
- 同一 bool-cast pass 的 view-chain：p50/p99 改善 36.30%/39.90%，
  `supported-beneficial`。

因此后续 P-009 只保留 view-chain 改写，direct cast×x 保持原图。中性的 masked-add 与
sign-Hamming 保留现有安全语义，但不为它们手写功能重复的 Triton kernel。

## 方法

- 环境：`benchmark-py311`、PyTorch `2.14.0a0+git8e86e0a`、T-040 source-built
  torch_npu wheel、Triton 3.2.0、CANN 9.0.1、Ascend 910B2。
- 物理 NPU1，运行前后无其他进程；default backend、inference、static fullgraph。
- baseline 与 candidate 加载同一 wheel；registry wrapper 只让目标 pass 在 baseline 返回、
  candidate 正常执行。每个 worker 使用独立 output/cache。
- 每 case 三轮，顺序 `B-C / C-B / B-C`；每 worker warmup 10、runs 100、逐次同步；
  memory warmup3/runs10。第一轮两侧另做 warmup1/active10 NPU profile。
- 当前 launcher 仍使用 T-022 C++20/header audit shim，证据范围是
  `development-audit-shim`。首次 compile+run 只记录，不进入稳态 verdict。

主 gate 是三轮 p50 中位数改善严格超过 10%、p99 不回退超过 5%、allocated peak 不增加。
只有 task 数或峰值显存下降才算 resource benefit；一次 profiler duration 下降不能单独把
neutral 改成 beneficial。

## 聚合结果

| case | mean±stdev baseline→candidate (ms) | p50 (ms) | p50 | p99 (ms) | p99 | task/step | allocated peak delta | verdict |
|---|---|---|---:|---|---:|---|---:|---|
| masked-add int32 | 0.261700±0.005947 → 0.251876±0.005903 | 0.260785→0.251115 | +3.71% | 0.295460→0.266380 | +9.84% | 1→1 | 0 B | neutral |
| bool-cast direct int32 | 0.256894±0.006110 → 0.272659±0.046537 | 0.254935→0.256695 | -0.69% | 0.276030→0.328520 | -19.02% | 1→1 | 0 B | performance-regressed |
| bool-cast view int32 | 0.446580±0.015785 → 0.281878±0.005950 | 0.440025→0.280315 | +36.30% | 0.492800→0.296190 | +39.90% | 1→1 | 0 B | beneficial |
| sign-Hamming fp32 | 0.268474±0.006603 → 0.257735±0.004607 | 0.266480→0.256770 | +3.64% | 0.290000→0.274090 | +5.49% | 1→1 | 0 B | neutral |

三轮的 mean 与 stdev 取各 variant 的中位 worker；不是把 300 次调用混成一轮。四个 case
的 additional allocated peak 分别均为 4,194,816 B、4,194,816 B、4,194,816 B 与
4,608 B，candidate 没有增加或减少；reserved peak delta 均为 0。

## Profiler 解释

第一轮 active10 的 device duration 总和如下；每侧均为 10 个 kernel，即每 step 1 task：

| case | baseline→candidate active10 duration (μs) | 解释 |
|---|---:|---|
| masked-add | 81.10→64.18 | kernel 内部更短，但端到端 p50 仅 +3.71% |
| bool-cast direct | 62.56→73.86 | where kernel 比 direct cast×mul 融合 kernel 更慢，与回退方向一致 |
| bool-cast view | 2377.58→96.28 | 避免 cast 后的广播/view-chain 代价，端到端收益稳定超过 gate |
| sign-Hamming | 126.46→86.10 | 指令链缩短，但固定 host/launch 开销使端到端仅 +3.64% |

这再次说明 FX 节点数、device kernel duration 和端到端延迟是三层不同证据。原图已经都被
scheduler 融为单 task，因此 masked-add/Hamming 没有可由另一个 Triton kernel消除的 launch。

## bool direct 的尾延迟复核

direct 三轮 p50 baseline 为 0.254935/0.282490/0.252895 ms，candidate 为
0.256695/0.251055/0.279125 ms；中位 p50 基本中性。p99 baseline 为
0.275070/0.518770/0.276030 ms，candidate 为 0.328520/0.267410/0.532510 ms。交错顺序中的
第二 worker 确有共享环境尾延迟噪声，但 candidate 的 active10 device duration仍比 baseline
高约 18.1%，且 direct 路径没有任何 task/显存收益。因此无需用额外挑选轮次把它包装成
neutral；保守选择是停止 direct rewrite。

## 结果与下一步

四个聚合文件位于
`results/t041_mask_hamming_performance_20260825/<case>/aggregate/aggregate.json`。下一步按
P-009 在 `bool_cast_mul_to_where_pass` 增加“必须存在非空 view/expand chain”的 guard，更新
direct 结构测试为保持原图；view-chain 逻辑不变。随后重建/安装 wheel，完成 76/76 与
direct/view NPU 功能复验。只有该 guard 进入安装态后，bool-cast pass 才能按 view-chain
cohort 关闭为 beneficial。
