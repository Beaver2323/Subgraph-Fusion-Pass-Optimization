# T-033：fold_cat 单 pass NPU 性能

## 结论

`fold_cat` 在本轮 fp16 contiguous 代表 shape 上通过功能、p50、p99、任务数和显存门禁，
verdict 为 `supported-beneficial`。它把同维、单用户 nested cat 从两个 `aten.cat` 展平为
一个；无需修改产品源码，也无需手写 Triton。

## 测试合同

- 输入：`a=(2048,256)`、`b=(2048,256)`、`c=(2048,512)`，fp16 contiguous。
- 图：`cat([cat([a,b], dim=1), c], dim=1)`，输出 `(2048,1024)`。
- baseline：只跳过 `fold_cat`，图门禁 cat `2→2`。
- candidate：调用当前产品 `fold_cat`，图门禁 cat `2→1`。
- 顺序：`B1,C1,C2,B2,B3,C3`；每 worker 为 fresh process/cache，warmup 10、
  runs 100；B1/C1 另采 warmup 1、active 10 的 NPU profile。
- 环境：PyTorch `2.14.0a0+git8e86e0a`、torch_npu
  `2.14.0a0+git83cc452` source-built wheel、CANN 9.0.1、Ascend910B2、物理 NPU 1。
  设备在批次前后均无外部进程。

## 三轮汇总

| 指标 | baseline | candidate | 改善 |
|---|---:|---:|---:|
| mean 中位数 | `0.300154 ms` | `0.269550 ms` | `10.20%` |
| stdev 中位数 | `0.006694 ms` | `0.005401 ms` | — |
| p50 中位数 | `0.298020 ms` | `0.267790 ms` | `10.14%` |
| p99 中位数 | `0.322720 ms` | `0.289430 ms` | `10.32%` |
| additional allocated peak | `6,292,480 B` | `4,194,816 B` | `-2,097,664 B` |
| additional reserved peak | `0 B` | `0 B` | `0 B` |

三轮 p50 分别为：baseline `0.298020/0.297730/0.318815 ms`，candidate
`0.254195/0.267790/0.276415 ms`。全部 worker 在测量前后均为 max/mean absolute error 0，
shape、dtype、stride、相对三个输入的 storage alias、对象身份和 `requires_grad` 与 eager
一致。

## NPU task 归因

首轮 10 个 active step 中：

- baseline：20 个 `aclnnCat_ConcatD_ConcatD`，即每 step 2 个，device duration 合计
  `95.94 μs`。
- candidate：10 个同名 task，即每 step 1 个，device duration 合计 `56.16 μs`。

因此端到端收益与图改写一致：少一次 cat、少一个约 2 MiB 的中间输出，而不是来自未知
fallback。当前结论仅覆盖本轮 shape/dtype/layout；扩大 verdict 前仍需 bf16/fp32、非连续
输入、dynamic width 和更深 nested cat 的扩展验证。

原始 worker、profile 与聚合结果位于
`results/t033_fold_cat_performance_20260824/`。
