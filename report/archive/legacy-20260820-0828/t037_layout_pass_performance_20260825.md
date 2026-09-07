# T-037：cat-slice-cat / pad-slice 三轮 paired 性能

## 结论

T-037 对 T-036 已修复并通过完整 alias 合同的两个 safe positive 做单 pass A/B。
`cat_slice_cat_fold_pass` 与 `pad_slice_fold` 均达到预登记的
`supported-beneficial` 门槛：三轮中位 p50 分别改善 24.00% 和 31.35%，p99 分别改善
22.87% 和 30.34%，task 均减少且 additional allocated peak 不增加。现有 FX pass 已能
获得收益，不需要手写 Triton 替身。

本结论只覆盖下表登记的 fp16 contiguous、dynamic/fullgraph、forward/no-grad safe cohort。
T-036 的保守 alias guard 必须保留；不能为了扩大触发范围重新引入可观察 storage/stride
错误。

## 环境与方法

- PyTorch `2.14.0a0+git8e86e0a`，torch_npu `2.14.0a0+git83cc452` T-036
  source-built wheel，CANN 9.0.1，Ascend910B2，物理 NPU 1。
- 所有 worker 从 `/home/z50063656/tmp` 启动；测试前后 NPU 1 无其他进程。
- default backend、`dynamic=True`、`fullgraph=True`、`torch.no_grad()`；每个 worker 为
  fresh Python process 与独立 Inductor/Triton/debug cache。
- baseline 只在 PRE registry wrapper 中跳过目标 pass；candidate 执行同一安装 wheel 的
  目标 pass。执行顺序为 `B1,C1,C2,B2,B3,C3`。
- 每轮 warmup 10、runs 100；显存 warmup 3、runs 10。B1/C1 另采 profiler warmup 1、
  active 10。
- 每轮测量前后检查数值、shape、dtype、stride、`requires_grad`、相对输入 storage alias、
  Python 对象身份与跨输出 alias；12 个 worker 全部通过，max/mean absolute error 均为 0。
- beneficial gate：p50 改善严格超过 10%、p99 不回退超过 5%、allocated peak 不增加。

## `cat_slice_cat_fold_pass`

输入为两个 fp16 contiguous `(2048,512)` tensor；图先拼成 `(2048,1024)`，再以两个
连续完整 slice 重新 cat。baseline 保持 cat/getitem `2→2/2→2`，candidate 为
`2→1/2→0`。

| 指标（三轮中位） | baseline | candidate | 改善/差值 |
|---|---:|---:|---:|
| mean ± stdev | `0.318666 ± 0.006658 ms` | `0.241956 ± 0.004486 ms` | mean `+24.07%` |
| p50 | `0.317155 ms` | `0.241035 ms` | `+24.00%` |
| p99 | `0.333430 ms` | `0.257190 ms` | `+22.87%` |
| additional allocated peak | `4,194,816 B` | `4,194,816 B` | `0 B` |
| additional reserved peak | `0 B` | `0 B` | `0 B` |
| task / active step | `2` | `1` | `-1` |
| 10 active steps device duration | `107.80 μs` | `56.04 μs` | `+48.01%` |

三轮 p50 改善分别为 `15.94%/18.60%/35.43%`，p99 改善分别为
`14.86%/19.66%/56.49%`，都没有回退。B3 baseline 出现 `1.30112 ms` 最大值，导致该轮
stdev `0.101827 ms`、p99 `0.59110 ms`，明显比另两轮噪声大；因此最终使用预登记的三轮
中位而不是选择该轮。即便排除 B3，前两轮 p50 仍分别超过 10% 门槛。

B1 profile 每 step 是一个 `aclnnCat` 加一个 slice/cat Triton task；C1 只剩一个
`aclnnCat`。candidate additional peak 与纯输出分配相同，说明被删除的中间结果没有扩大
本轮测得的运行峰值。

首次 compile+run 三轮中位为 baseline `24,264.42 ms`、candidate `2,778.67 ms`；
baseline 需要为第二段生成 Triton launcher。该一次性差异仅作为编译开销记录，没有进入
稳态 beneficial 判定。

## `pad_slice_fold`

输入为 fp16 contiguous `(2048,2048)`；末维右 pad 256 后只切回原数据区，并接 `relu`
物化新 storage。baseline 保持 pad/getitem `1→1/1→1`，candidate 为 `1→0/1→1`。

| 指标（三轮中位） | baseline | candidate | 改善/差值 |
|---|---:|---:|---:|
| mean ± stdev | `0.365093 ± 0.006282 ms` | `0.250833 ± 0.004924 ms` | mean `+31.30%` |
| p50 | `0.363850 ms` | `0.249770 ms` | `+31.35%` |
| p99 | `0.383640 ms` | `0.267240 ms` | `+30.34%` |
| additional allocated peak | `18,874,368 B` | `8,389,120 B` | `-10,485,248 B` |
| additional reserved peak | `0 B` | `0 B` | `0 B` |
| task / active step | `3` | `1` | `-2` |
| 10 active steps device duration | `487.86 μs` | `105.30 μs` | `+78.42%` |

三轮 p50 改善分别为 `28.24%/27.29%/33.29%`，p99 改善分别为
`26.84%/26.92%/32.67%`，三轮一致。B1 profile 中 baseline 每 step 包含 PadV3、MemSet、
slice+relu Triton 三个 task；candidate 只剩 fused relu Triton task。candidate peak 恰好是
逻辑输出分配 `8,389,120 B`，baseline 还承担 padding 中间 storage。

首次 compile+run 三轮中位为 baseline `24,988.27 ms`、candidate `24,868.66 ms`，两侧
相近；它们没有进入稳态 verdict。

## 判断与替代方案

两条 pass 的收益都来自删除明确冗余的 copy/pad 和 task，不是设备算子缺失：

- cat 场景复用第一个 cat 输出，少一次切片重组；
- pad 场景在后续必然物化新 storage 时，直接从原输入计算 relu，删除 padding buffer、
  PadV3 与 MemSet。

因此最合适的实现仍是 T-036 修复后的 FX pass。为相同工作再写 Triton只会重复当前
Inductor 已生成的 pointwise/copy kernel，且无法代替 alias capability gate。矩阵可把两条
记录从 `not-run` 提升为 `supported-beneficial`，replacement 均为
`existing-pass-verified-no-replacement`。

原始 worker、profiler 与聚合结果位于
`results/t037_layout_pass_performance_20260825/`；其中两个 `aggregate.json` 是 verdict
入口，单轮 JSON 用于审查波动、图门禁与语义合同。
