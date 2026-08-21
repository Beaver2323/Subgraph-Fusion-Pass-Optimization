# P0 代表覆盖性能矩阵（2026-08-20 至 2026-08-21）

## 当前结论

两个 pass 的 dtype、shape、transposed layout 和 dynamic cohort 共 96/96 `compile-correct`，dynamic 的 12/12 worker 都完成第二 shape replay；mm_plus_mm dynamic 又完成 6/6 runs 300 独立复核。addmm 的 8 个代表配置全部超过 10% p50 收益；mm_plus_mm 的 8 个配置中 6 个超过门槛，transposed 为 6.4%，dynamic 高样本复核为 8.74%，两者属于“功能可用但性能 neutral”的代表配置。没有配置出现 p50 回退。

| pass/backend | dtype | current p50 中位数 | disabled p50 中位数 | p50 延迟下降 | p99 延迟下降 |
|---|---|---:|---:|---:|---:|
| addmm fusion/default | fp16 | 0.245615 ms | 0.292720 ms | 16.1% | 12.4% |
| addmm fusion/default | bf16 | 0.256425 ms | 0.295280 ms | 13.2% | 12.5% |
| addmm fusion/default | fp32 | 0.250550 ms | 0.291505 ms | 14.0% | 12.2% |
| mm_plus_mm/experimental | fp16 | 0.272185 ms | 0.311030 ms | 12.5% | 9.9% |
| mm_plus_mm/experimental | bf16 | 0.261255 ms | 0.312805 ms | 16.5% | 19.1% |
| mm_plus_mm/experimental | fp32 | 0.279350 ms | 0.313920 ms | 11.0% | 9.6% |
| addmm fusion/default | fp16 small | 0.248640 ms | 0.296220 ms | 16.1% | 15.6% |
| addmm fusion/default | fp16 unaligned | 0.249810 ms | 0.298825 ms | 16.4% | 10.5% |
| addmm fusion/default | fp16 large | 0.248930 ms | 0.296935 ms | 16.2% | 18.1% |
| mm_plus_mm/experimental | fp16 small | 0.270065 ms | 0.303390 ms | 11.0% | 22.6% |
| mm_plus_mm/experimental | fp16 unaligned | 0.276050 ms | 0.317490 ms | 13.1% | 3.5% |
| mm_plus_mm/experimental | fp16 large | 0.284715 ms | 0.319755 ms | 11.0% | 13.7% |
| addmm fusion/default | fp16 transposed | 0.252995 ms | 0.299155 ms | 15.4% | 18.1% |
| mm_plus_mm/experimental | fp16 transposed | 0.283075 ms | 0.302365 ms | 6.4% | 1.6% |
| addmm fusion/default | fp16 dynamic | 0.265665 ms | 0.305795 ms | 13.1% | 16.5% |
| mm_plus_mm/experimental | fp16 dynamic | 0.298015 ms | 0.329350 ms | 9.51% | 0.42% |

## addmm dtype 三轮明细

| dtype/mode | p50 三轮（ms） | p99 三轮（ms） | 首次编译+执行（s） |
|---|---|---|---|
| fp16 current | 0.245030 / 0.245615 / 0.287255 | 0.265850 / 0.273520 / 0.790580 | 13.830 / 12.865 / 13.060 |
| fp16 disabled | 0.294240 / 0.278850 / 0.292720 | 0.313450 / 0.295350 / 0.312320 | 19.647 / 19.480 / 20.474 |
| bf16 current | 0.260120 / 0.256425 / 0.248835 | 0.283300 / 0.291230 / 0.271010 | 13.171 / 13.916 / 13.808 |
| bf16 disabled | 0.295280 / 0.296340 / 0.289935 | 0.386200 / 0.323610 / 0.303830 | 19.663 / 19.686 / 20.330 |
| fp32 current | 0.251735 / 0.242730 / 0.250550 | 0.280800 / 0.258570 / 0.267930 | 13.641 / 13.641 / 13.669 |
| fp32 disabled | 0.291505 / 0.287345 / 0.298585 | 0.304960 / 0.304990 / 0.319410 | 19.963 / 20.110 / 20.231 |

fp16 current 第 3 轮 p99 为 0.790580 ms，明显高于同模式另两轮；该尾部抖动保留在原始数据中。p50、p99 的汇总结论使用三轮中位数，不删除该轮。

current 的编译期峰值 allocator：fp16/bf16 为 1,249,792 bytes，fp32 为 1,511,936 bytes；disabled 分别为 201,961,472、201,961,472、202,592,768 bytes。reset 位于编译前，disabled 又触发 Triton 编译/autotune，因此这不是纯 runtime 内存。

## mm_plus_mm dtype 三轮明细

| dtype/mode | p50 三轮（ms） | p99 三轮（ms） | 首次编译+执行（s） |
|---|---|---|---|
| fp16 current | 0.261565 / 0.272185 / 0.282505 | 0.286010 / 0.300380 / 0.300510 | 14.182 / 14.137 / 14.005 |
| fp16 disabled | 0.311030 / 0.298095 / 0.311855 | 0.337610 / 0.318860 / 0.333280 | 20.174 / 20.021 / 19.444 |
| bf16 current | 0.261255 / 0.274710 / 0.256865 | 0.286510 / 0.301590 / 0.285380 | 14.009 / 13.960 / 14.125 |
| bf16 disabled | 0.317105 / 0.311005 / 0.312805 | 0.353990 / 0.528390 / 0.344830 | 20.107 / 19.915 / 19.930 |
| fp32 current | 0.279350 / 0.279875 / 0.268420 | 0.303340 / 0.310700 / 0.300120 | 14.074 / 13.928 / 13.225 |
| fp32 disabled | 0.313920 / 0.315050 / 0.304530 | 0.337070 / 0.335640 / 0.332930 | 19.646 / 19.879 / 19.139 |

mm_plus_mm current 的编译期峰值 allocator：fp16/bf16 为 1,511,936 bytes，fp32 为 2,035,712 bytes；disabled 分别为 2,253,824、2,253,824、4,498,432 bytes。bf16 disabled 第 2 轮 p99 为 0.528390 ms，原始尾部抖动保留。fp16/fp32 的 p99 中位数改善为 9.9%/9.6%，低于 10% 但没有回退；最终判定以 p50 主门槛和完整覆盖共同决定。

## addmm shape 三轮摘要

| shape | current p50 三轮（ms） | disabled p50 三轮（ms） | p50 延迟下降 | p99 延迟下降 |
|---|---|---|---:|---:|
| small | 0.248640 / 0.270510 / 0.246500 | 0.296290 / 0.296220 / 0.286900 | 16.1% | 15.6% |
| unaligned | 0.257245 / 0.247730 / 0.249810 | 0.314680 / 0.298825 / 0.297835 | 16.4% | 10.5% |
| large | 0.248930 / 0.252585 / 0.238940 | 0.296935 / 0.299190 / 0.289190 | 16.2% | 18.1% |

small disabled 三轮首次编译+执行为 35.539/35.016/35.535 s，明显高于该 cohort 其余 13.648-22.314 s；原始值保留，尚未将其归因为 pass 本身或工具链缺陷。unaligned disabled 第 1 轮 p99 为 1.222570 ms，三轮 p99 中位数仍为 0.315230 ms。

## mm_plus_mm shape 三轮摘要

| shape | current p50 三轮（ms） | disabled p50 三轮（ms） | p50 延迟下降 | p99 延迟下降 |
|---|---|---|---:|---:|
| small | 0.270065 / 0.269540 / 0.275135 | 0.317425 / 0.303390 / 0.300410 | 11.0% | 22.6% |
| unaligned | 0.303545 / 0.276050 / 0.270370 | 0.317490 / 0.306880 / 0.318760 | 13.1% | 3.5% |
| large | 0.286455 / 0.280570 / 0.284715 | 0.335765 / 0.314280 / 0.319755 | 11.0% | 13.7% |

unaligned current 三轮 p99 为 0.391930/0.304470/0.330510 ms，disabled 为 0.342500/0.434550/0.339270 ms；三轮中位数只改善 3.5%。这不是 p50 回退，但表明该 shape 的尾延迟收益弱，应在更高样本量或业务模型中复核。

## transposed layout 三轮摘要

| pass/backend | current p50 三轮（ms） | disabled p50 三轮（ms） | p50 延迟下降 | p99 延迟下降 |
|---|---|---|---:|---:|
| addmm/default | 0.242240 / 0.269510 / 0.252995 | 0.300925 / 0.299155 / 0.297030 | 15.4% | 18.1% |
| mm_plus_mm/experimental | 0.273845 / 0.283075 / 0.287095 | 0.302365 / 0.297280 / 0.305740 | 6.4% | 1.6% |

两条路径都已在功能矩阵中确认目标图，差别来自性能而不是未触发。mm_plus_mm/transposed 的 p50 改善低于项目当前 10% 准入门槛，应记为该 layout 上的 `supported-neutral` 证据；不能用 contiguous 配置的收益覆盖它。addmm disabled 第 1 轮 p99 0.788800 ms 作为尾部抖动保留。

## dynamic replay 三轮摘要

| pass/backend | current p50 三轮（ms） | disabled p50 三轮（ms） | p50 延迟下降 | p99 延迟下降 | mean 变化 |
|---|---|---|---:|---:|---:|
| addmm/default | 0.265665 / 0.260440 / 0.268030 | 0.303965 / 0.305795 / 0.324050 | 13.1% | 16.5% | 改善 12.7% |
| mm_plus_mm/experimental | 0.295565 / 0.310265 / 0.298015 | 0.325550 / 0.351785 / 0.329350 | 9.51% | 0.42% | 回退 2.89% |

两组均使用 shape-A 首次输入和 M/K/N 各增加 8 的第二组 replay，6/6 current/disabled 都包含 replay 正确性。addmm dynamic 可记为代表配置正收益；mm_plus_mm dynamic 的 p50 接近 10% 但 p99 几乎 neutral，current 第 1 轮 p99 0.476890 ms 又使 mean 中位数回退 2.89%。按项目规则需要增加样本复核，不直接升级为 beneficial。

### mm_plus_mm dynamic 高样本复核

独立使用 warmup 20、runs 300、current/disabled 三轮交错复核，6/6 正确且包含第二 shape replay：

| mode | p50 三轮（ms） | p99 三轮（ms） | mean 三轮（ms） |
|---|---|---|---|
| current | 0.291220 / 0.306045 / 0.316245 | 0.314200 / 0.332990 / 0.350230 | 0.294312 / 0.308666 / 0.318637 |
| disabled | 0.335340 / 0.330810 / 0.346265 | 0.353480 / 0.351560 / 0.368810 | 0.344181 / 0.332144 / 0.348447 |

三轮中位数的 p50 改善 8.74%，p99 改善 5.80%，mean 改善 10.32%。项目主门槛是稳态 p50 至少 10%，因此高样本结果确认 dynamic 配置为 `supported-neutral` 证据，而不是把原 runs 100 的 9.51% 四舍五入为 beneficial。

## 代表网格结论

| pass/backend | 功能 | p50 >10% | p50 0%-10% | p50 回退 | 当前解释 |
|---|---:|---:|---:|---:|---|
| addmm/default | 8/8 | 8 | 0 | 0 | 代表网格稳定正收益；仍缺 bias 语义变体与 backward，pass 最终 verdict 暂不关闭 |
| mm_plus_mm/experimental | 8/8 | 6 | 2 | 0 | contiguous static dtype/shape 有益；transposed 与 dynamic 可用但 neutral；default backend 仍保持 gate |

## 方法与证据

- 物理 NPU 2；采样前无运行进程。
- 同 backend、同输入、fresh worker、`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`。
- warmup 10、runs 100、三轮；第 2 轮反转 current/disabled 顺序。
- disabled 每轮均确认临时 patch 2 个 addmm pattern entry。
- 原始 JSON：`results/p0_sweep_perf_addmm_dtype_20260820/p0_gate_probe.json`。
- mm_plus_mm disabled 每轮确认 patch 1 个目标 entry；原始 JSON：`results/p0_sweep_perf_mmplus_dtype_20260820/p0_gate_probe.json`。
- addmm shape 原始 JSON：`results/p0_sweep_perf_addmm_shape_20260820/p0_gate_probe.json`。
- mm_plus_mm shape 原始 JSON：`results/p0_sweep_perf_mmplus_shape_20260820/p0_gate_probe.json`。
- layout 原始 JSON：`results/p0_sweep_perf_{addmm,mmplus}_layout_20260820/p0_gate_probe.json`。
- dynamic 原始 JSON：`results/p0_sweep_perf_{addmm,mmplus}_dynamic_20260821/p0_gate_probe.json`。
- mm_plus_mm dynamic runs 300：`results/p0_sweep_perf_mmplus_dynamic_retest300_20260821/p0_gate_probe.json`。

## 设备闭环与剩余边界

所有扩展性能采样使用物理 NPU 2。dtype/shape/layout/dynamic 主矩阵前、dynamic 高样本复核前和全部结束后，`npu-smi info` 均显示该卡无运行进程；没有终止或混用 NPU 0/1/5/7 上的外部任务。

本报告完成的是两个正例 family 的代表性能网格，不替代语义覆盖。报告生成当时，addmm 仍需 `(M,N)`、`(1,N)` bias、dtype mismatch、训练/backward，mm_plus_mm 仍需不同 K、更多负例、训练/backward，因此当时只升级 performance status，final verdict 暂保持 `not-run`。

后续状态（2026-08-21）：上述语义与 backward 覆盖已完成；T-011 关闭 torch_npu reduction `strict_sum` 接口 blocker 后，addmm 最终 verdict 已升级为 `supported-beneficial`。mm_plus_mm 因 default backend gate 仍保持 `not-run`。当前结论以 `report/p0_semantic_matrix_20260821.md` 和评估矩阵为准。
