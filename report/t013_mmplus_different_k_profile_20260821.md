# T-013 mm_plus_mm different-K fallback kernel profile

## 结论

T-012 的 different-K current/disabled 端到端 p50 只有 -0.30%/+2.53% 差异，因为两种模式的实际 NPU task 组成完全相同：每次迭代固定执行两个 `aclnnMm_MatMulCommon_MatMulV2` 和一个 `triton_unk_fused_add_0`。四组正式 profile 均采集 10 个 active step，因此每组都是 20 条 matmul + 10 条 add，共 30 条 kernel 记录；输出误差全部为 0。

add 本身占纯 device kernel duration 的 13.70%–15.75%。更大的机会来自两个相邻 task gap：shape-A/current 每次迭代的两个步内 gap 合计 `55.15±3.88 μs`，unaligned/current 为 `49.59±6.27 μs`。它们与 add 一起形成约 17.7%/16.0% 的端到端理论可消除上限，超过项目 10% 门槛。因此 P-005 可以进入“不接入源码的微原型”阶段，但这仍不是候选已证明更快：手写 kernel 必须在保留两次 matmul 数值工作的同时，把 fused task 控制在约 34/31 μs 内。

## 环境与方法

- CANN 9.0.1；Ascend 910B2；物理 NPU 6。
- PyTorch `2.14.0a0+git8e86e0a`；torch_npu `2.14.0a0+git83cc452`；Triton Ascend metadata `3.2.2`。
- fp16、contiguous、static；shape-A `(192,256,320)` 且 `K2=128`，unaligned `(191,255,319)` 且 `K2=127`。
- current/disabled 都在导入 torch 前固定 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`；disabled 每次 patch 1 个目标 entry。
- 先编译、与 eager 比较并 warmup 10；随后使用 `torch_npu.profiler` 的 NPU activity、Level0/AiCoreNone，profile warmup 1、active 10，每 step 后同步。
- 原始 `kernel_details.csv` 保留；统计给出 mean±stdev、p50、p99。task gap 是相邻 NPU task 的时间差，步间 gap 包含显式 synchronize，报告的融合机会只使用每组三个 task 内部的两个 gap。

采样开始前、E-025 修正后扩展前和全部结束后，NPU 6 均无其他进程。

## Kernel 组成与 duration

| shape/mode | task | count | mean±stdev（μs） | p50（μs） | p99（μs） | 纯 kernel duration 占比 |
|---|---|---:|---:|---:|---:|---:|
| shape-A/current | aclnnMm | 20 | 4.611±0.353 | 4.650 | 5.280 | 84.51% |
| shape-A/current | Triton add | 10 | 1.690±0.089 | 1.720 | 1.780 | 15.49% |
| shape-A/disabled | aclnnMm | 20 | 4.553±0.381 | 4.530 | 5.100 | 84.25% |
| shape-A/disabled | Triton add | 10 | 1.702±0.045 | 1.700 | 1.760 | 15.75% |
| unaligned/current | aclnnMm | 20 | 5.892±0.256 | 5.900 | 6.340 | 85.86% |
| unaligned/current | Triton add | 10 | 1.940±0.176 | 1.940 | 2.220 | 14.14% |
| unaligned/disabled | aclnnMm | 20 | 5.922±0.247 | 5.970 | 6.260 | 86.30% |
| unaligned/disabled | Triton add | 10 | 1.880±0.107 | 1.860 | 2.060 | 13.70% |

current 与 disabled 的 kernel 名称、顺序和数量完全一致，duration 也处于同一分布。这与 T-012 的端到端 neutral 结果互相验证：现有 different-K pattern 只有匹配与安全 unfuse，没有生成专用融合 task。

## 每次迭代的 device timeline

| shape/mode | 三 kernel duration 合计 mean±stdev | 两个步内 gap 合计 mean±stdev | gap p50/p99 | 首 task 到 add 结束 span mean±stdev | span p50/p99 |
|---|---:|---:|---:|---:|---:|
| shape-A/current | 10.912±0.489 μs | 55.153±3.876 μs | 56.360/59.460 μs | 66.075±3.631 μs | 67.375/69.750 μs |
| shape-A/disabled | 10.808±0.365 μs | 55.844±6.772 μs | 51.785/70.260 μs | 66.675±6.725 μs | 62.250/80.750 μs |
| unaligned/current | 13.724±0.416 μs | 49.591±6.274 μs | 49.675/60.280 μs | 63.300±6.154 μs | 63.500/73.750 μs |
| unaligned/disabled | 13.724±0.315 μs | 52.131±6.035 μs | 49.460/62.010 μs | 65.850±6.218 μs | 63.250/76.250 μs |

shape-A/current 的第一、第二个 matmul 分别为 `4.820±0.258 μs` 和 `4.402±0.314 μs`；unaligned/current 为 `5.768±0.235 μs` 和 `6.016±0.221 μs`。不同 K 的两次 GEMM 工作都是真实存在的，不能通过删除第二个 matmul 来构造伪“融合收益”。

## 理论上限与微原型门槛

用 T-012 的 current p50 与本轮 current timeline 建立保守可行性估算：

| shape | T-012 current p50 | 步内 gap + add mean | 完全消除时的端到端上限 | 达到 10% 时 fused task 的最大允许 duration |
|---|---:|---:|---:|---:|
| shape-A | 321.445 μs | 56.843 μs | 17.68% | 33.93 μs |
| unaligned | 321.665 μs | 51.531 μs | 16.02% | 31.13 μs |

计算假设首个 task 之前和 add 之后的 host/wrapper/synchronize 开销不变，融合只把当前 `mm1 → mm2 → add` 的内部 timeline 替换成一个 task。它是候选是否值得写的预算，不是性能承诺。候选即使比两个 aclnnMm 的纯计算合计慢，只要 shape-A/unaligned 的单 task 分别稳定低于约 34/31 μs，仍有可能达到端到端 10%；最终必须由真实微原型 paired benchmark 决定。

## 源码约束

上游 `torch/_inductor/kernel/mm_plus_mm.py` 的模板只定义 `K1`，第二个循环也写成 `range(K1, ...)`，并复用基于 K1 的 stride/`EVEN_K` 假设。`tuned_mm_plus_mm()` 因而要求两对输入 size 分别相同，different-K 时主动返回两个 mm 加 add。微原型必须至少独立处理：

- `K1` 与 `K2` 两个循环边界；
- 两套尾块 mask/EVEN_K 判断；
- C/D stride 连续性不能引用 K1；
- fp16/bf16/fp32 accumulation 与输出转换；
- 非对齐、转置、dynamic 和 backward 的 capability 边界。

不得只删除 size guard 或把第二个循环的范围机械改为 K2 后直接接入。

## E-025 profiling bootstrap 失败记录

初版 shape-A/current 哨兵虽成功，但在 torch 导入前没有固定 experimental backend；首个 disabled 随后暴露 default `patch_algorithm_selector` 与 PyTorch `best_config_future` 的接口漂移。失败发生在 lowering，检查确认没有 `output_code.py`。T-013 脚本修正为导入 torch 前固定 backend，正式结果全部使用修正后的四个目录；初始成功/失败目录均保留，不参与正式对比。

## 下一步

1. 登记 T-014：在审计目录实现 standalone different-K Triton 微原型，不注册到 Inductor、不修改功能源码。
2. 先只做 fp16 shape-A/unaligned 正确性和单 kernel profiler；任何 shape 超过 34/31 μs 时先调 tile，不能用端到端噪声掩盖。
3. 两个 shape 的单 task 和端到端 p50 都达到 10% 后，才扩展 dtype/layout/dynamic/backward 并讨论正式接入位置。
4. 若微原型无法达到预算，P-005 保持“不实现”，转向 pad family/P1。

## 证据

- 正式结果根目录：[`results/t013_mmplus_different_k_profile_20260821`](../results/t013_mmplus_different_k_profile_20260821/)
- T-012 baseline：[`t012_mmplus_different_k_baseline_20260821.md`](t012_mmplus_different_k_baseline_20260821.md)
- 变更与失败记录：[`change_control.md`](../change_control.md)
