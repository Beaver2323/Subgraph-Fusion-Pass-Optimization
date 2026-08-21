# mm_plus_mm different-K standalone Triton candidate 报告

## 1. 结论

T-014 至 T-016 已证明一个限定能力范围的 standalone candidate 可行：在 CANN 9.0.1、Ascend910B2、fp16、row-major contiguous、static forward 下，`bm128_bn128_bk128` Triton kernel 对 shape-A 与 unaligned 都与 eager 完全对齐，并把当前 Inductor different-K fallback 的端到端 p50 分别改善 15.60% 和 17.12%。

这不是正式 pass 接入结论。candidate 尚未覆盖 bf16/fp32、非连续 layout、dynamic 和 backward；额外峰值 allocated memory 比 baseline 多约 1.38 MiB。`mm_plus_mm` 的矩阵最终 verdict 因 default backend gate 和能力覆盖未闭环，仍保持 `not-run`。

## 2. 环境与方法

| 项目 | 值 |
|---|---|
| Python | 3.11.15 |
| PyTorch | `2.14.0a0+git8e86e0a`，Python source 为 `Pass/src/pytorch` |
| torch_npu | `2.14.0a0+git83cc452`，源码 wheel SHA256 `263ffec...2704`，`--no-deps` 安装 |
| Triton | runtime `3.2.0`，Triton Ascend source `release/3.2.2@8bd9f38` |
| CANN / NPU | CANN 9.0.1 / Ascend910B2 |
| 设备 | 物理 NPU 1；每个动态阶段开始和结束时进程表为空 |
| dtype/layout/mode | fp16 / row-major contiguous / static forward |
| candidate tile | `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128` |

所有测试从 `/home/z50063656/tmp` 启动。baseline 是 experimental backend 下当前不同 K 安全 fallback，即两个矩阵乘和一个 add；candidate 是直接 standalone Triton 调用，不向 Inductor registry 注册。

性能合同为每个 shape 独立 fresh process，baseline/candidate 各 warmup 10、runs 100、3 轮，轮次顺序交替。主判定使用三轮 p50 中位数；同时记录每轮 mean±stdev、p99、首次编译和峰值 allocated memory。

## 3. 正确性演进

| 阶段 | 结果 | 解释 |
|---|---|---|
| 初版 split-store | 128³ 有 20,475/61,440 个失败；64³ 非有限 | split-store 后半块没有可靠写回 |
| 完整 store + 单 accumulator | 只剩 4 个失败，max abs `0.02978515625` | 与 fallback 的两次 fp16 cast + fp16 add 舍入顺序不同 |
| 双 fp32 accumulator，各自 cast fp16 后 add | shape-A、unaligned 最大/平均绝对误差均为 0 | 达到 `rtol=atol=0.01`，尾块 mask 也通过 |

正式正确性结果：

| shape | `(M,K1,N,K2)` | fallback | candidate |
|---|---|---|---|
| shape-A | `(192,256,320,128)` | max/mean abs `0/0` | max/mean abs `0/0` |
| unaligned | `(191,255,319,127)` | max/mean abs `0/0` | max/mean abs `0/0` |

成功的 Inductor baseline 按图模式分流不展开 `output_code.py`；当前 fallback 结构已经由 T-012/T-013 的源码、图与 profiler 证据确认。

## 4. Candidate NPU profiler

合同：普通 warmup 10、profile warmup 1、active 10，Level0/AiCoreNone，每 step 同步。

| shape | task 数 | kernel | mean±stdev (μs) | p50 (μs) | p99 (μs) | 预算 (μs) |
|---|---:|---|---:|---:|---:|---:|
| shape-A | 10/10 | `different_k_mm_plus_mm_kernel` | 8.714±0.337 | 8.76 | 9.30 | 33.93 |
| unaligned | 10/10 | `different_k_mm_plus_mm_kernel` | 13.824±0.541 | 13.81 | 14.68 | 31.13 |

两个 shape 都是每次调用一个 NPU task，p50 均低于 T-013 根据端到端 10% 门槛反推的预算。profiler 解析日志出现 ACL→NPU flow 关联告警，但 `kernel_details.csv` 完整导出 10 条 duration；本报告不使用包含显式同步的 step 间 gap 推导 launch 开销。

## 5. 端到端 paired benchmark

### 5.1 三轮 mean±stdev

单位均为 ms。

| shape/mode | round 1 | round 2 | round 3 |
|---|---:|---:|---:|
| shape-A baseline | 0.263275±0.006765 | 0.270264±0.016217 | 0.271469±0.006962 |
| shape-A candidate | 0.226943±0.006384 | 0.225072±0.004835 | 0.225999±0.004832 |
| unaligned baseline | 0.274050±0.019315 | 0.279959±0.007528 | 0.283600±0.008389 |
| unaligned candidate | 0.231612±0.009603 | 0.231958±0.004581 | 0.233039±0.008070 |

### 5.2 三轮中位数汇总

| shape | baseline p50 | candidate p50 | p50 改善 | baseline p99 | candidate p99 | p99 改善 |
|---|---:|---:|---:|---:|---:|---:|
| shape-A | 0.265935 | 0.224440 | 15.60% | 0.286370 | 0.243030 | 15.13% |
| unaligned | 0.278290 | 0.230640 | 17.12% | 0.305010 | 0.257210 | 15.67% |

两 shape 的 p50 都超过预设 10% 门槛，p99 方向一致，没有观察到尾延迟回退。benchmark 前后 baseline/candidate 的最大/平均绝对误差仍全部为 0。

## 6. 编译与内存 trade-off

| shape | baseline 首次编译+运行 | candidate 首次编译+运行 | baseline additional peak | candidate additional peak |
|---|---:|---:|---:|---:|
| shape-A | 20295.573 ms | 2036.690 ms | 246,784 B | 1,696,768 B |
| unaligned | 20047.084 ms | 2007.131 ms | 244,736 B | 1,695,744 B |

首次编译值受 Inductor/Triton 不同编译路径及缓存影响，只是诊断数据，不能作为加速结论。candidate additional peak 在两个 shape 都比 baseline 多约 1.38 MiB；绝对值较小，但约为 baseline 的 6.9 倍，正式接入前需确认来源和规模扩展行为。

## 7. 判定与下一步

当前可写为：`standalone-fp16-contiguous-static-beneficial`。它证明“不同 K 单 task 融合”值得继续开发，但不证明通用 pass 已可用。

下一阶段按以下顺序扩展：

1. bf16/fp32 正确性，并让 kernel 输出 dtype/舍入语义与 fallback 对齐；
2. 真实 transposed/non-contiguous stride；
3. dynamic shape 与第二组 replay，确认不会错误 specialize；
4. forward/backward 语义与梯度；
5. 复核 additional peak memory 来源与更大 shape 的增长规律；
6. 全部通过后另立正式 Inductor template/lowering、capability gate 和 fallback 提案。

任一覆盖面失败时都应保留原图 fallback，不直接解除上游 size guard。

## 8. 原始证据

- T-014 correctness：`results/t014_mmplus_different_k_triton_20260821/shape_a_two_acc_retry2/`、`unaligned_128_resumed1/`
- T-015 profiler：`results/t015_mmplus_different_k_candidate_profile_20260821/shape_a_resumed1/`、`unaligned_resumed1/`
- T-016 benchmark：`results/t016_mmplus_different_k_candidate_benchmark_20260821/shape_a/`、`unaligned/`
- 当前 fallback 基线：[T-012 报告](t012_mmplus_different_k_baseline_20260821.md)
- fallback kernel profile：[T-013 报告](t013_mmplus_different_k_profile_20260821.md)
