# mm_plus_mm different-K 扩展性能与内存报告

## 1. 结论

T-020 补齐了 bf16、fp32、真实转置 stride 和更大 shape 的 paired benchmark。在 CANN 9.0.1、Ascend910B2、warmup 10、runs 100、3 轮交错采样下，前三个扩展配置的 candidate p50 相对当前 Inductor different-K safe fallback 改善 11.20%–15.25%，且 p99 同向改善。

large 的隔离有效重测按预设主判据也得到 11.58% p50 改善，但第 3 轮 candidate p50 轻微回退，且高方差、长尾和 additional peak memory 放大明显。因此 large 当前是 `p50-pass-but-unstable-hold`，不进入自动启用范围。

后续 T-022 已完成更稳定的 large 三 tile profile/paired/内存分解：device task 正确且稳定，但三组 paired p50 改善只有 6.64%–7.55%，large 最终收敛为 `supported-neutral-hold`；删除首次长生命周期输出后的 candidate/baseline additional allocated peak 差值为一个 `655,360 B` logical output。阅读本报告的 large 初始结论时，应同时阅读 [T-022 后续报告](t022_mmplus_different_k_large_profile_20260821.md)，后者优先用于当前 capability verdict。

T-020 仍是 standalone candidate 审计，没有向 Inductor registry/lowering 注册，也没有修改 PyTorch、torch_npu 或 Triton Ascend 功能源码。

## 2. 环境与方法

| 项目 | 值 |
|---|---|
| Python | 3.11.15 |
| PyTorch | `2.14.0a0+git8e86e0a`，commit `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| torch_npu | `2.14.0a0+git83cc452`，commit `83cc452480c3546fd5cccf853bfe3a360ce9dbfc` |
| torch_npu 安装 | `dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，SHA256 `263ffec23ae37c651f3d57199c0cfa8b14f398ea603f8e1434ea14b237792704`，`--no-deps` |
| Triton | runtime 3.2.0，Triton Ascend source commit `8bd9f380d2786002b84b5248f00838c26f900515` |
| CANN / NPU | CANN 9.0.1 / Ascend910B2 |
| backend | `triton_experimental`，`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| candidate | standalone different-K 单 task kernel，`BLOCK_M/N/K=128/128/128` |

所有进程都从 `/home/z50063656/tmp` 启动，在 `TORCH_COMPILE_DEBUG=1` 和独立 debug 目录下运行。baseline 均编译成功，因此按图模式分流规则没有读取成功用例的 `output_code.py`。

每个配置在一个 fresh process 内产生 eager reference、当前 compiled fallback 和 standalone candidate；baseline/candidate 各 warmup 10、runs 100、3 轮，每轮交替顺序。主判据是三轮 p50 的中位数，同时保留 mean±stdev、p99、首次编译+执行和 allocated peak。

由于外部任务频繁变动，bf16/fp32/transposed/large 分别在采样时空闲的物理 NPU 7/6/3/7 执行。每个配置的改善率是同卡、同进程的成对对比；不用跨卡绝对时延比较 dtype/layout。

## 3. 正确性

| 配置 | `(M,K1,N,K2)` | baseline max/mean abs | candidate max/mean abs | 预设容差 |
|---|---|---:|---:|---:|
| bf16/shape-A/contiguous | `(192,256,320,128)` | `0/0` | `0/0` | `3e-2` |
| fp32/shape-A/contiguous | `(192,256,320,128)` | `0/0` | `2.8610e-5/1.4306e-6` | `1e-4` |
| fp16/unaligned/transposed | `(191,255,319,127)` | `0/0` | `0/0` | `1e-2` |
| fp16/large/contiguous | `(512,768,640,384)` | `0/0` | `0.0625/8.7023e-7` | `1e-2` |

表中数据在 benchmark 前后一致，输出 dtype 与 reference 相同且都为有限值。large 的最大绝对误差较大，但 `torch.testing.assert_close` 在预先登记的 fp16 `rtol=atol=0.01` 下通过；正式接入前应增加数值分布与更大 K 的压测，不应只看 mean error。

转置配置的 4 个输入均为 `is_contiguous=false`，stride 分别为 `(1,191)`、`(1,255)`、`(1,191)`、`(1,127)`。

## 4. 稳态性能

### 4.1 三轮中位数汇总

单位为 ms。

| 配置 | baseline p50 | candidate p50 | p50 改善 | baseline p99 | candidate p99 | p99 改善 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| bf16/shape-A/contiguous | 0.271645 | 0.241225 | 11.20% | 0.299060 | 0.265340 | 11.28% | beneficial |
| fp32/shape-A/contiguous | 0.283520 | 0.240285 | 15.25% | 0.309150 | 0.260550 | 15.72% | beneficial，保留 baseline 单轮长尾 |
| fp16/unaligned/transposed | 0.289620 | 0.252825 | 12.70% | 0.347840 | 0.277940 | 20.10% | beneficial，当前 baseline autotune 限制 |
| fp16/large/contiguous | 0.344085 | 0.304245 | 11.58% | 1.290280 | 0.771170 | 40.23% | p50 pass，unstable hold |

### 4.2 每轮 mean±stdev

| 配置/模式 | round 1 | round 2 | round 3 |
|---|---:|---:|---:|
| bf16 baseline | 0.272762±0.008741 | 0.273804±0.009134 | 0.274082±0.008977 |
| bf16 candidate | 0.244170±0.009801 | 0.243306±0.005949 | 0.245677±0.006257 |
| fp32 baseline | 0.267286±0.009249 | 0.285468±0.007711 | 0.341249±0.368972 |
| fp32 candidate | 0.242225±0.005805 | 0.240690±0.004513 | 0.252955±0.004992 |
| transposed baseline | 0.290048±0.014185 | 0.282209±0.011751 | 0.309490±0.042942 |
| transposed candidate | 0.253883±0.007105 | 0.250492±0.007208 | 0.259522±0.009272 |
| large baseline | 0.450021±0.734061 | 0.360428±0.049186 | 0.445529±0.217806 |
| large candidate | 0.301185±0.063499 | 0.389957±0.659924 | 0.541238±0.630181 |

large 的 3 轮 p50 成对值为 `0.327660/0.278710`、`0.344085/0.304245`、`0.367740/0.370085`。第 3 轮 candidate 比 baseline 慢约 0.64%，且 candidate 第 3 轮 p99 为 3.611300 ms。即使三轮中位数通过 10% 门槛，当前也不具备足够稳定性来支持 large 默认启用。

transposed 和隔离无效的 large 首轮编译时均观察到 mspti device-time 不可用、baseline 回退 event-based autotune 的告警。因此收益只代表当前 Inductor NPU 实际路径，不等价于与理论最优 vendor GEMM 比较。

## 5. 编译与内存 trade-off

| 配置 | baseline 首次编译+执行 | candidate 首次编译+执行 | baseline additional peak | candidate additional peak | candidate 多用 |
|---|---:|---:|---:|---:|---:|
| bf16 | 25,862.263 | 1,858.602 | 246,784 B | 1,696,768 B | 1,449,984 B |
| fp32 | 19,195.779 | 1,890.044 | 492,544 B | 3,392,512 B | 2,899,968 B |
| transposed fp16 | 21,479.457 | 2,046.156 | 244,736 B | 1,695,744 B | 1,451,008 B |
| large fp16 | 31,639.715 | 3,387.853 | 1,311,744 B | 5,899,264 B | 4,587,520 B |

首次编译数据只是诊断值：baseline 包含整个 Inductor 图编译和 autotune，candidate 是直接 Triton specialization，两者不能用于声称 compile-time 改善。

additional peak 不是固定 workspace：fp32 在相同 shape 上约为 bf16 的 2 倍，large fp16 由约 1.70 MB 增长到 5.90 MB。绝对增量仍不大，但正式 capability gate 不能忽略 dtype 和输出规模。

## 6. Capability gate

当前证据支持以下分层：

| 范围 | 功能 | 性能 | 处置 |
|---|---|---|---|
| fp16/contiguous/static，shape-A 与 unaligned | 通过 | p50 +15.60%/+17.12% | 候选 beneficial 区间 |
| bf16/fp32/contiguous/static，shape-A | 通过 | p50 +11.20%/+15.25% | 候选 beneficial，带 dtype 内存 gate |
| fp16/unaligned/transposed/static | 通过 | p50 +12.70% | 候选 beneficial，需保留 fallback/autotune |
| fp16/large/contiguous/static | 通过 | p50 中位数 +11.58%，方差/长尾不稳定 | hold，不自动启用 |
| dynamic replay | 语义通过 | 新 shape/divisibility 可有一次 specialization 编译，未做稳态矩阵 | 保留 fallback，不全局开启 |
| backward | output 和 4 输入梯度语义通过 | 未测候选 backward 性能 | 由 AOTAutograd 独立 backward graph 承接 |

上表不是任意 shape 范围的泛化证明。正式设计不应简单解除上游 different-K size guard 或对所有 NPU 图强制使用固定 128³ tile；应将 candidate 作为可选实现，并保留当前两个 mm + add fallback。large 需先通过 profiler/tile 稳定性调优，才能扩展 gate。

## 7. 无效采样与边界

large 首轮在 NPU 1 采样前进程表为空，但结束后出现外部 PID 1594529（约 38 GiB）。它的原始目录完整保留，但不进入上述 capability 结论。报告使用的 large 数据是 NPU 7 采样前后均空闲的 `rerun1`。

本轮仍未覆盖空维度、混合 dtype、批量 matmul、更广的 M/N/K 分布、数值压力输入、与 Inductor 正式 autotune/cache 的集成，以及真实模型中的 end-to-end 收益。

## 8. 原始证据

- bf16：`results/t020_mmplus_different_k_extended_benchmark_20260821/bf16_shape_a_contiguous/result.json`
- fp32：`results/t020_mmplus_different_k_extended_benchmark_20260821/fp32_shape_a_contiguous/result.json`
- transposed：`results/t020_mmplus_different_k_extended_benchmark_20260821/fp16_unaligned_transposed/result.json`
- large 有效重测：`results/t020_mmplus_different_k_extended_benchmark_20260821/fp16_large_contiguous_rerun1/result.json`
- large 隔离无效首轮：`results/t020_mmplus_different_k_extended_benchmark_20260821/fp16_large_contiguous/result.json`
- [T-014–T-016 candidate 正确性/profile/基础性能](t014_t016_mmplus_different_k_candidate_20260821.md)
- [T-017–T-019 dtype/layout/dynamic/backward 覆盖](t017_t019_mmplus_different_k_coverage_20260821.md)
