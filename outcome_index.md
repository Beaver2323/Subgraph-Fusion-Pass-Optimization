# 成果、失败与中性尝试索引

> 更新到 2026-08-24。这里汇总已执行的动态工作，不代表 251 条 pass 全部完成；逐条状态
> 仍以 `report/pass_src_20260820/pass_evaluation_matrix.csv` 为准。

## 已形成正向成果

| 对象 | 结果 | 当前状态 | 入口 |
|---|---|---|---|
| addmm fusion | 8/8 代表配置 p50 均超过 10%，训练 blocker 已修 | `supported-beneficial` | `report/p0_sweep_performance_20260820.md` |
| different-K mm_plus_mm | 首批集成 p50 +15.29%/+18.04%，单 task | `conditional-supported-beneficial`，default-off | `report/t023_mmplus_different_k_integration_20260821.md` |
| fold_cat | nested cat 2→1，p50/p99 +10.14%/+10.32%，allocated peak -2,097,664 B | `supported-beneficial` | `report/t033_fold_cat_performance_20260824.md` |
| cat_to_view_pass | alias-safe，task 3→1，allocated peak -4,195,840 B | `supported-neutral-resource-beneficial` | `report/t029_t030_b2_alias_fix_performance_20260824.md` |
| fold_expand | 正例删除 identity expand，负例保留广播 expand | 功能通过，性能待测 | `report/t028_p1_b2_npu_compile_20260821.md` |
| repeat_to_expand_pass | broadcast-only repeat→expand，物理 copy 负例不改 | 功能通过，性能待测 | `report/t028_p1_b2_npu_compile_20260821.md` |

## 已否决或失败的尝试

| 对象/尝试 | 失败原因 | 最终处理 |
|---|---|---|
| pad mm/bmm/addmm gate bypass | 功能可运行，但 p50 回退 72.65%/65.31%/120.63%，task/显存增加 | 保持产品 gate，不写独立 padding Triton |
| fold_reduce 直返输入 | 数值为 0 误差，但 compiled 输出错误 alias 输入 | 判 correctness failure，禁止使用 |
| cat_to_view 直返输入 | 同上，破坏 eager cat 的新 storage 语义 | 改为 contiguous clone |
| fold_reduce alias-safe clone | 功能通过，但 p50/p99 回退 3.06%/6.72%，task/显存无收益 | 最终保留原 sum，禁用折叠 |
| large different-K 三 tile | paired p50 只有 6.64%–7.55%，未过 10% 门槛 | `supported-neutral-hold`，不扩大 gate |
| different-K workspace/grouped 搜索 | 降显存候选的 task duration 超门槛 | 不接入，保留 default-off 与 fallback |

## 中性、未归因与环境类尝试

| 对象/尝试 | 结果 | 为什么不是成功/失败 |
|---|---|---|
| current different-K safe fallback | shape-A -0.30%、unaligned +2.53% | 功能基线正常，性能中性 |
| same-K mm_plus_mm transposed/dynamic | p50 +6.4%/+8.74% | 支持但未过 10% 性能门槛 |
| view_fold/fold_slice/fold_four 正例 | 目标节点在目标 pass 前已被消除 | reachability-neutral，收益不能归因给目标 pass |
| fold_cast/fold_clone/fold_detach 正例 | 目标节点在目标 pass 前已被规范化消除或整图旁路 | 完整语义通过，但收益不能归因给目标 pass |
| 首次 grad-enabled B2 worker | inference-only POST driver 未调用目标 pass | 合法模式分流，不是产品失败 |
| 受限执行层 NPU case | `aclInit 507008`，驱动可见层同命令通过 | sandbox/设备可见性环境阻塞 |
| fresh Triton launcher 无 shim | PyTorch C++20、Triton C++17、torch_npu/CANN headers 不匹配 | 环境合同未闭环；审计 shim 不能冒充产品环境成功 |

## 当前安装态

- 最终 torch_npu wheel SHA256：
  `29c3c105453a36d8f2eb648eeb0a2d35cfd0cb871c34697c6aaf17fb1a96a6f5`。
- 安装态旧 FX UT：33/33；当前测试源码扩展为 41/41。T-032 的 8 个 NPU case 与
  T-033 的 6 个性能 worker 全部通过图/完整语义门禁。
- 性能失败的 T-029 clone candidate wheel 单独保留在
  `artifacts/torch_npu_t029_alias_safe_clone_candidate.whl`，不是当前安装态。

## 下一步

为 B2 其余 16 个 custom pass 建立最小结构正负例，优先覆盖 alias、dtype、dynamic shape
和默认开关；结构层通过后再逐项进入 fresh NPU compile。随后执行 B3 DVM/MLIR 与 B4
attention。T-023 的无 shim 环境复验作为独立环境支线，不阻塞主线。
