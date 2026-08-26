# 成果、失败与中性尝试索引

> 更新到 2026-08-26。这里汇总已执行的动态工作，不代表 251 条 pass 全部完成；逐条状态
> 仍以 `report/pass_src_20260820/pass_evaluation_matrix.csv` 为准。
> P-013/default 工作已归档；用户随后恢复任务并将目标切换到
> `triton_experimental`，环境保持不变。迁移入口为
> `triton_experimental_migration_20260826.md`。

## 已形成正向成果

| 对象 | 结果 | 当前状态 | 入口 |
|---|---|---|---|
| addmm fusion | 8/8 代表配置 p50 均超过 10%，训练 blocker 已修 | `supported-beneficial` | `report/p0_sweep_performance_20260820.md` |
| different-K mm_plus_mm | 首批集成 p50 +15.29%/+18.04%，单 task | `conditional-supported-beneficial`，default-off | `report/t023_mmplus_different_k_integration_20260821.md` |
| fold_cat | nested cat 2→1，p50/p99 +10.14%/+10.32%，allocated peak -2,097,664 B | `supported-beneficial` | `report/t033_fold_cat_performance_20260824.md` |
| cat_to_view_pass | alias-safe，task 3→1，allocated peak -4,195,840 B | `supported-neutral-resource-beneficial` | `report/t029_t030_b2_alias_fix_performance_20260824.md` |
| fold_expand | 正例删除 identity expand，负例保留广播 expand | 功能通过，性能待测 | `report/t028_p1_b2_npu_compile_20260821.md` |
| repeat_to_expand_pass | broadcast-only repeat→expand，物理 copy 负例不改 | 功能通过，性能待测 | `report/t028_p1_b2_npu_compile_20260821.md` |
| fold_sink_view/fold_squeeze/fold_redundant_ops | 正例真实改图，负例保持，完整 alias/对象身份合同通过 | 功能通过，性能待测 | `report/t034_b2_view_copy_compile_20260824.md` |
| cat_slice_cat/pad_slice | 修复 alias/stride 后，p50 +24.00%/+31.35%，task 2→1/3→1，pad peak -10,485,248 B | 均为 `supported-beneficial` | `report/t036_b2_layout_alias_fix_20260825.md`、`report/t037_layout_pass_performance_20260825.md` |
| dtype_optimal_pass | 仅比较闭包安全降宽；p50/p99 +52.06%/+50.48%，显存不增 | `supported-beneficial`（development/audit-shim） | `report/t038_dtype_index_mask_semantic_fix_20260825.md`、`report/t039_dtype_index_mask_performance_20260825.md` |
| fold_iota_arithmetic_pass | 保留安全 iota 降宽并停用 Inf/overflow 不安全 cmp-sub；p50/p99 +55.78%/+54.51% | `supported-beneficial`（development/audit-shim） | `report/t038_dtype_index_mask_semantic_fix_20260825.md`、`report/t039_dtype_index_mask_performance_20260825.md` |
| bool_cast_mul_to_where_pass | view-chain p50/p99 +36.30%/+39.90%；direct 性能负域已由 T-042 guard 停用 | `supported-beneficial`（exact-zero 整数/布尔 + 非空单用户 view chain） | `report/t041_mask_hamming_performance_20260825.md`、`report/t042_bool_view_guard_integration_20260825.md` |
| batch_embedding_fusion_pass | 修复 step/dtype/alias；default/cat P50 +23.50%/+43.90%，tasks 9→3/13→3，但 peak/compile 增加 | `supported-neutral-resource-beneficial` | `report/t043_t046_b2_composite_passes_20260825.md` |
| dvm_graph_fusion | 15/15 NPU；default→DVM P50/P99 +24.20%/+39.93%，首次编译 20.32→2.81 s，allocated peak 相同 | `supported-beneficial` | `report/t047_t048_b3_dvm_mlir_20260826.md` |
| `_sfdp_pattern_1` | fp16 静态 inference 精确命中 vendor attention；P50/P99 +46.70%/+44.26%，task 4→1，allocated peak -87.31% | `supported-beneficial` | `report/t049_t050_b4_attention_first_20260826.md` |
| `_sfdp_pattern_13` | 三维 BMM inference 落到 vendor attention；P50 仅 +0.99%，但 task 3→1、首次编译 +91.37%、allocated peak -87.31% | `supported-neutral-resource-beneficial` | `report/t051_b4_attention_pattern13_performance_20260826.md` |
| P-013 pattern 5 NPU guard | 旧 rewrite P50 回退 103.23%、task 3→8；guard 后 P50 +50.28%、task 8→3、peak -1,054,720 B | guard `supported-beneficial`；rewrite 已停用 | `report/t053_b4_attention_pattern5_performance_20260826.md`、`report/t054_b4_attention_pattern5_guard_20260826.md` |
| P-014 backend decomposition 重入 | 三种 experimental 入口 12/12；erfc cleanup 后 source registrar 重入与 experimental→default→experimental 1/1 通过 | `source-verified-wheel-pending-shared-diff` | `report/t055_triton_experimental_enable_20260826.md` |
| P-016 int→float→int 默认安全门禁 | Float32 ON/OFF 单点证明边界数值差 1；三 dtype ON 均破坏输出 alias；source 默认已改为 opt-in | `source-verified-wheel-pending-shared-diff` | `report/t057_int_float_int_boundary_20260826.md` |

## 已否决或失败的尝试

| 对象/尝试 | 失败原因 | 最终处理 |
|---|---|---|
| pad mm/bmm/addmm gate bypass | 功能可运行，但 p50 回退 72.65%/65.31%/120.63%，task/显存增加 | 保持产品 gate，不写独立 padding Triton |
| fold_reduce 直返输入 | 数值为 0 误差，但 compiled 输出错误 alias 输入 | 判 correctness failure，禁止使用 |
| cat_to_view 直返输入 | 同上，破坏 eager cat 的新 storage 语义 | 改为 contiguous clone |
| fold_reduce alias-safe clone | 功能通过，但 p50/p99 回退 3.06%/6.72%，task/显存无收益 | 最终保留原 sum，禁用折叠 |
| large different-K 三 tile | paired p50 只有 6.64%–7.55%，未过 10% 门槛 | `supported-neutral-hold`，不扩大 gate |
| different-K workspace/grouped 搜索 | 降显存候选的 task duration 超门槛 | 不接入，保留 default-off 与 fallback |
| cat-slice-cat 原改写 | 两输出数值误差均为 0，但两个 eager 独立 cat storage/对象被合并 | 增加 cat1 完整用户集合 guard，风险图保持 |
| pad-slice 原改写 | 数值误差为 0，但结果 stride 改变且从新 storage 变为输入 view | 仅在所有 slice 用户确定物化新 storage 时折叠 |
| dtype int64 无条件降宽 | arange 直出改变可观察 dtype；float32→int64 可能截断大值 | 缩窄为静态安全值域且仅布尔比较闭包 |
| cmp(a-b,0)→cmp(a,b) | 普通数值样本相等，但 Inf/NaN 与定宽整数溢出存在反例 | 停用该子改写，保留安全 iota 优化 |
| broadcast mask 无 shape guard | 小 mask 经 where 广播后的大输出会被错误改回小 shape | 仅静态证明 mask/output shape 完全相等时压缩 |
| masked add 浮点无 guard | 普通相等误差为 0，但 `-0 + +0` 与直接选择的 signbit 不同 | 仅 exact-zero 整数/布尔 dtype 改写；浮点保持原图 |
| bool cast-mul 浮点无 guard | false 位置的 `0*Inf/NaN` 是 NaN，where false 分支却是常量零 | 仅 exact-zero 整数/布尔 dtype 改写；浮点保持原图 |
| bool cast-mul direct rewrite | p50/p99 回退 0.69%/19.02%，active10 device duration 62.56→73.86 μs | T-042 要求非空 view chain；direct 保持原图 |
| fusion_attention_v3_pass（910B2） | 与 legacy 同为单个 FlashAttentionScore task、显存相同，但 P50/P99 回退 4.85%/31.72% | T-046 在非 A5 停用升级并保持 legacy；不写重复 Triton attention |
| DVM sum pass-off 冒烟 | 人工跳过 fp32 pre/post-cast 后生成 bare fp16 DVM sum，首次 native 执行 segfault | 只作为“sum pass 对可用性必需”的证据；不可计算 paired speedup |
| expand_to_reshape aggregate | direct helper 已修复，但 capability 列表排除 expand/reshape/view，组合图都留在 custom op 外 | 当前 verdict `unsupported`；先评估 partition capability，不写 Triton |
| attention 初版只关闭 pattern 1 | 等价 `_sfdp_pattern_3_half_inference` 接管同一图，baseline 实际仍融合 | 该 smoke 性能数字作废；正式 baseline 同时关闭 1/3 |
| T-055 installed backend 隔离 | experimental 回切 default 时重复注册 `aten.erfc.default`，失败早于 lowering/codegen | P-014 已修 source；当前 installed wheel 仍待复验 |
| T-057 backend 状态隔离 | 回切 default 后 addmm check、五项 config/guard、matmul fold 与五个 decomposition 残留 experimental 状态 | `state-isolation-failed`；P-015 设计中，跨 backend 强制 fresh process |
| experimental int→float→int elision | Float32 pass ON 在 `±(2**24+1)` 差 1、OFF 精确；FP16/BF16 ON 也删图；三者 ON 均 alias 输入 | P-016 已 source 默认关闭；错误结果不测性能、不写 Triton |
| FP16/BF16 convert pass-OFF | FX/IR 保留 lowp cast，但 generated Triton 用 `tl.float32` compute type，未复现 eager lowp 舍入 | 独立 lowering/codegen 数值策略现象；不混入 P-016 最小修复 |

## 中性、未归因与环境类尝试

| 对象/尝试 | 结果 | 为什么不是成功/失败 |
|---|---|---|
| current different-K safe fallback | shape-A -0.30%、unaligned +2.53% | 功能基线正常，性能中性 |
| same-K mm_plus_mm transposed/dynamic | p50 +6.4%/+8.74% | 支持但未过 10% 性能门槛 |
| view_fold/fold_slice/fold_four 正例 | 目标节点在目标 pass 前已被消除 | reachability-neutral，收益不能归因给目标 pass |
| fold_cast/fold_clone/fold_detach 正例 | 目标节点在目标 pass 前已被规范化消除或整图旁路 | 完整语义通过，但收益不能归因给目标 pass |
| fold_to_copy 正/负例 | same-dtype copy 前序消除，dtype conversion 前序规范化为 prims cast | 完整语义通过，但 custom pass 当前可达性中性 |
| fold_where 三轮 paired | p50 +1.16%、p99 +3.12%，task 1→1、显存不变 | `supported-neutral`；kernel 更快但端到端未过 10% 门槛 |
| broadcast_const_mask_compress 三轮 paired | p50 +0.30%、p99 +1.01%，task 1→1、显存不变 | `supported-neutral`；baseline 已融合为单 kernel，不写 Triton 替身 |
| masked_add_compose_pass 三轮 paired | p50/p99 +3.71%/+9.84%，task 1→1、显存不变 | `supported-neutral`；未过 10% p50 gate，不写重复 Triton |
| sign_diff_hamming_fuse_pass 三轮 paired | p50/p99 +3.64%/+5.49%，task 1→1、显存不变 | `supported-neutral`；device kernel 更快但端到端固定开销主导 |
| fused_matmul_relu_pass（910B2） | `is_ascend950=false`，pass 不注册且 fused op 不解析 | `not-applicable`；A5 功能/性能仍待对应硬件验证 |
| batch embedding 完整性能 gate | 两个 safe cohort 的 P50/task 均改善，但 allocated peak 分别增加 1,831,936/13,628,416 B，首次编译约翻倍 | 保留 safe pass 并明确 trade-off，不升级为全面 `supported-beneficial` |
| T-039 初版 aggregate | 把一次 profiler device duration 略降误当 resource 改善 | 复核预登记后只接受 task/显存下降，最终 mask verdict 为 neutral；原始样本未重跑 |
| T-034 首轮 view 门禁 | 编译前 view 已规范化为 reshape，且首个 sink 正例是恒等 view | 审计假设错误；v2 修正后全过，不计产品失败 |
| 首次 grad-enabled B2 worker | inference-only POST driver 未调用目标 pass | 合法模式分流，不是产品失败 |
| 受限执行层 NPU case | `aclInit 507008`，驱动可见层同命令通过 | sandbox/设备可见性环境阻塞 |
| fresh Triton launcher 无 shim | PyTorch C++20、Triton C++17、torch_npu/CANN headers 不匹配 | 环境合同未闭环；审计 shim 不能冒充产品环境成功 |
| T-036 首个修复后 cat alias worker | `CPATH` 误指 editable 源码，缺少 `ATen/ATen.h` | 环境命令错误；改用 wheel headers 后全过，不计产品失败 |
| T-040 首个 NPU worker | 同样误用了 editable PyTorch include view，缺少 `ATen/ATen.h` | 失败原样保留；改用 site-packages wheel headers 后 9/9 通过 |
| T-040 installed 测试的 autoload-off 启动 | test stub 后再次加载 native torch_npu，`_npu_dtype_cast` schema 重复注册 | 启动方式错误；正常 autoload 下 installed 76/76 |
| DVM K=1 子 pass isolate | 进入 torch_npu helper 前已是 `cast→mul→cast`，跳过 helper 后图不变 | direct FX 正确，但当前 compile 增量为零，记 `not-applicable/upstream-predecomposed` |
| DVM backend 首次 32-case 启动 | autoload=0 且测试未显式 import torch_npu，全部在设备注册前失败 | 中性启动错误；autoload retry 32/32 |
| 完整 MLIR backend | `_load_mlir_backend()` 显式要求 `torch_mlir`，当前环境未安装 | environment-blocked；DVM 自有 codegen 不受阻且 32/32 |
| attention 5/18/21/28/29 无 shim 初跑 | 辅助 Triton fresh launcher 在 editable PyTorch include view 缺 `ATen/ATen.h` | 环境失败保留；audit-only shim 重跑只用于归因 |
| attention pattern 5/21/29 | exact matcher 与数值通过，但 dispatcher 最终重新展开为 BMM + Triton | 功能通过、性能未测；不能把 matcher counter 冒充 vendor fusion |
| `npu_fa` scale 怀疑 | positional/keyword 动态结果完全相同；wrapper 与 vendor inverse-scale 误差为 0 | 确认 legacy divisor 合同，不作产品修复；backward/dynamic 仍待测 |
| pattern 5/21/29 re-expansion | replacement 保留 additive float mask；vendor SDPA 只接受 bool/None，因而走 math | 安全 capability fallback，不误报 lowering 缺失；一般 float bias 不转 bool |
| pattern 30 前两次对照 | 未设置 fresh cache或只打开 debug，generated code可见但 counter 未重新执行 | 方法中性记录；独立 cache 第三次 exact counter=1 并落到 vendor attention |

## 当前安装态

- 最终 torch_npu wheel SHA256：
  `3909fd649d777b8dfd393342da0ff2b88c5cce2ef219f0d103d063af4c2d4989`。
- 当前 wheel 已归档为 `artifacts/torch_npu_t054_p013_verified.whl`；P-013 三文件
  snapshot 与文件级哈希位于 `artifacts/p013_source_snapshot/`。
- P-013 guard 单测 2/2、installed paired 和邻近 3 family 通过；P-012 旧 wheel 保留为
  `artifacts/torch_npu_t053_before_p013.whl`。
- 性能失败的 T-029 clone candidate wheel 单独保留在
  `artifacts/torch_npu_t029_alias_safe_clone_candidate.whl`，不是当前安装态。

## 下一步

B2 27 条、B3 8 条、B4 八个代表 family 功能以及 pattern 1/13/5 性能和 P-013 均作为
default-backend 历史闭环保存。T-055/T-056 已完成；T-057 已关闭 backend 状态、
int-float-int 和 GELU approximate 三项。P-017 证明 installed P-013 的 none 合同失败，并在
current source 通过 FP32/FP16/BF16 六组 NPU source-overlay、非法参数和生成代码验证恢复合同。
P-014/P-016/P-017/P-018 均完成 source 验证但 wheel 因共享未安装 diff 暂缓。T-058 addmm 首个
shape-A cohort p50/p99 中位数改善 22.17%/14.05%，后续 11/11 capability 与 unaligned 第二性能
cohort 通过；P-018 current source 已默认启用 fusion 并提供可恢复 opt-out，状态为
`source-verified-wheel-pending-host-tail-monitor`。下一步继续显式隔离 installed P-013 的
`elide_int_float_int`，进入 permute-gather、outer rsplit。旧 default 结论只作为迁移优先级，不直接计入
experimental 成功率。
