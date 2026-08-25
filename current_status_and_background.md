# Inductor Pass NPU 调研：现状、环境选择与背景知识

## 1. 当前结论

截至 2026-08-25，本任务已完成基于 `Pass/src` 的 251 条概念级静态清单、逐项评估矩阵、P0 gate/功能/性能闭环，以及 P1 B2 前 24 个 custom pass 的结构与真实 NPU 审计。T-011 修复 `strict_sum`；T-012 至 T-024 完成 different-K mm_plus_mm 的 default-off 条件性接入；T-025/T-026 证明 pad family 功能可承接但性能否决；T-027 至 T-031 发现并关闭 `fold_reduce`、`cat_to_view_pass` 的 storage alias 问题；T-032/T-033 证明 `fold_cat` 有益；T-034/T-035 确认 `fold_where` 中性；T-036/T-037 修复 layout alias/stride 并确认两个 pass 有益；T-038/T-039 修复 dtype/index/mask 边界并证明 safe dtype/iota 的 p50 改善 52.06%/55.78%。T-040 修复 mask arithmetic 的 signed-zero 与 `0*Inf/NaN` 语义；T-041 进一步把 masked-add/sign-Hamming 判为 neutral，发现 bool-cast view-chain p50/p99 改善 36.30%/39.90% 而 direct 路径回退；T-042 通过 view-chain guard 停止 direct 改写，源码态/安装态 76/76、NPU 3/3 通过。当前 Benchmark 环境安装的是 T-042 源码构建 wheel（`--no-deps`）。

**用户已要求从暂停检查点继续。** 历史恢复基线为
[PAUSED_CHECKPOINT_20260821.md](PAUSED_CHECKPOINT_20260821.md)，恢复后的 E-030 至
E-135 已完成。T-014 至 T-041 的正式性能结论均来自运行前后空闲设备、fresh-process paired 采样；隔离失效、审计门禁错误或环境失败的原始结果保留但不进入 verdict。different-K 当前是 `integration-supported-beneficial-opt-in-memory-tradeoff-environment-pending`：shape-A/unaligned p50 改善 `15.29%/18.04%`，但 candidate peak allocated 比 baseline 均多 `270,336 B`。B2 前 24 条已形成各自功能/性能或 reachability 结论；T-041/T-042 新关闭 bool-cast view-chain beneficial 与 masked-add/Hamming neutral。当前 editable PyTorch / Triton launcher / torch_npu-CANN header 组合仍不能无垫片 fresh compile host launcher，所以带 Triton fresh launcher 的结论继续明确标注 development/audit-shim。

当前 251 条矩阵记录分为：可直接构造用例 164 条、阶段 observer 41 条、registry container 25 条、人工审查 21 条。verdict 为 234 条 `not-run`、3 条 `unsupported`、7 条 `supported-beneficial`、1 条 `conditional-supported-beneficial`、4 条 `supported-neutral`、1 条 `supported-neutral-resource-beneficial`、1 条 `supported-pass-disabled-performance-rejected`。七个直接 beneficial 项是 addmm、fold_cat、cat-slice-cat、pad-slice、dtype optimal、safe iota 与 bool-cast view-chain；`fold_where`、mask compression、masked-add 和 sign-Hamming 为端到端性能中性，另有 alias-safe `cat_to_view_pass` 的资源收益和最终保留原 sum 的 `fold_reduce`。旧 `/Dynamo` 194 条清单只保留作历史对照。

此前因其他进程修改共享环境而冻结动态测试；用户确认环境稳定后已恢复。当前动态测试按用户指定固定从 `/home/z50063656/Benchmark/env.sh` 启动 Conda `benchmark-py311`，测试进程仍从 `/home/z50063656/tmp` 发起。T-022/T-023 区分了“runtime 与已缓存 kernel 可用”和“fresh host launcher 编译合同完整”两个层次：新 launcher仍需匹配 PyTorch C++20、Triton Ascend、torch_npu 与 CANN headers，不能用审计垫片冒充正式环境已修复。

关于“使用 PyTorch wheel 还是源码编译”，建议如下：

| 阶段 | 推荐形态 | 是否需要源码编译 |
|---|---|---|
| 静态枚举和源码机制分析 | 直接读取 PyTorch/torch_npu 源码树 | 不需要 |
| 第一轮可用性与性能基线 | 独立、冻结、版本严格匹配的 PyTorch + torch_npu + Triton Ascend wheel 环境 | 通常不需要，优先使用 wheel |
| 修改纯 Python pass、pattern、lowering | 对应 commit 的源码树，加受控源码导入或构建出的 wheel | 通常不需要重编 C++，但必须保证运行代码确实来自该源码 |
| 修改 PyTorch/torch_npu C++、注册、ABI 或编译扩展 | 对应 commit 的源码树 | 需要构建，建议产出 wheel 后安装到任务专属环境 |
| 只开发独立手写 Triton kernel | 匹配的 Triton Ascend + PyTorch/torch_npu wheel | 通常不需要重编 PyTorch |
| 将 Triton kernel 正式接入内置 pass | 取决于接入点；Python 注册可不重编，C++/打包变更需要构建 | 按修改范围决定 |

因此，本任务不应一开始就全量源码编译。先用冻结 wheel 环境建立可复现基线；确认某个 pass 需要修改后，再使用完全匹配的源码做开发。最终验证时至少要有一套干净 wheel 安装结果，避免 `PYTHONPATH`、源码树和 site-packages 混用。

## 2. 已完成工作

### 2.1 静态 pass 清单

已对以下源码范围做 AST/文本级静态扫描：

- `/home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes`
- `/home/z50063656/Pass/src/torch_npu/torch_npu/_inductor`
- NPU Triton experimental 控制点
- DVM/MLIR 后端图变换和子 pass

当前清单共 251 条概念级记录。它们是 pipeline、pattern、扩展函数和 NPU 自定义 pass 的“审计记录”，不等于 251 个可以逐一独立运行的测试用例。旧 `/Dynamo` 快照只有 194 条，漏掉的主要是函数式/生成式 pattern、8 个 DVM/MLIR 变换和一个 pad-mm 控制 gate。

按阶段统计：

| 阶段 | 数量 |
|---|---:|
| inductor-extension | 91 |
| joint_graph | 48 |
| npu-custom | 3 |
| npu-dvm | 1 |
| npu-dvm-subpass | 5 |
| npu-mlir | 1 |
| npu-mlir-subpass | 1 |
| npu-pattern | 1 |
| npu-triton-experimental | 4 |
| post_grad | 70 |
| pre_grad | 26 |

按机制统计：

| 机制 | 数量 |
|---|---:|
| backend-graph-pass | 2 |
| backend-subpass | 6 |
| extension-function | 21 |
| generated-pattern-family | 33 |
| npu-custom-pass | 27 |
| pattern-entry | 86 |
| pattern-registry | 25 |
| pipeline | 41 |
| replacement-entry | 10 |

静态路由标签统计：

| 标签 | 数量 | 含义 |
|---|---:|---|
| backend-sensitive | 2 | 明确依赖后端行为，需要优先验证 |
| generic-needs-validation | 147 | 通用逻辑，但必须在 NPU 图和生成代码中验证 |
| generic-pipeline | 41 | pipeline/observer 类入口 |
| needs-npu-validation | 17 | 已发现 NPU 相关风险点 |
| npu-specific | 44 | NPU 专用逻辑 |

产物：

- `report/pass_src_20260820/pass_inventory.json`：机器可读清单。
- `report/pass_src_20260820/pass_inventory.md`：人工审阅索引。
- `report/pass_src_20260820/pass_evaluation_matrix.csv`：251 条逐项验收合同；当前 234 条 `not-run`，另有 17 条已形成 verdict；bool-cast view-chain、masked-add、sign-Hamming、dtype/iota/mask compression 以及此前 addmm、fold_cat、layout、pad family 等均已登记最终、条件性或中性结论。
- `audit_passes.py`：不导入 torch 的静态清单生成器。

当前源码基线为 PyTorch `release/2.14@8e86e0a`、torch_npu `master@83cc452`、Triton Ascend `release/3.2.2@8bd9f38`。torch_npu 的 torchair/inductor-npu-ext 子模块未初始化，因此旧清单中的 5 条 npu-ext 记录没有冒充为当前源码可用项。

### 2.2 已识别的首批候选

1. `mm_plus_mm`

   上游存在 `mm + mm -> tuned_mm_plus_mm` pattern。experimental same-K 代表网格 8/8 功能正确、6 个配置 p50 收益超过 10%；different-K 原 lowering会安全退回两个 mm 加 add。T-012 至 T-022 已关闭 fallback 分解、standalone K1/K2 kernel、三 dtype、真实转置、dynamic/backward、扩展性能、正式设计与 large hold。T-023 现已在 torch_npu default backend 接入 NPU-only、default-off、ATen extern fallback-first template；row/column 正例、AOTAutograd 与五类 negative 通过。shape-A/unaligned fresh-process integrated p50 改善 `15.29%/18.04%`，每 step 是唯一融合 task；candidate additional allocated peak比 baseline均多 `270,336 B`，根因为 `65,536 B × 6` 的 Triton Ascend workspace。T-024 的大 tile触碰 L0C 上限，group2 虽把 peak 降到 320,512 B，但 task p50 10.42/10.54 μs 超过预登记门槛，未接入产品。因此该 pass 为 default-off 条件性有益，large/dynamic/empty/arbitrary-stride/same-K仍 fallback；下一步只需在 headers 匹配环境做无 shim fresh compile 复验，不全局解除 size guard。

2. `pad_mm`

   根因已确认：上游 `pad_mm.py:check_device()` 只接受 CUDA/XPU，NPU 即使在 pass 执行期间设置 `force_shape_pad=True` 也会在 `can_pad()` 直接返回 False；experimental 还额外关闭 `shape_padding`。T-025 仅在测试进程内绕过 device gate 后，mm/bmm/addmm positive 均真实生成 pad→GEMM→slice且正确，aligned negative保持不触发，证明下游功能可承接。T-026 三轮 fresh-process paired p50却分别回退 `72.65%/65.31%/120.63%`，每步task由1增至`5/5/7`，allocated peak多`278,528/294,400/279,552 B`。因此当前产品 verdict仍为`unsupported`，replacement为`rejected-performance-regression`：保持gate，不复制CUDA策略，也不手写Triton padding。

3. `addmm` fusion

   default 的 tensor-bias 正例已触发 `aten.addmm`，experimental 正例保持 `mm + add`，两个 scalar 负例均未误触发。default 的 8 个 dtype/shape/layout/dynamic 代表配置均功能正确且 p50 收益超过 10%，没有 p50 回退。`(M,N)`、`(1,N)` 和 `(N,)` bias 正确融合，mixed dtype 正确不融合；`strict_sum` 兼容修复后，full-bias 与 vector-bias backward 的输出和 3 个输入梯度均正确。该 pass 的最终 verdict 已关闭为 `supported-beneficial`。

4. P1 B2 custom pass 前四批

   T-027 至 T-031 已把结构测试扩为 33/33，并完成首批 7 个 pass 的 fresh NPU 正负例。`fold_expand` 与 `repeat_to_expand_pass` 功能通过、性能待测；`view_fold_pass`、`fold_slice`、`fold_four_op_pass` 的正例在目标 pass 前已被前序 pipeline 消除，记 reachability-neutral。`fold_reduce` 和 `cat_to_view_pass` 初始数值虽为 0 误差，却错误复用了输入 storage；T-029 clone 修复关闭正确性 blocker。T-030 三轮 paired 显示 fold_reduce clone p50/p99 回退 3.06%/6.72%，最终 T-031 保留原 sum并禁用该折叠；cat clone p50 +2.29%、p99 -0.78%，task 3→1、allocated peak 减少 4,195,840 B，保留为 latency-neutral/resource-beneficial。

   T-032 把结构测试扩为 41/41，并完成第二批 4 个 pass 的 8 个 NPU 正负例。`fold_cast`、`fold_clone`、`fold_detach` 的目标正例在目标 pass 前已被规范化删除或旁路，记 partial reachability；真实 dtype conversion、输出 clone 以及 detach 的 alias/对象身份/`requires_grad` 语义均保持。`fold_cat` 正例真实执行 cat 2→1，multi-user 负例保持 2→2；T-033 三轮 paired p50/p99 改善 10.14%/10.32%，task 2→1、allocated peak 减少 2,097,664 B，关闭为 `supported-beneficial`。

   T-034 把结构测试扩为 51/51，并完成第三批 5 个 pass 的正负例。`fold_sink_view` 真实交换 reshape/relu 拓扑，`fold_squeeze` 与 `fold_redundant_ops` 删除可证明冗余的 view 组合，`fold_where` 以 clone 保持 eager 的新 storage，四条 negative 均未误改且完整语义通过；`fold_to_copy` 在目标 pass 前被消除或规范化，记 reachability-neutral。T-035 对 `fold_where` 做三轮 paired：p50/p99 改善 1.16%/3.12%，task 1→1、additional allocated peak 不变，关闭为 `supported-neutral`，不需要手写 Triton。

   T-036 把结构测试扩为 60/60，并覆盖剩余 layout pass `cat_slice_cat_fold_pass` 与
   `pad_slice_fold`。修复前两条 safe positive 都能执行，但额外可观察 cat 输出会从两个独立
   storage 退化为同一对象，pad slice 直出会从新 storage 变成输入 view，二者数值误差仍为
   0。产品源码已增加 cat1 用户集合与 pad slice 消费者物化证明；新源码 wheel 安装后 6/6
   NPU 正/负/alias worker 通过，安全 positive 仍触发，风险图保持。T-037 随后完成各六个
   paired worker：cat-slice-cat p50/p99 改善 24.00%/22.87%、task 2→1、allocated peak
   不增；pad-slice p50/p99 改善 31.35%/30.34%、task 3→1、allocated peak 减少
   10,485,248 B。两条均关闭为 `supported-beneficial`，无需 Triton 替身。

   T-038 把结构测试扩为 67/67，并修复第五批三条 dtype/index/mask pass：
   `dtype_optimal_pass` 只在静态安全值域且所有直接用户为布尔比较时降宽；
   `fold_iota_arithmetic_pass` 保留安全 iota downcast、停用对 Inf/NaN 与整数溢出不成立的
   cmp-sub；`broadcast_const_mask_compress` 只在 mask 与 where 输出 shape 静态完全相等时
   压缩。源码/安装态 67/67 与 9/9 NPU 功能 worker 通过。T-039 的 18/18 paired worker
   显示 dtype/iota p50 分别改善 52.06%/55.78%，均 beneficial；mask 只有 0.30%，task/显存
   不变，记 neutral。三条都无需手写 Triton。

   T-040 把结构测试扩为 76/76，并关闭第六批三条 mask/hamming pass 的功能层。
   `masked_add_compose_pass` 和 `bool_cast_mul_to_where_pass` 分别发现 signed-zero 与
   `0*Inf/NaN` 反例，现只在 bool/整数 exact-zero dtype 改写；`sign_diff_hamming_fuse_pass`
   的特殊浮点、整数 keepdim 与 multi-user 边界不需新增 guard。新 source wheel 安装后
   9/9 NPU worker 的图门禁、NaN/signbit 与完整语义通过。T-041 的 24/24 paired worker
   将 masked-add/sign-Hamming 关闭为 neutral；bool direct p99 回退 19.02%，view-chain
   p50/p99 改善 36.30%/39.90%。T-042 因此只保留非空 view-chain 改写，新 wheel 的
   76/76 与 direct/view/float 3/3 NPU 复验通过，bool pass 在该 capability 内 beneficial。

所有产品修改都已先在 `change_control.md` 登记。addmm、different-K、pad family 与 B2 前六批均已有成功、失败或中性证据；当前转入 B2 最后 3 个复合 custom pass，而 T-023 只保留正式无 shim 环境复验。

### 2.3 探针框架

`run_npu_probe.py` 已形成草案，设计目标包括：

- 每个 backend 使用独立进程，避免全局 monkeypatch 和 cache 污染。
- 比较 eager 与 compiled 正确性。
- 记录首次编译/执行延迟、稳态 mean/stdev/p50/p99、峰值 NPU 内存。
- 通过 `GraphTransformObserver` 和 `PatternMatcherPass` 观察实际运行的 pass。
- 缺少 NPU 时记录 `skip`，不能记为成功。

该脚本尚未在稳定的任务环境中验证，不应直接作为正式 benchmark。

另已新增 `run_p0_gate_probe.py`，专门覆盖 `mm_plus_mm`、`pad_mm`、`pad_bmm`、`pad_addmm`、`addmm fusion` 五个 P0 family，共 10 个正/负用例。它的主进程不导入 torch，每个 case/backend 使用独立 worker，并区分“编译与正确性通过”和“目标 pass 已确认触发”。首轮 `default,triton_experimental` 共 20 个 NPU 组合已全部 `compile-correct`；详细触发结论见 `report/p0_gate_first_run_20260820.md`。探针还提供测试侧 `--disable-target-pass`，用于 fresh-worker 单 pass A/B；首个 shape 的三轮结果见 `report/p0_ab_first_shape_20260820.md`。

2026-08-20 又完成 P0 覆盖扩展哨兵：旧 fp16/shape-A 行为 4/4 回归通过；default/addmm 在 bf16、small、真实转置 stride 和第二组动态 shape 上仍生成符号化 `aten.addmm`；experimental/mm_plus_mm 在 fp32、非对齐、真实转置 stride和第二组动态 shape 上仍生成符号化 `_mm_plus_mm`。本轮没有采性能，详细证据见 `report/p0_sweep_smoke_20260820.md`。

随后完成 P0 非笛卡尔代表功能矩阵：addmm/default 与 mm_plus_mm/experimental 各 8 个配置，共 16/16 正确且 16/16 目标图确认，覆盖三 dtype、四个代表 shape、真实转置输入和动态 replay。addmm 的 bf16 最大绝对差 0.5，但逐元素 bf16 容差通过，已保留为后续精度边界。详细证据见 `report/p0_sweep_function_matrix_20260820.md`。

相同 cohort 的 current/disabled paired benchmark 主矩阵 96/96 正确，另有 mm_plus_mm dynamic runs 300 复核 6/6 正确。addmm 8/8 配置的 p50 收益超过 10%；mm_plus_mm 6/8 超过 10%，transposed 6.4% 和 dynamic 8.74% 记为 `supported-neutral`，没有配置出现 p50 回退。全部采样使用开始和结束时均无进程的物理 NPU 2，未混用其他卡上的外部任务。详细证据见 `report/p0_sweep_performance_20260820.md`。结合后续语义和 T-011 回归，addmm 已关闭 final verdict；mm_plus_mm 仍受 default gate 限制。

P0 语义层先完成 6 个 inference case 和 3 个 forward/backward case。inference 6/6 正确并符合触发/不触发预期；mm_plus_mm same-K 与 addmm full-bias backward 的输出和所有输入梯度均正确。最初 addmm vector-bias backward 在 bias 梯度的 `aten.sum.dim_IntList` lowering 被 torch_npu `make_reduction()` 缺少 `strict_sum` 参数阻断；T-011 补齐接口后，最小 sum、原 vector-bias backward、full-bias backward、addmm inference 和 mm_plus_mm backward 全部通过。不同 K 的 mm_plus_mm 仍按源码设计安全 unfuse。详细证据见 `report/p0_semantic_matrix_20260821.md` 和 `change_control.md:E-023`。

## 3. 已知运行环境事实

当前实际动态环境由 `/home/z50063656/Benchmark/env.sh` 启动，Conda 环境为 `/home/z50063656/envs/benchmark-py311`：Python 3.11.15；PyTorch `2.14.0a0+git8e86e0a` 是指向 `/home/z50063656/Benchmark/pytorch-upstream` 的 editable 安装；torch_npu 使用 `master@83cc452480c3546fd5cccf853bfe3a360ce9dbfc` 源码构建的 wheel `torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，以 `--no-deps` 安装到 site-packages；runtime `triton.__version__` 为 3.2.0，对应任务记录的 Triton Ascend source commit 为 `8bd9f380d2786002b84b5248f00838c26f900515`；CANN 为 9.0.1，设备为 8 张 Ascend910B2。静态清单仍读取 `/home/z50063656/Pass/src`，PyTorch 与 torch_npu commit 和运行时记录对齐。

2026-08-21 T-023 结束复核时，PyTorch 产品源码未改；Triton Ascend 产品源码未改。torch_npu commit 未变，tracked diff 包含 T-011 `lowering.py` 和 T-023 的 `__init__.py`、`config.py`、`fx_passes/post_grad.py`、`kernel/__init__.py`，另有新 kernel 与目标 UT；既有未跟踪代码生成文件不纳入功能 diff。构建过程的临时 torchgen link/workspace均已清理，ACL 子模块恢复；详细记录见 `change_control.md:E-057` 至 `E-071`。

该组合已从 `/home/z50063656/tmp` 完成 CPU/NPU eager 与 Inductor smoke、P0、T-011 至 T-042 审计。当前安装的是 T-042 源码 wheel，SHA256 为 `ea801e791373b0bd3adf9d4bfb6253ace75afa800c71b0451c9b206e4664fe5a`；它包含此前 guard 与 bool view-chain 性能保护。T-040 wheel 已保存在 `artifacts/torch_npu_t040_before_t042_bool_view_perf_guard.whl`，SHA256 为 `b273aeedcb9d1367de65328bea78a448ff1eb81fa4a85dca3f910e556c7b2460`；更早的 T-038、T-036、T-031 与 T-029 审计 wheel 也仍保留。

T-022 进一步确认“已安装 runtime 可运行”不等于“fresh Triton host launcher 可编译”：editable PyTorch 指向的源码树当前缺少 `torch/include`；wheel headers 已要求 C++20，而 Triton Ascend 3.2.0 launcher 固定 `-std=c++17`；installed torch_npu headers 还引用 CANN 9.0.1 没有声明的两个 conditional graph 类型。审计专用 compiler shim 只用于隔离 profiler/benchmark，不能进入产品或最终环境结论。正式 integration 最终需要一套 headers/编译标准匹配的 wheel 或完整源码构建环境。

性能采样期间每个配置都重新选择当时进程表为空的卡；出现外部进程或环境失败的原始目录均保留但排除。T-023 两组正式 fresh-process paired benchmark在物理 NPU 1 完成，批次前、中、后进程表均为空；T-024 workspace筛选也在运行前空闲的物理 NPU 1 完成。没有终止或清理任何外部进程。

## 4. 为什么优先使用 wheel 基线

wheel 基线的主要价值不是“安装方便”，而是可复现：

- PyTorch、torch_npu、Triton Ascend 和 CANN 的版本边界明确。
- 不会因为当前目录、`PYTHONPATH` 或源码树中未提交修改而改变导入代码。
- 每次 fresh process 使用相同二进制和 Python 文件，性能比较才有意义。
- 可将 baseline wheel 和 candidate wheel 安装到两个独立环境，做成真正的 paired benchmark。

建议的第一套正式环境是“发布/基线环境”：

1. 创建任务专属环境，不复用正在变化的 `benchmark-py311`。
2. 选择一组官方支持矩阵内的 PyTorch、torch_npu、Triton Ascend、CANN 版本。
3. 清除源码树注入的 `PYTHONPATH`，确认 `torch.__file__` 和 `torch_npu.__file__` 都指向该环境的 site-packages。
4. 固化 wheel 文件名、SHA256、Python 版本、CANN 路径、driver/npu-smi 版本和 SoC。
5. 运行 eager、Dynamo import、最小 Inductor compile，再开始 pass 测试。

## 5. 什么时候必须源码编译

以下情况需要 PyTorch 或 torch_npu 源码构建：

- 修改了 C++ dispatcher、设备注册、codegen extension 或 ABI。
- 修改的 commit 没有对应 wheel，且不能用纯 Python 覆盖验证。
- 需要验证 PyTorch 与 torch_npu 的跨仓库接口变更。
- 最终交付要求证明修改能从干净源码构建并打包。

推荐构建策略不是在工作环境里原地编译，而是：

1. 锁定 PyTorch commit。
2. 以该 commit 构建 PyTorch wheel。
3. 用这个 PyTorch wheel 对齐构建 torch_npu wheel。
4. 将两个 wheel 安装到新的 candidate 环境。
5. baseline 与 candidate 分环境测试，避免同一 site-packages 被覆盖。

纯 Python pass/pattern/lowering 的早期开发通常可以不重编 C++，但必须记录导入路径，并证明运行时加载了 candidate 文件。最终仍建议产出 wheel 做交付验证。

## 6. Inductor pass 背景

简化后的 `torch.compile(backend="inductor")` 链路如下：

```text
Python model
  -> TorchDynamo 捕获 FX graph
  -> AOTAutograd 拆分 forward/backward
  -> pre_grad passes
  -> joint_graph passes
  -> post_grad passes
  -> lowering 到 Inductor IR
  -> scheduler / fusion / codegen
  -> NPU backend kernel
  -> runtime execution
```

pass 的作用不只是一种“图融合”。常见类型包括：

- pattern rewrite：把一组算子替换成更适合后端的等价图。
- decomposition：把高级算子拆成后端已经支持的基础算子。
- canonicalization：统一 shape、layout、dtype 或表达式形式，帮助后续匹配。
- fusion：减少 kernel 数、访存和 launch 开销。
- lowering selection：为同一语义选择 vendor op、template、Triton 或 fallback。
- scheduler/codegen hook：改变分组、分块、并行和最终内核生成。

一个 pass 在 NPU 上“可用”，至少要同时满足：

1. 能触发：输入 FX graph 符合 pattern，设备 gate 没有把 NPU 排除。
2. 语义正确：dtype、shape、stride、动态 shape、forward/backward 和边界值正确。
3. 能 lowering：替换后的算子有 NPU lowering 或合法 fallback。
4. 能 codegen：backend 依赖、编译器和运行时能够生成、加载内核。
5. 没有隐性 graph break 或 CPU fallback。
6. 性能合理：稳定态和端到端收益抵消编译、额外内存与同步开销。

因此，“源码里注册了 pass”不等于“在 NPU 上可用”；“输出正确”也不等于“pass 实际触发”。必须同时检查 pass observer、FX 图、generated code、kernel 数和 profiler。

## 7. NPU backend 与替代实现

当前 torch_npu 代码中可见多种 Inductor NPU backend loader：`default`、`ascendc`、`mlir`、`dvm`、`triton_experimental`。典型选择方式是：

```python
torch.compile(
    fn,
    backend="inductor",
    options={"npu_backend": "ascendc"},
)
```

不同 backend 不是简单的性能开关。它们可能注册不同 lowering、scheduler、codegen 和 pass，且会修改 Inductor 全局状态。因此正式测试必须每个 backend 使用 fresh process。

当某个 pass 不可用或性能差时，替代方案的优先级应按问题类型选择：

| 问题 | 优先方案 |
|---|---|
| pattern 没触发 | 修 pattern、decomposition 或 device capability gate |
| 替换图正确但没有 NPU lowering | 增加 NPU lowering 或选择已有 vendor op |
| GEMM/attention/collective 性能差 | 先比较 vendor/CATLASS/AscendC，不默认手写 Triton |
| pointwise/reduction 多 kernel、访存主导 | 适合评估手写 Triton fusion |
| 动态 shape/layout 不支持 | 限定 capability gate，保留原图 fallback |
| Triton 编译失败 | 先修工具链/版本匹配，不能归因到 pass |

手写 Triton 是实现手段，不是目标。它最适合表达规则、可融合、访存主导的 pointwise/reduction；对于矩阵乘、attention 等算子，成熟 vendor kernel 往往有更完整的 tiling、流水和硬件特性支持。只有在正确性和 paired benchmark 都证明收益时，才能把 Triton 替代接入 pass。

## 8. 正式性能方法

每个 pass/case/backend 至少需要一组同机、同环境、fresh process 的 baseline/candidate 数据：

- CANN、driver、SoC、PyTorch、torch_npu、Triton 版本。
- 输入 dtype、shape、stride、dynamic/static、forward/backward。
- warmup 次数和 runs 次数；建议起点为 warmup 10、runs 100。
- 首次编译/首轮延迟。
- 稳态 mean、stdev、p50、p99。
- 峰值 NPU 内存、kernel 数、graph break/fallback。
- 每次计时前后正确同步 NPU。
- 记录设备占用，避免和其他进程共享同一张卡做性能结论。

验收建议分两层：

- 可用度替身：原 pass 在 NPU 不支持时，candidate 能正确执行、无非预期 CPU fallback，并有明确 capability gate 和回退路径。
- 性能提升：candidate 相对相同输入的 baseline 在稳态 p50 有明确收益，同时 p99、编译时间和峰值内存没有不可接受回退。

## 9. 后续执行顺序

1. 环境、导入路径、源码 commit、CPU/NPU eager 与 Inductor smoke 已完成并冻结为当前 baseline。
2. P0 gate、generated code、backend 隔离、代表功能矩阵和 paired performance 网格已完成。
3. torch_npu reduction `strict_sum` 接口兼容、wheel 构建/安装与 addmm vector/row-bias backward blocker 已完成；addmm verdict 已关闭为 `supported-beneficial`。
4. mm_plus_mm different-K 的当前安全 fallback paired baseline 已完成，shape-A/unaligned 的 p50 分别为 -0.30%/+2.53%，结论为 neutral。
5. different-K fallback profile、standalone 两 shape 正确性、单 task profiler 和三轮 paired benchmark 已完成；fp16/contiguous/static p50 改善 15.60%/17.12%，但 additional peak 多约 1.38 MiB。
6. bf16/fp32、真实 transposed/non-contiguous、dynamic replay、backward、正式接入设计，以及 large profiler/tile/memory 分解均已完成；large 最终为 supported-neutral-hold，中小 static cohort 保留 beneficial gate。
7. different-K default-off template、源码 wheel、首批功能/性能/memory和 workspace 替代搜索已完成；状态为 `conditional-supported-beneficial`。不再重复 T-014 至 T-024，环境支线只需在匹配 headers 的独立环境做无 shim fresh compile smoke。
8. pad family 与 P1 B2 前 24 条已完成结构/NPU/性能分流，其中 bool-cast view-chain 新关闭为 beneficial，masked-add 与 sign-Hamming 新关闭为 neutral。当前覆盖 B2 最后 3 个复合 pass，之后进入 B3 DVM/MLIR 与 B4 attention。已闭环 case 不重跑，除非环境或源码基线改变。

## 10. 当前文件导航

- `PAUSED_CHECKPOINT_20260821.md`：本次暂停的环境、工作树、设备状态和精确恢复点；恢复工作时首先阅读。
- `README.md`：入口与运行约束。
- `outcome_index.md`：已执行工作的成功、失败、中性、未归因和环境类索引。
- `change_control.md`：环境、提案和源码修改冻结记录。
- `replacement_plan.md`：NPU 替代实现总体策略。
- `audit_passes.py`：静态清单生成器。
- `run_npu_probe.py`：动态探针草案。
- `run_p0_gate_probe.py`：P0 gate 当前行为的进程隔离探针（首轮 20 个 NPU 组合已执行）。
- `report/p0_gate_first_run_20260820.md`：P0 首轮可用性、目标触发和 backend 分化报告。
- `report/p0_ab_first_shape_20260820.md`：P0 两个已触发 pass 的首形状三轮 paired performance 报告。
- `report/p0_sweep_smoke_20260820.md`：P0 dtype/layout/dynamic 扩展的旧行为回归与实机哨兵。
- `report/p0_sweep_function_matrix_20260820.md`：P0 两个候选的 16 个代表配置功能与目标图矩阵。
- `report/p0_sweep_performance_20260820.md`：P0 两个候选的 96-worker 主性能矩阵与 mm_plus_mm dynamic 高样本复核。
- `report/p0_semantic_matrix_20260821.md`：P0 bias/shape guard、different-K fallback、forward/backward、`strict_sum` 根因与 T-011 修复闭环报告。
- `report/t012_mmplus_different_k_baseline_20260821.md`：different-K current/disabled 两 shape、三轮 paired baseline 与后续 profile 闸门。
- `report/t013_mmplus_different_k_profile_20260821.md`：different-K fallback 的 NPU kernel 组成、duration、步内 gap 与微原型性能预算。
- `report/t014_t016_mmplus_different_k_candidate_20260821.md`：standalone candidate 的正确性演进、两 shape profiler、paired performance 和内存 trade-off。
- `report/t017_t019_mmplus_different_k_coverage_20260821.md`：三 dtype、真实转置、dynamic replay 和 backward 语义。
- `report/t020_mmplus_different_k_extended_benchmark_20260821.md`：扩展性能矩阵和 large 初次 hold 证据。
- `report/t021_mmplus_different_k_integration_design_20260821.md`：正式 template/pattern/fallback 数据流与拟修文件。
- `report/t022_mmplus_different_k_large_profile_20260821.md`：large 三 tile device profiler、paired 稳态、内存分解和环境限制。
- `report/t023_mmplus_different_k_integration_20260821.md`：default-off wheel 接入、功能、集成 paired 性能、workspace 根因与条件性结论。
- `report/t024_mmplus_different_k_workspace_20260821.md`：七种 workspace/tile/grouped-program 配置和停止结论。
- `report/t025_t026_pad_family_20260821.md`：pad 三 family 的 capability 可用、性能失败与停止结论。
- `report/t028_p1_b2_npu_compile_20260821.md`：B2 首批 7 pass 的 NPU 正负例、alias blocker 和最终闭环。
- `report/t029_t030_b2_alias_fix_performance_20260824.md`：alias 修复、三轮性能、失败/中性尝试与 T-031 最终 wheel。
- `report/t032_b2_redundancy_compile_20260824.md`：fold_cast/fold_cat/fold_clone/fold_detach 的结构、NPU 可达性和完整语义合同。
- `report/t033_fold_cat_performance_20260824.md`：fold_cat 的三轮 paired 性能、task 与显存收益。
- `report/t034_b2_view_copy_compile_20260824.md`：第三批五条 pass 的结构、NPU 拓扑、完整语义和中性门禁修正。
- `report/t035_fold_where_performance_20260824.md`：fold_where 的三轮 paired 中性性能、task duration 与显存结论。
- `report/t036_b2_layout_alias_fix_20260825.md`：cat-slice-cat/pad-slice 的零数值误差 alias 缺陷、源码 guard、wheel 与 NPU 闭环。
- `report/t037_layout_pass_performance_20260825.md`：两条 layout pass 的三轮 paired 延迟、task、显存与 beneficial 结论。
- `report/t038_dtype_index_mask_semantic_fix_20260825.md`：dtype/index/mask 的边界反例、保守 guard、67/67 测试、wheel 与 9/9 NPU 功能闭环。
- `report/t039_dtype_index_mask_performance_20260825.md`：safe dtype/iota 的有益性能与 mask compression 的中性性能结论。
- `report/t040_mask_hamming_semantic_fix_20260825.md`：mask arithmetic IEEE 反例、exact-zero dtype guard、76/76 测试、wheel 与 9/9 NPU 功能闭环。
- `report/t041_mask_hamming_performance_20260825.md`：四个 safe-positive case 的 24/24 paired 性能与 direct/view 分流。
- `report/t042_bool_view_guard_integration_20260825.md`：bool direct 性能 guard、新 wheel、76/76 与 3/3 NPU 安装态闭环。
- `t014_mmplus_different_k_triton.py`：不同 K standalone Triton 微原型。
- `t015_mmplus_different_k_candidate_profile.py`：candidate-only NPU profiler。
- `t016_mmplus_different_k_candidate_benchmark.py`：fallback/candidate 三轮 paired benchmark。
- `t022_mmplus_different_k_large_profile.py`：large tile 单任务 profiler。
- `t022_mmplus_different_k_large_benchmark.py`：large paired benchmark 与 pure output/baseline/candidate 内存分解。
- `t023_mmplus_different_k_integration.py`：installed-package pattern/template 功能探针。
- `t023_mmplus_different_k_integration_performance.py`：fresh-process profiler、paired worker 与 allocator trace。
- `t023_mmplus_different_k_integration_performance_aggregate.py`：六 worker 纯读取聚合器。
- `t024_mmplus_different_k_workspace_screen.py`：workspace/tile/grouped-program audit-only 筛选器。
- `t028_b2_custom_pass_compile.py`：B2 正负例、图计数与 dtype/stride/storage alias 门禁。
- `t030_b2_alias_performance.py`、`t030_b2_alias_performance_aggregate.py`：fold_reduce/cat 的 fresh-process 单 pass paired 性能与聚合。
- `t032_b2_redundancy_compile.py`：B2 第二批四条 pass 的多输入/tuple/完整语义 NPU worker。
- `t033_fold_cat_performance.py`、`t033_fold_cat_performance_aggregate.py`：fold_cat 的 fresh-process 单 pass paired 性能与聚合。
- `t034_b2_view_copy_compile.py`：B2 第三批五条 pass 的拓扑/计数与完整语义 NPU worker。
- `t035_fold_where_performance.py`、`t035_fold_where_performance_aggregate.py`：fold_where 的 fresh-process 单 pass paired 性能与聚合。
- `t036_b2_layout_alias_compile.py`：两条 layout pass 的正/负/alias 图计数、输入与跨输出完整语义 worker。
- `t037_layout_pass_performance.py`、`t037_layout_pass_performance_aggregate.py`：两条 layout pass 的 fresh-process 单 pass paired 性能与聚合。
- `t038_dtype_index_mask_compile.py`：第五批三条 pass 的 dtype/overflow/broadcast 图门禁与完整 NPU 语义 worker。
- `t039_dtype_index_mask_performance.py`、`t039_dtype_index_mask_performance_aggregate.py`：三条 safe-positive 的 fresh-process paired 性能与严格预登记聚合。
- `t022_launcher_cc_wrapper.sh`、`t022_cann_header_compat.h`：仅限审计的 fresh host launcher 编译垫片，不能进入产品。
- `p0_case_design.md`：五个 P0 family 的触发条件、正负用例和验收边界。
- `p1_batch_design.md`：66 条 P1 的分组、已有测试证据与动态验收顺序。
- `report/pass_src_20260820/pass_inventory.md`：当前静态 pass 索引。
- `report/pass_src_20260820/pass_inventory.json`：当前静态 pass 机器可读数据。
- `report/pass_src_20260820/pass_evaluation_matrix.csv`：逐 pass 评估矩阵。
- `report/npu_probe.json`：旧环境历史诊断，不能作为 baseline。
