# 实验报告与数据索引

> 索引更新时间：2026-09-15 11:56 CST（UTC+08:00）
> 原则：报告保存当时环境和结论，不因主线变化回写历史；当前任务状态以
> `../docs/CURRENT_STATUS.md` 为准。

## 当前主线证据

| 文件 | 作用 | 当前边界 |
| --- | --- | --- |
| [Pattern 22部署与边界回归](../issues/REF-sfdp-pattern-22-native/部署与边界回归报告.md) / [功能性能讲解](../results/current/T-106/pattern-22_讲解.md) / [修复验证](../issues/REF-sfdp-pattern-22-native/修复验证报告.md) | 条件授权、前后代码及调用栈、实际FX/IR/codegen、原件与六臂样本 | 已部署Pass并验证，PERF_REGRESSED；T-106为3/4，21号归因阻塞仍保留 |
| [Pattern 2注册候选验证](../issues/REF-sfdp-pattern-2-native/注册候选验证报告.md) | 原失败调用栈、NPU训练分解、具体代码和完整原方法复验 | 10次Tensor比较、4次精确改写通过；未部署、未计性能 |
| [剩余任务与专项入口](remaining_work_20260915.md) | 17/15去重源码证明、16/29针对性GPU补测、22设备边界与T-098全量实测状态 | 71个ID/69独立合同，48闭环、21剩余；不把候选和排队任务算完成 |
| [当前不依赖GPU的执行进度](unblocked_work_20260914.md) | T-091/T-100及attention验收、T-098精度模式诊断，附学习入口 | T-091 MIXED、T-100 IMPROVED、T-104 pattern13 REGRESSED；其余仍按真实阶段记录 |
| [T-104 pattern13结果与讲解](../results/current/T-104/pattern-13_讲解.md) | pattern意图、社区/派生测例合同、真实GPU/NPU代码、六臂性能 | 功能通过；微图回退约4%～8%，不调整默认配置 |
| [T-105 pattern18讲解](../results/current/T-105/pattern-18_讲解.md) / [T-106 pattern23讲解](../results/current/T-106/pattern-23_讲解.md) | 布尔mask/零加性mask、多输出K/V、原例与派生性能域、六臂真实代码与样本 | 两项PERF_IMPROVED，仅限各自微图；不是整个attention批次或模型收益 |
| [T-106 pattern22数值失败分析](../issues/REF-sfdp-pattern-22-native/数值失败分析.md) | 原始失败栈、实际输出代码和逐kernel定位 | 历史失败：首处分歧在行最大值kernel；后续修复见本表部署报告 |
| [T-106 pattern22修复候选验证](../issues/REF-sfdp-pattern-22-native/修复候选验证报告.md) | 修复前后代码、原例及三个邻接、源码指纹与边界 | 历史隔离候选21次Tensor/9次改写通过；后续安装态验证单独留证，不覆盖候选记录 |
| [T-105 pattern19讲解](../results/current/T-105/pattern-19_讲解.md) | FP32社区/派生输入、数学SDPA实际代码、六臂样本 | PERF_REGRESSED / PARTIAL_ALIGNED；half混合mask编译缺口仍保留 |
| [T-104 pattern15性能OFF失败](../issues/REF-sfdp-pattern-15-native/性能OFF数值失败.md) | 入口标量适配之后的新NaN、实际调用栈及代码线索 | 未计时，待逐kernel因果验证；不直接归为22号同根因 |
| [历史日志与T-078补证复核](history_logs_and_t078_extension_review_20260914.md) | 新历史日志24/24重解析；FP16 value=1正确单例与NPU修复原件对照 | 日志缺口已清零；严格历史再认证仍pending，不能混为全量完成 |
| [Attention性能阶段复核](attention_performance_contract_review_20260914.md) | inference的dropout置零/去重风险、极低非零dropout设计的社区来源、训练前向测量边界 | 13/18/19/23/24有逐例实测，其余仍需各自目标/数值门禁，不整体宣称收益 |
| [pattern 13真实GPU/NPU代码对照](../issues/REF-sfdp-pattern-13-native/GPU与NPU代码对照.md) / [14代码断言适配](../issues/REF-sfdp-pattern-14-native/代码断言适配分析.md) / [18容器观察器](../issues/REF-sfdp-pattern-18-native/数值观察器修正说明.md) | Flash/FA/math展开区别、误判边界、原社区tuple数值比较及最小适配 | 逐例实证，不把测试适配修正当产品修复或性能收益 |
| [最新 GPU 精确归因](gpu_observer_review_20260914.md) / [16/17/29映射解释](gpu_target_mapping_16_17_29_20260914.md) | 7批28原例、114个具名观察、1370份正文；逐编号判定与源码讲解 | 25个归因确认，16/17/29命中别的编号；T-091后续验收见上一行 |
| [current_acceptance_unit_matrix.md](current_acceptance_unit_matrix.md) / [CSV](current_acceptance_unit_matrix.csv) | 当前逐 acceptance-unit 状态入口 | 71个ID/69独立单元，47个reference冻结、47个NPU/comparison、47项性能处置；不等于严格历史再认证通过 |
| [T-106 pattern24讲解](../results/current/T-106/pattern-24_讲解.md) | reshape+BMM链、混合dtype mask、SDPA数学展开、六臂真实代码 | PARTIAL_ALIGNED / PERF_REGRESSED，保留代码断言适配与产品路径差异 |
| [非补证阻塞工作完成报告](t087_t096_unblocked_completion_20260911.md) | 七个单元闭环、逐pattern讲解；T-087/T-096安装态修复与原失败/候选分列 | 3改善、1中性、1混合、1免测、1回退；E8M0默认开启是正确性修复，额外NPU边界部分对齐 |
| [T-087安装态修复验证](../issues/REF-respecialize-current-device-native/修复验证报告.md) / [T-096安装态修复验证](../issues/REF-e8m0-log2-pattern-native/修复验证报告.md) | 原例、必要调用栈、源码patch、备份、近邻和前后FX/IR/output_code | 原例1/1+近邻4/4；原例3/3+功能对照7/7+域检查+六臂性能，均部署Pass未社区合入 |
| [本轮 GPU 包复核](gpu_incoming_review_20260911.md) | 13批40 cases、T-102完整收件、目标归因/数值边界和T-112去重 | 1226份正文可恢复；28个合同目标归因仍待补 |
| [T-087～T-090 NPU推进与适配讲解](t087_t090_npu_progress_20260911.md) | 原生阻断、最小适配、真实代码框/调用链、负例判据修正、FX/IR/output_code入口 | 保留12:37阶段记录；后续正式完成与产品问题见上行报告 |
| [T-084～T-086 NPU功能/性能/修复闭环](t084_t086_npu_function_performance_and_fix_20260910.md) | 五个pattern的源码意图、GPU/NPU对照、调用栈、最小适配、FX/IR/codegen及六臂性能 | 1/3/1单元闭环；1改善、2回退、2混合；pointless-cumsum已产品gate |
| [T-085 cumsum产品门禁](../issues/REF-pointless-cumsum-native/性能回退与产品门禁报告.md) / [overlap适配](../issues/REF-overlap-device-put-sync-native/NPU最小适配报告.md) / [partitioned-scatter适配](../issues/REF-partitioned-scatter-positive-native/NPU最小适配与性能报告.md) | 对应issue内的真实问题、调用链、最小改动及修复前后FX/IR/output_code入口 | cumsum保存前后原件；两个能力适配链接canonical证据，避免重复大文件 |
| [t076_t077_history_reaudit_20260907.md](t076_t077_history_reaudit_20260907.md) | 当前逐项核验、源码 oracle 边界、历史性能复算与轻量日志补证命令 | 24 GPU 关键正文/19 NPU 原件/4 性能汇总已核验；严格总门禁仍 pending，原始 verdict 不改写 |
| [t076_t077_history_reaudit_20260906.md](t076_t077_history_reaudit_20260906.md) | 历史再认证规则与统一检查入口的初版说明 | 最新缺口与命令以上一行为准 |
| [tracker_validation_hardening_20260906.md](tracker_validation_hardening_20260906.md) | 验收校验、失败落盘与 latest 一致性修复 | 零设备回归；不重写既有 GPU/NPU 实测结果 |
| [NPU 最小适配报告规范](../docs/ADAPTER_REPORT_STANDARD.md) | T-076～T-079 共 32 个实际 adapter 的逐 case 报告入口、代码框、调用链与审计合同 | 32/32 已补齐；adapter 不能代替产品修复或 comparison verdict |
| [t076_t077_performance_20260903.md](t076_t077_performance_20260903.md) | 两批性能处置、backend 门禁、B2B capability 与四项 experimental OFF/ON 实测 | T-076 2测/3显式关闭免测；T-077 5/5 已处置、pending=0 |
| [t076_npu_completion_20260902.md](t076_npu_completion_20260902.md) | T-076 五个单元的 NPU/comparison 闭环及 addmm 运行态纠偏 | 正式闭环 5/5；1 个行为一致、4 个预期产品分歧 |
| [T-076 P-018 gate 分析](../issues/REF-addmm-contract-native/根因分析.md) / [候选验证](../issues/REF-addmm-contract-native/修复验证报告.md) | 社区 pattern、安装态 gate、必要调用链、live opt-out 与验证矩阵 | capability 候选已验证；正式安装态仍关闭 |
| [t076_pattern_gpu_npu_guide_20260902.md](t076_pattern_gpu_npu_guide_20260902.md) | T-076 每个 pattern/variant 的源码意图与 GPU/NPU 行为导读 | 20/20 variant 学习说明；P-018 候选单列 |
| [t077_gpu_preparation_20260902.md](t077_gpu_preparation_20260902.md) | 第二波 5 units / 11 direct cases / 17 variants 的人工映射、参数化入口与 GPU 交付 | 准备阶段历史；已被有效 reference supersede |
| [t077_gpu_reference_20260902.md](t077_gpu_reference_20260902.md) | T-077 GPU 文本 handoff 的 11/11 case、17/17 variant、环境、哈希与 1.3 FX 正文复核 | reference 已冻结；68 份关键正文可恢复 |
| [t077_npu_completion_20260902.md](t077_npu_completion_20260902.md) | T-077 五单元 NPU/comparison 闭环与 MM lowering 修复验证 | 正式闭环 5/5；候选尚未合入 |
| [T-077 small-MM 根因分析](../issues/REF-decompose-mm-native/根因分析.md) / [修复验证](../issues/REF-decompose-mm-native/修复验证报告.md) | 触发代码、必要调用栈、首个 lowering 分歧、修复代码与六变体复验 | 候选 `dfbcc25` 已验证、尚未合入 |
| [T-078 FP16 value=1 根因](../issues/REF-addcdiv-fma-codegen-native/根因分析.md) / [精度修复](../issues/REF-addcdiv-fma-codegen-native/FP16精度修复报告.md) / [GPU补证代码对照](../issues/REF-addcdiv-fma-fp16-value1-derived/GPU补证与NPU修复对照.md) | 不重融合合同、NPU quotient 舍入根因、修复前后FX/IR/codegen及正确GPU派生单例 | 9月14日GPU邻接已补齐；各自eager位级正确，保留PARTIAL_ALIGNED |
| [t077_pattern_gpu_npu_guide_20260902.md](t077_pattern_gpu_npu_guide_20260902.md) | T-077 pattern 意图、源码块、GPU/NPU 对照和修复代码 | 17/17 variant 已解释 |
| [t078_mapping_review_20260903.md](t078_mapping_review_20260903.md) | 第三批四单元的人工映射修正、源码意图和 GPU 合同 | 12 direct cases/20 variants 已准备，等待 reference |
| [t078_npu_completion_20260906.md](t078_npu_completion_20260906.md) | 第三批原冻结范围及 addcdiv 低精度历史 | 原12/12 GPU cases、4/4 NPU comparison有效；FP16 value=1当前补证闭环见上方9月14日对照报告 |
| [t079_t080_mapping_review_20260904.md](t079_t080_mapping_review_20260904.md) | 第四/五批七单元的人工映射、源码意图、性能来源与后端边界 | 17 direct cases/27 variants 已准备，等待 reference |
| [T-078 功能/性能 guide](../docs/T078_FUNCTION_PERFORMANCE_GUIDE.md) | 四单元功能 case、GPU/NPU 行为、修复与派生 benchmark 的源码化讲解 | 20/20 variants 已解释，含正式动态 verdict |
| [t079_gpu_reference_review_20260907.md](t079_gpu_reference_review_20260907.md) | T-079 1.3 review 的完整性、逐 case FX 与冻结边界 | GPU 复核时点 4/4 cases、14/14 variants 已冻结；后续状态见下一行 guide |
| [T-079 功能/性能 guide](../docs/T079_FUNCTION_PERFORMANCE_GUIDE.md) | 四个图改写/消除合同及性能证据解释 | GPU/NPU 功能、性能与 bmm 产品门禁均已完成 |
| [t080_gpu_reference_review_20260907.md](t080_gpu_reference_review_20260907.md) | T-080 分片 1.3 review 的完整性、逐单元 FX/原生断言与冻结边界 | 13/13 cases、13/13 variants 已冻结并进入正式 NPU 对照 |
| [T-080 结果与学习 guide](../docs/T080_RESULT_AND_LEARNING_GUIDE.md) | 三个 pattern 的代码、GPU/NPU 行为、适配、修复、性能与产品处置 | 3/3 units 已闭环；Scatter 默认关闭、Softmax 免测、Constructor 中性 |
| [T-081～T-083 GPU复核](t081_t083_gpu_reference_review_20260908.md) | 新三批11 cases/24 variants的输入校验、FX行为、证据范围与观察器最小修复 | 7个单元GPU分母已冻结；该报告的GPU时点边界保留 |
| [T-081～T-083 NPU/性能闭环](t081_t083_npu_function_performance_20260908.md) | 7单元源码、调用栈、GPU/NPU行为、真实问题、最小适配和六臂性能 | NPU功能7/7、性能处置7/7；T-083为真实HCCL双rank |
| [T-081 测例 guide](../docs/T081_FUNCTION_PERFORMANCE_GUIDE.md) / [T-082](../docs/T082_FUNCTION_PERFORMANCE_GUIDE.md) / [T-083](../docs/T083_FUNCTION_PERFORMANCE_GUIDE.md) | 新三批2/2/3单元的源码、实际GPU/NPU行为、功能/性能测例 | 7个单元全部完成正式处置 |
| [T-078 邻接补证与 T-084～T-086 GPU复核](t078_t084_t086_gpu_reference_review_20260910.md) | handoff完整性、`device_execution=false`边界、逐单元FX行为与真实单/双卡范围 | T-084～T-086共8/8 cases、11/11 variants、5单元GPU分母冻结；T-078误命名包不关闭FP16 value=1缺口 |
| [T-084 测例 guide](../docs/T084_FUNCTION_PERFORMANCE_GUIDE.md) / [T-085](../docs/T085_FUNCTION_PERFORMANCE_GUIDE.md) / [T-086](../docs/T086_FUNCTION_PERFORMANCE_GUIDE.md) | 5个单元的源码合同、功能/性能来源、OFF/ON和延期理由 | GPU/NPU功能与性能处置已完成 |
| [REF-mm-plus-mm-native NPU 复现报告](../issues/REF-mm-plus-mm-native/复现报告.md) | 原生直接 `NO_TESTS`、最小 adapter、4/4 NPU 目标合同和 graph-mode 证据 | 统一 comparison 已落盘；`BEHAVIOR_UNCHANGED` |
| [REF-pad-mm-dynamic-m-native NPU 复现报告](../issues/REF-pad-mm-dynamic-m-native/复现报告.md) | 原生 `NO_TESTS`、TRITON-only lowering 阻断、产品 gate baseline | 归属单元已正式闭环 |
| [t076_gpu_reference_20260901.md](t076_gpu_reference_20260901.md) | 13/13 GPU direct valid、环境、哈希与 1.3 FX 正文复核 | reference 已冻结；80 份关键正文可恢复，完整大 artifacts 保留在 GPU |
| [t076_reference_runner_20260831.md](t076_reference_runner_20260831.md) | 首批 direct GPU/reference plan、runner、schema 与静态验证 | runner 已完成 GPU 执行；设计边界保留 |
| [t075_acceptance_unit_mapping_review_20260831.md](t075_acceptance_unit_mapping_review_20260831.md) | 首批 5 个 acceptance units 的 contract/variant 与证据角色人工复核 | 静态 mapping 完成；reference 已冻结 |
| [t074_upstream_pass_test_index_20260829.md](t074_upstream_pass_test_index_20260829.md) | T-074 registration candidate、community test 和 provisional unit 总结 | 静态 v1，不是冻结分母 |
| [candidate_test_index.csv](upstream_pass_test_index_20260829/candidate_test_index.csv) | 207 条 candidate/control 行与测试映射 | inventory 输入，动态状态均未运行 |
| [acceptance_units.csv](upstream_pass_test_index_20260829/acceptance_units.csv) | 188 个 heuristic 去重单元 | 158 eligible 仍为 provisional |
| [t056_triton_experimental_inventory_20260826.md](archive/legacy-20260820-0828/t056_triton_experimental_inventory_20260826.md) | T-074 的静态路由来源 | candidate discovery 辅助证据 |
| [triton_experimental_20260826/](archive/legacy-20260820-0828/triton_experimental_20260826/) | config、feature family 和 route CSV | previous inventory，不是任务分母 |

## 历史阶段导航

| 阶段 | 文件范围 | 说明 |
| --- | --- | --- |
| 2026-08-20～2026-08-28 旧工作线 | [archive/legacy-20260820-0828/](archive/legacy-20260820-0828/) | P0、MM/pad、torch_npu custom pass、DVM/MLIR、attention 与 experimental feature-family 历史证据；原结论不改写 |

## 使用规则

1. 先从文件名日期和报告正文确认真实环境；
2. old Benchmark/isolated venv 结果不得自动升级为当前 Pass 环境 verdict；
3. 历史报告中的 “pass 数” 可能使用旧 inventory 口径，引用时必须同时说明是 registration、
   pattern family、custom pass 还是当前 acceptance unit；
4. 新 tracker 产物必须记录生成时间戳、schema version、source commit 和环境指纹；
5. 报告增长时优先更新本索引，不在根 README 重复堆叠完整实验叙述。
