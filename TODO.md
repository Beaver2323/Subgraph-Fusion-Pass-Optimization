# Triton Experimental 原生优化持续兼容性跟踪 TODO

> 更新时间：2026-09-04 09:08 CST（UTC+08:00）
> 状态：T-076/T-077 已完成；T-078～T-080 的 11 个新单元、29 cases、47 variants、性能合同与测例讲解已准备，等待 GPU reference；MM lowering 修复已验证、尚未推送/合入。
> 约束：只在原生入口真实阻断后创建 case-specific adapter，不新增大规模 pass 测例。

## 任务计数规则

- registration candidate 只用于静态 inventory 和 coverage 查漏；
- community test 是 expected behavior 的主要事实源；
- acceptance unit 是跟踪、比较和 verdict 的基本单位；
- 一个 registration 可以展开多个 pattern/variant，也可能与其他 registration 共同服务一个 contract；
- 只有人工审核并冻结的 acceptance unit 才能进入完成率分母；
- T-074 当前 188/158 均为 provisional；首批冻结 denominator 的正式闭环数为 5/5。

## P0-A：仓库与文档收束

- [x] 将仓库逻辑定位改为持续兼容性 tracker；
- [x] 根目录收束为 README、TODO、WORKFLOW 三个活动入口；
- [x] 将当前文档归入 `docs/`，旧计划/检查点/交接归入 `docs/archive/`；
- [x] 为 `report/` 增加证据导航，不删除历史报告；
- [x] 统一中文活动文档并增加 2026-08-31 时间戳；
- [x] 明确 registration、pattern、community test、acceptance unit、variant 的区别；
- [x] 明确 GPU 无 Agent、NPU 为控制节点的双机工作流；
- [ ] 是否重命名远端仓库另行决策；当前保留原名以避免破坏链接。

## P0-B：T-075 acceptance-unit mapping 收敛

首批 schema/mapping 已完成；B3 的规模化人工审核继续待办，不阻塞首批 T-076 runner。

### B1. 冻结 schema

- [x] 定义 `acceptance_unit_id`，不得从 registration 数量直接推导；
- [x] 定义 `contract_name` 和预期 transformation/behavior；
- [x] 记录 upstream source/test/commit；
- [x] 记录 registration/pattern evidence，但将其标记为辅助证据；
- [x] 定义 `variant_id` 及正例、负例、guard、regression 分支；
- [x] 定义 `tracking_mode`：`direct`、`adapter`、`extracted`；
- [x] `extracted` 必须记录 `extraction_reason` 和 `local_deviation`；
- [x] 定义 `review_status`：`needs-review`、`mapped`、`frozen`、`retired`；
- [x] 定义 denominator 是否进入分母及理由；
- [x] 定义 GPU/NPU/result/failure/repair 字段。

### B2. 复核首批 5 个单元

- [x] `mm_plus_mm`：以 community `test_mm_plus_mm` 为事实源，确认 same-K/different-K 是
  一个 contract 的 variants 还是独立 acceptance units；
- [x] pad-mm：确认 mm/bmm/addmm 是三个独立 contracts，保留 positive/negative guard tests；
- [x] add+mm → addmm：确认 symbolic-scalar 负例与正常融合的 contract/variant 关系；
- [x] 对每个单元记录 upstream/GPU expected match 和现有历史 NPU 证据边界；
- [x] 不把旧自建 runner 直接升级为 community-test baseline。

### B3. 收敛未确认映射

- [ ] 先按 acceptance unit 审核 48 个 `no-test-found`，同时保留其对应的 54 条 candidate 行；
- [ ] 再审核 29 个 indirect 单元，保留其对应的 33 条 candidate 行；
- [ ] 优先从 community tests 反向识别 optimization contracts；
- [ ] 判断多个 registration 是否属于一个 contract；
- [ ] 判断一个 registration 是否展开多个必须分别验证的 variants；
- [ ] 无法确认时标记 `needs-review`，不得猜测；
- [ ] 审核完成后才决定是否重新生成 T-074 acceptance-unit 输出；
- [x] 冻结第一版 denominator=5；188/158 仍为 T-074 provisional inventory，不与冻结数混算。

## P0-C：Manifest 与映射文件

T-075 schema 确认后执行。

- [x] 创建 `upstream/manifest.yaml`，主键是 acceptance unit，不是 pass/registration 行；
- [x] 创建 `upstream/pass_map.yaml`，保存 registration/pattern/test 的多对多证据；
- [x] 记录 source commit、source test 和 reference device 合同；具体 GPU 指纹由 T-076 采集；
- [x] 记录 expected reference match、重要 counter/assertion；
- [ ] 建立 `adapters/{pre_grad,joint_graph,post_grad}`；
- [ ] direct 可运行的 community test 不保留长期复制；
- [x] adapter 合同限定为只注入 device/backend/input/artifact capture；
- [ ] extracted case 必须建立 upstream drift 检查；首批当前不使用 extracted；
- [x] 先纳入首批已审核单元，不一次导入全部 provisional 单元。

## P0-D：T-076 GPU/reference runner

GPU 机器没有 Agent，runner 必须可由人工一次执行完整批次。

- [x] 从 manifest 枚举 5 个 acceptance units、20 个 variants 和 13 个 community cases；
- [x] runner 支持 direct/adapter/extracted 入口合同；adapter/extracted 必须先带 direct blocker；
- [x] 保存 PyTorch commit、source test、tracking mode 与环境指纹；
- [x] 保存 execution、预期 match/counter/FileCheck 断言及其社区测试内验证状态；
- [x] 保存 FX before/after、artifact inventory 与归一化稳定 signature；
- [x] 保存 correctness、runner error、stdout/stderr 和 structured trace；
- [x] 目标优化可启用且功能/artifact 门禁通过后才允许 benchmark；明确关闭的单元直接免测；T-076 为 2 个同 backend 实测和 3 个关闭免测；
- [x] 单个 case 失败、skip 或超时不终止整个 suite；
- [x] 生成结构化 `reference_summary.json/.md`；
- [x] 提供 GPU 人工操作说明：`git pull`、静态校验、整批/单 case、打包/回传；
- [x] 固定 GPU `z00824525`/sudo、A100/R550、`/data`、CUDA 12.6.3、pip venv Python 3.12 和精确 source build 合同；
- [x] 环境指纹补充执行用户、CUDA/compat 路径、Driver API、宿主驱动和缓存路径；
- [x] 增加受限机器长期复用的 `export_reference_text.py`，不修改原始 run 即可导出可复制 JSON；
- [x] 不要求 GPU 机器进行交互分析或自动 Git 决策；
- [x] GPU 机器完成 `/data/z50063656` 环境安装和精确 commit 运行时验真；
- [x] GPU 机器执行 13 个 direct cases，13/13 passed 且 `reference_valid=true`；
- [x] 通过文本 handoff 回传结构化摘要与哈希，并在 NPU 控制节点完成证据复核；
- [x] 13 个 direct 均有效，没有 blocker，不创建 GPU adapter；
- [x] 5 个单元更新为 `review_status=frozen`、`denominator_eligible=yes-frozen`。

## P0-E：统一 result schema

T-076 已完成 reference、统一 NPU/comparison schema 和 5/5 单元结果落盘。

- [x] 环境：upstream/torch_npu/Triton/CANN/driver/SoC commit 或版本；
- [x] 输入：dtype、shape、stride、dynamic、forward/backward；
- [x] reference：execution、matched、signature、correctness、latency/未配置状态；
- [x] NPU control：enabled/disabled/guarded/patched；
- [x] NPU 链路：graph、pattern、replacement、decomposition、lowering、scheduler、codegen；
- [x] `runtime_path`：triton/extern/fallback/mixed；
- [x] `first_divergence`、`root_cause`、`recommended_action`；
- [x] pass-on/pass-off 结果（仅需要时；不需要时必须记录原因）；
- [x] final verdict、failure layer、repair status；
- [x] schema 版本和生成时间戳。

## P0-F：NPU runner 与 comparison

- [x] 从同一 manifest 运行 NPU；
- [x] reference 缺失或 invalid 时停止 compatibility verdict；
- [x] 使用 `options={"npu_backend": "triton_experimental"}`；
- [x] 首批已执行 case 使用 fresh process 隔离 backend 与 pass-on/pass-off；
- [x] 首个正式 comparison 采集 trigger/counter、FX、IR、generated code、fallback/graph-break；
- [x] correctness 通过后才 benchmark；T-077 B2B 先做模板选模门禁，decompose 三项先通过正确性再执行 OFF/ON；
- [ ] 实现 reference previous/current、NPU previous/current、reference/NPU 三组比较；
- [x] schema 支持输出 `UPSTREAM_CHANGED`、`NPU_REGRESSION`、`NEWLY_SUPPORTED`、
  `PERF_IMPROVED`、`PERF_REGRESSED`、`BEHAVIOR_UNCHANGED`。

首个单元进展：

- [x] `REF-mm-plus-mm-native` 先尝试不修改上游入口，确认 `NO_TESTS` 而非 PASS；
- [x] 最小 adapter 仅注入 `device=npu`、`npu_backend=triton_experimental` 和目标专属断言；
- [x] same-K、different-K 两个正例和两个 output-shape mismatch 负例 4/4 通过；
- [x] 保存 counter、generated code、FX/IR/output_code、trace 和环境选择；
- [x] 将首个单元的已有 JSON 转入统一 NPU/comparison schema，verdict 升级为 `BEHAVIOR_UNCHANGED`。

pad-mm 单元进展：

- [x] `REF-pad-mm-dynamic-m-native` 原生入口确认为 `NO_TESTS`；
- [x] 记录 GPU-only `TRITON` 候选在 NPU gate baseline 下触发的 lowering `NoValidChoicesError`；
- [x] 最小扩为 `TRITON,ATEN`，不绕过 `disable_pad_mm`，correctness 通过；
- [x] 记录 `disable_pad_mm=true`、`shape_padding=false`、目标计数 0/0，为 `EXPECTED_DISABLED`；
- [x] original-aten、stride、exclusion cases 均按原生优先完成；
- [x] 形成 `AU-pad-mm-mm` 单元级 NPU/comparison，verdict 为 `EXPECTED_PRODUCT_DIVERGENCE`。

T-077 GPU 准备：

- [x] 人工复核第二波 5 个 acceptance units；
- [x] 建立 11 个 direct cases、17 个 variants 的独立 manifest/reference plan；
- [x] 精确展开 decompose bmm/mm 参数化生成名称并排除 CPU-only 方法；
- [x] 提供 `run_t077_reference_all.sh`、GPU 中文说明和文本回传路径；
- [x] 在 GPU 执行完整 T-077 suite，11/11 valid 后冻结第二波 denominator；
- [x] 完成 Gumbel、B2B GEMM 的 NPU/comparison 与 pattern 导读；
- [x] 在空闲 NPU 上清洁重跑 decompose-BMM，并完成 decompose-MM 六个合同和 dynamic addmm 原始大 M 合同；
- [x] 定位 decompose-MM 阈值负例的 small-mm pointwise 反向 lowering 回归；
- [x] 在独立 torch_npu worktree 实现 NPU-only 幂等 guard，定向单测 2/2、完整 MM 合同 6/6 通过；
- [x] 提供 GPU pull 后按 `T-076`/`T-077` 一键运行、自动识别 timestamp、自动文本导出与 `latest` 入口；
- [x] T-077 Gumbel 在同一 `triton_experimental` backend 完成三轮 fresh-process OFF/ON；host p50/p99 改善 45.79%/45.75%；
- [x] T-077 B2B/decompose 四项完成最小 capability 评估：B2B 0 个模板获选，decompose 三项均正确但性能回退，统一保留 NPU guard；
- [ ] 评审并决定是否推送/合入本地候选 `dfbcc25b76743ea6c1c5cd61b6b30f0a910148a6`。

## P0-H：T-076/T-077 性能阶段

- [x] T-xxx 任务定义为一批 acceptance units 的完整流程，不另建 T-078 承担前两批性能债；
- [x] 历史性能复用必须同时匹配源码、输入、backend、gate 生命周期与测量方法；
- [x] T-076 `mm_plus_mm` 与 P-018 addmm 复用 experimental 同后端实测；
- [x] 三类 pad 的 default bypass 历史数据降为不计入诊断，不迁移成 experimental 结论；
- [x] T-077 Gumbel 六个 fresh worker 正确性、counter、host/device timing 与峰值内存完整；
- [x] 新增 `run_npu_performance_task.sh --task T-076|T-077`；
- [x] T-077 性能处置 5/5：measured=4、capability-assessed=1、pending=0；未把 device guard 冒充显式关闭免测。

## P0-G：首批跟踪闭环

- [x] negative case：从 pad/addmm 显式关闭控制中选择，记录 expected disabled/guarded 行为；
- [x] positive case：优先选择 community test 直接、未被 NPU 禁用的简单 contract；
- [x] 至少一个 case 输出可复核的 `first_divergence/root_cause`；
- [x] 建立 reference baseline 和 NPU baseline；
- [x] 生成第一版 compatibility matrix 和 changes；
- [x] 将失败 acceptance units 写入 repair queue；T-076 addmm 后续因安装态 gate 证据完成纠偏并移出 open queue。

## P1：Repair 与 regression

- [x] 建立 `regressions/known_issues.yaml` 和 `fixed_issues.yaml`；
- [x] 记录原 failure artifacts、root cause 和修复层；fixed commit 在修复后补录；
- [x] 按最早分歧层选择 config/graph/pattern/decomposition/lowering/scheduler/codegen/runtime 修复；T-077 MM 已在 lowering guard 修复；
- [x] 禁止用 FX workaround 掩盖纯 scheduler/codegen 问题；T-077 保留原 FX/contract，仅修复 lowering 选择；
- [ ] 区分 `SUPPORTED_NATIVE`、`SUPPORTED_FALLBACK` 和 `UNSUPPORTED`；
- [x] 修复后使用同一 tracked community contract 回归；T-077 MM 六变体 6/6 通过；
- [ ] 保存历史 compatibility 到 `results/history/`；
- [ ] PyTorch/torch_npu 升级后自动检查 fixed issue 是否重新失败。

## P2：规模化与持续看护

- [x] T-078 从 no-test-found/indirect inventory 反向审核 4 个 acceptance units，修正 addcdiv no-test 与 addmm/baddbmm 混映射；
- [x] T-078 建立 12 个 direct community cases、20 个 variants、独立 manifest/reference plan 与 GPU 一键入口；
- [x] 修正 `copy_tests` 静态展开：真实 GPU 方法追加 `_cuda`，并纠正 T-078 两个 addcdiv nodeid；
- [x] T-079 建立 4 个单元、4 个 direct cases、14 个 variants，覆盖 bmm→mm 与三类 cat/split 合同；
- [x] T-080 建立 3 个单元、13 个 direct cases、13 个 variants，覆盖 const-scatter、prepare-softmax 与 constructor mover；
- [x] T-078～T-080 逐单元补齐性能计划：功能门禁、社区 benchmark/功能例派生来源、目标级 OFF/ON、三轮交错、Event/host/内存与负例边界；
- [x] T-078～T-080 增加中文功能/性能测例讲解，并用零设备 validator 与 reference wrapper 联检；
- [ ] GPU 执行 T-078 12/12 direct cases；全部 reference valid 后冻结 4 个单元；
- [ ] GPU 执行 T-079 4/4 direct cases；全部 reference valid 后冻结 4 个单元；
- [ ] GPU 执行 T-080 13/13 direct cases；全部 reference valid 后冻结 3 个单元；
- [ ] T-078 内完整执行 reference → NPU → compare → 条件 repair → performance；
- [ ] T-079/T-080 分别完成 reference → NPU → compare → 条件 repair → performance；
- [ ] 建立 upstream source/test/mapping drift 检测；
- [ ] 支持一条命令运行 NPU suite；
- [ ] 支持一条命令生成 comparison report；
- [ ] 评估 CI/定期任务；
- [ ] 将稳定 verdict 反馈到 NPU capability registry 和条件 guard。

## 立即执行顺序

1. [x] T-075：冻结首批 acceptance-unit schema 并复核 5 个单元；
2. [x] T-076：生成 GPU/reference runner、reference 子集 schema 和操作说明；
3. [x] GPU 上执行 13 个原生 community cases，必要时才进入最小 adapter；
4. [x] 接收并复核 GPU 文本 artifacts；no-test-found 和 indirect 映射仍继续审核；
5. [x] reference 有效后冻结首版 denominator，并启动 NPU 单 case 执行；
6. [x] 执行 NPU、compare、failure classification（正式闭环 5/5）；
7. [x] 复核 repair queue；`AU-post-grad-addmm` 纠偏为预期产品分歧，P-018 候选完成精确上游合同验证。
8. [x] T-076、T-077 性能均已完成处置；T-078 留给下一批。
9. [x] T-078 静态映射与 GPU runner 准备完成；等待 GPU 人工执行并回传 latest 文本 handoff。
10. [x] T-079/T-080 静态映射、独立 runner 与一键任务入口准备完成；GPU 按 T-078→T-079→T-080 顺序执行即可。
11. [x] 三批性能准备收束到各自 T：T-080 优先复用社区 benchmark，T-078/T-079 明示为社区功能例派生；NPU 功能门禁前性能入口保持锁定。

## 第一阶段完成标准

```text
acceptance-unit schema 已冻结
+
至少 1 个 negative contract 和 1 个 positive contract
+
upstream manifest / registration-test map
+
GPU reference baseline / NPU baseline
+
runtime_path / first_divergence / root_cause
+
compatibility matrix / previous-current diff
+
repair queue 与 regression 入口
```
