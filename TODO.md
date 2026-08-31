# Triton Experimental 原生优化持续兼容性跟踪 TODO

> 更新时间：2026-08-31 17:50 CST（UTC+08:00）
> 状态：文档与工作线校准完成，等待 T-075 acceptance-unit mapping 收敛。
> 约束：在 T-075 完成前，不新增大规模 pass 测例，不启动 GPU/NPU 批量执行。

## 任务计数规则

- registration candidate 只用于静态 inventory 和 coverage 查漏；
- community test 是 expected behavior 的主要事实源；
- acceptance unit 是跟踪、比较和 verdict 的基本单位；
- 一个 registration 可以展开多个 pattern/variant，也可能与其他 registration 共同服务一个 contract；
- 只有人工审核并冻结的 acceptance unit 才能进入完成率分母；
- T-074 当前 188/158 均为 provisional，正式闭环数为 0。

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

这是当前唯一立即执行项。

### B1. 冻结 schema

- [ ] 定义 `acceptance_unit_id`，不得从 registration 数量直接推导；
- [ ] 定义 `contract_name` 和预期 transformation/behavior；
- [ ] 记录 upstream source/test/commit；
- [ ] 记录 registration/pattern evidence，但将其标记为辅助证据；
- [ ] 定义 `variant_id` 及 dtype/shape/layout/dynamic/forward/backward 维度；
- [ ] 定义 `tracking_mode`：`direct`、`adapter`、`extracted`；
- [ ] `extracted` 必须记录 `extraction_reason` 和 `local_deviation`；
- [ ] 定义 `review_status`：`needs-review`、`mapped`、`frozen`、`retired`；
- [ ] 定义 denominator 是否进入分母及理由；
- [ ] 定义 GPU/NPU/result/failure/repair 字段。

### B2. 复核首批 5 个单元

- [ ] `mm_plus_mm`：以 community `test_mm_plus_mm` 为事实源，确认 same-K/different-K 是
  一个 contract 的 variants 还是独立 acceptance units；
- [ ] pad-mm：确认 mm/bmm/addmm 是三个独立 contracts，保留 positive/negative guard tests；
- [ ] add+mm → addmm：确认 symbolic-scalar 负例与正常融合的 contract/variant 关系；
- [ ] 对每个单元记录 upstream/GPU expected match 和现有历史 NPU 证据边界；
- [ ] 不把旧自建 runner 直接升级为 community-test baseline。

### B3. 收敛未确认映射

- [ ] 先按 acceptance unit 审核 48 个 `no-test-found`，同时保留其对应的 54 条 candidate 行；
- [ ] 再审核 29 个 indirect 单元，保留其对应的 33 条 candidate 行；
- [ ] 优先从 community tests 反向识别 optimization contracts；
- [ ] 判断多个 registration 是否属于一个 contract；
- [ ] 判断一个 registration 是否展开多个必须分别验证的 variants；
- [ ] 无法确认时标记 `needs-review`，不得猜测；
- [ ] 审核完成后才决定是否重新生成 T-074 acceptance-unit 输出；
- [ ] 冻结第一版 denominator，并记录从 188/158 到新数量的可审计变化。

## P0-C：Manifest 与映射文件

T-075 schema 确认后执行。

- [ ] 创建 `upstream/manifest.yaml`，主键是 acceptance unit，不是 pass/registration 行；
- [ ] 创建 `upstream/pass_map.yaml`，保存 registration/pattern/test 的多对多证据；
- [ ] 记录 source commit、source test 和 reference device；
- [ ] 记录 expected reference match、重要 counter/assertion；
- [ ] 建立 `adapters/{pre_grad,joint_graph,post_grad}`；
- [ ] direct 可运行的 community test 不保留长期复制；
- [ ] adapter 只注入 device/backend/input/artifact capture；
- [ ] extracted case 必须建立 upstream drift 检查；
- [ ] 先纳入首批已审核单元，不一次导入全部 provisional 单元。

## P0-D：T-076 GPU/reference runner

GPU 机器没有 Agent，runner 必须可由人工一次执行完整批次。

- [ ] 从 manifest 枚举 acceptance units/variants；
- [ ] 支持 direct/adapter/extracted；
- [ ] 保存 PyTorch commit、source test、tracking mode；
- [ ] 保存 execution、match count、counter/assertion；
- [ ] 保存 FX before/after 与稳定 signature；
- [ ] 保存 correctness、traceback、stdout/stderr；
- [ ] 功能门禁通过后才运行 benchmark；
- [ ] 单个 case 失败不终止整个 suite；
- [ ] 生成结构化 reference summary；
- [ ] 提供 GPU 人工操作说明：`git pull`、运行脚本、打包/提交 artifacts；
- [ ] 不要求 GPU 机器进行交互分析或自动 Git 决策。

## P0-E：统一 result schema

- [ ] 环境：upstream/torch_npu/Triton/CANN/driver/SoC commit 或版本；
- [ ] 输入：dtype、shape、stride、dynamic、forward/backward；
- [ ] reference：execution、matched、signature、correctness、latency；
- [ ] NPU control：enabled/disabled/guarded/patched；
- [ ] NPU 链路：graph、pattern、replacement、decomposition、lowering、scheduler、codegen；
- [ ] `runtime_path`：triton/extern/fallback/mixed；
- [ ] `first_divergence`、`root_cause`、`recommended_action`；
- [ ] pass-on/pass-off 结果（仅需要时）；
- [ ] final verdict、failure layer、repair status；
- [ ] schema 版本和生成时间戳。

## P0-F：NPU runner 与 comparison

- [ ] 从同一 manifest 运行 NPU；
- [ ] reference 缺失或 invalid 时停止 compatibility verdict；
- [ ] 使用 `options={"npu_backend": "triton_experimental"}`；
- [ ] fresh process 隔离 backend 与 pass-on/pass-off；
- [ ] 采集 trigger/counter、FX、IR、generated code、fallback/graph-break；
- [ ] correctness 通过后才 benchmark；
- [ ] 实现 reference previous/current、NPU previous/current、reference/NPU 三组比较；
- [ ] 输出 `UPSTREAM_CHANGED`、`NPU_REGRESSION`、`NEWLY_SUPPORTED`、
  `PERF_IMPROVED`、`PERF_REGRESSED`、`BEHAVIOR_UNCHANGED`。

## P0-G：首批跟踪闭环

- [ ] negative case：从 pad/addmm 显式关闭控制中选择，记录 expected disabled/guarded 行为；
- [ ] positive case：优先选择 community test 直接、未被 NPU 禁用的简单 contract；
- [ ] 至少一个 case 输出可复核的 `first_divergence/root_cause`；
- [ ] 建立 reference baseline 和 NPU baseline；
- [ ] 生成第一版 compatibility matrix 和 changes；
- [ ] 将失败 acceptance units 写入 repair queue。

## P1：Repair 与 regression

- [ ] 建立 `regressions/known_issues.yaml` 和 `fixed_issues.yaml`；
- [ ] 记录原 failure artifacts、root cause、修复层和 fixed commit；
- [ ] 按最早分歧层选择 config/graph/pattern/decomposition/lowering/scheduler/codegen/runtime 修复；
- [ ] 禁止用 FX workaround 掩盖纯 scheduler/codegen 问题；
- [ ] 区分 `SUPPORTED_NATIVE`、`SUPPORTED_FALLBACK` 和 `UNSUPPORTED`；
- [ ] 修复后使用同一 tracked community contract 回归；
- [ ] 保存历史 compatibility 到 `results/history/`；
- [ ] PyTorch/torch_npu 升级后自动检查 fixed issue 是否重新失败。

## P2：规模化与持续看护

- [ ] 第一批扩展到 5～10 个已冻结 acceptance units，而不是“5～10 个 registration”；
- [ ] 建立 upstream source/test/mapping drift 检测；
- [ ] 支持一条命令运行 NPU suite；
- [ ] 支持一条命令生成 comparison report；
- [ ] 评估 CI/定期任务；
- [ ] 将稳定 verdict 反馈到 NPU capability registry 和条件 guard。

## 立即执行顺序

1. T-075：冻结 acceptance-unit schema；
2. 用 schema 复核首批 5 个单元；
3. 审核 no-test-found 和 indirect 映射；
4. 冻结首版 denominator 与 manifest；
5. T-076：生成 GPU/reference runner 和操作说明；
6. 等待/接收 GPU artifacts，同时完成不依赖 GPU 的 NPU runner 静态准备；
7. 执行 NPU、compare、failure classification；
8. 创建 repair queue 并逐项闭环。

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
