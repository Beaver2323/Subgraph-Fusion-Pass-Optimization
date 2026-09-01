# PyTorch Inductor 原生优化到 NPU 的持续兼容性工作流

> 更新时间：2026-09-02 05:44 CST（UTC+08:00）
> 适用主线：PyTorch community-native Inductor optimization contract
> → NPU `triton_experimental` compatibility tracker。

## 1. 项目定位

本仓库持续回答两个问题：

1. 当前冻结 PyTorch commit 中，community test 定义的 upstream optimization contract 是什么；
2. NPU `triton_experimental` 是否保持该 contract，并以什么运行路径、正确性和性能实现。

它不是 registration 数量统计器，也不是“一条 registration 写一个 NPU case”的测试集合。
源码 inventory 用于发现和查漏；community tests 和 reference baseline 定义预期；
acceptance unit 承载最终验证和版本跟踪。

## 2. 数据模型

### 2.1 registration site/candidate

源码中可静态识别的 decorator、registry、pipeline、hook 或调用入口。它只回答“哪里可能注册或
激活优化”，不能单独证明实际 pattern 数量或独立 contract 数量。

### 2.2 pattern

PatternMatcher 实际尝试匹配的模式。一个 registration 可以动态生成多个 pattern；同一 contract
也可能由多个 registration、helper 或 stage 共同实现。

### 2.3 community test

PyTorch upstream testcase/helper，是 expected behavior 的主要事实源。优先复用其图、输入、
正负例、counter 和断言；只有无法满足 NPU 后端采集需求时，才增加薄 adapter 或最小 extracted case。

### 2.4 acceptance unit

一个可独立回答以下问题的 upstream optimization contract：

> 该 upstream contract 在 NPU `triton_experimental` 上是否成立？

每个单元可以包含多个 registration evidence、community tests 和 variants，但必须拥有单一、明确、
可判定的 expected behavior。registration、pattern、pass 和 acceptance unit 不存在天然一一对应。

### 2.5 variant

同一 contract 下必须单独验证的重要分支，例如 dtype、shape、stride/layout、static/dynamic、
forward/backward、正例/负例或 capability guard。variant 不应被机械拆成独立 acceptance unit；
只有 expected behavior 或最终 verdict 可独立变化时才拆分。

## 3. 事实源优先级

```text
PyTorch upstream community test / helper
→ upstream source 与 registration/pattern evidence
→ GPU/reference baseline
→ NPU triton_experimental result
→ 历史本地 runner/benchmark（辅助证据）
```

旧自建图和历史 benchmark 可以帮助排序、复用性能方法或解释失败，但不能覆盖当前 community
contract，也不能在没有 current reference/NPU mapping 时升级为 tracker verdict。

## 4. 标准流水线

```text
upstream change / community test discovery
→ registration inventory 辅助 coverage 检查
→ acceptance-unit mapping 与人工审核
→ GPU/reference baseline
→ NPU triton_experimental execution
→ GPU/NPU compare
→ first-divergence failure classification
→ backend repair / capability decision
→ same-contract regression
→ version history
```

性能必须位于 trigger、正确性、fallback 和 graph-break 门禁之后。

## 5. 双机职责

### 5.1 GPU/reference 机器

GPU 机器没有 Agent，只做被动批量执行：

```text
人工 git pull 或接收 runner 包
→ 执行固定脚本
→ 生成结构化 artifacts 和 summary
→ 人工 push 或打包带回 NPU
```

GPU 机器不承担 testcase 设计、交互式归因、backend 修改或自动 Git 决策。runner 必须在单个 case
失败后继续执行，并保存足够 traceback、图、counter 和环境信息供 NPU 侧离线分析。

T-076 GPU 执行合同为 `z00824525`/sudo、A100/R550、`/data/z50063656` 上的 CUDA 12.6.3、
pip venv Python 3.12、cuDNN 9.25.1、Triton 3.8.0 和冻结 commit source build；compat 未启用。
GPU 测试从
`/data/z50063656/tmp` 发起；该路径是 NPU `/home/z50063656/tmp` 规则的显式机器级例外。

### 5.2 NPU 机器

NPU 机器是控制节点，负责：

- community test discovery 和 acceptance-unit mapping；
- manifest、adapter、GPU/NPU runner 和 schema；
- NPU 执行与 artifacts 采集；
- reference/NPU comparison；
- first divergence、root cause 和 repair queue；
- 产品修改、回归、文档和 Git 交付。

工作流不得依赖同一个 Agent 同时控制两台机器。GPU artifacts 暂未返回时，NPU 侧可以继续
schema、mapping、runner 静态校验和已知历史证据整理，但不能给出 compatibility final verdict。

## 6. 仓库结构

当前活动结构：

```text
.
├── README.md
├── TODO.md
├── WORKFLOW.md
├── upstream/
│   ├── README.md
│   ├── manifest.schema.json
│   ├── manifest.yaml
│   ├── pass_map.yaml
│   ├── reference_plan.schema.json
│   └── reference_plan.yaml
├── issues/
│   └── REF-*/
│       ├── npu_adapter.py
│       └── 复现报告.md
├── results/current/
│   └── <acceptance-unit>/
│       ├── npu_result.json
│       └── comparison_result.json
├── regressions/
│   ├── known_issues.yaml
│   └── fixed_issues.yaml
├── schemas/
│   └── reference_result.schema.json
├── runners/
│   └── reference_runner.py
├── scripts/
│   ├── run_reference_all.sh
│   ├── run_t077_reference_all.sh
│   └── validate_tracker_data.py
├── docs/
│   ├── CURRENT_STATUS.md
│   ├── SCOPE_AND_CODE_MAP.md
│   ├── GUIDE.md
│   ├── REFERENCE_RUNNER_GPU.md
│   ├── CHANGE_CONTROL.md
│   ├── HISTORY.md
│   ├── requirements/
│   └── archive/
└── report/
    ├── README.md
    ├── t075_acceptance_unit_mapping_review_20260831.md
    ├── t074_upstream_pass_test_index_20260829.md
    └── upstream_pass_test_index_20260829/
```

T-076 已增加首批真实 runner/schema/result/repair queue；后续只在有真实产物时增量增加：

```text
adapters/{pre_grad,joint_graph,post_grad}/
runners/npu_runner.py
scripts/{check_upstream.py,run_npu_all.sh,compare_runs.py}
baselines/
results/history/
```

不先创建大量空目录；每个目录在首个真实产物进入时创建。

## 7. Tracking mode

| mode | 使用条件 | 允许的本地变化 |
| --- | --- | --- |
| `direct` | community test/helper 可直接调用 | 只做 runner 参数和 artifacts 采集 |
| `adapter` | 测试不能直接 import，但核心图可复用 | device/backend/input 注入和薄封装 |
| `extracted` | 原测试与框架强耦合，无法直接运行 | 最小镜像；必须记录提取原因和偏离 |

选择优先级固定为 `direct > adapter > extracted`。upstream source/test 改变后，adapter 和
extracted case 必须重新审查。

## 8. Acceptance-unit manifest

建议最小字段：

```yaml
- acceptance_unit_id: AU-post-grad-mm-plus-mm
  contract_name: mm_plus_mm fusion contract
  stage: post_grad
  upstream:
    commit: <pytorch_commit>
    sources:
      - torch/_inductor/fx_passes/post_grad.py
    tests:
      - test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_mm_plus_mm
  registration_evidence:
    - source: torch/_inductor/fx_passes/post_grad.py
      symbol: mm_plus_mm
  variants:
    - variant_id: same_k
    - variant_id: different_k
  tracking_mode: direct
  review_status: mapped
  denominator_eligible: pending
```

`pass_map.yaml` 保存 registration/pattern/test/acceptance unit 的多对多关系；manifest 只保留已审核
的 contracts。T-074 candidate CSV 作为 inventory 输入，不直接等价于 manifest。

T-075 已在 `upstream/` 落盘首批 5 个审核单元，共 20 个 variants、13 个 community test 引用。
T-076 的 13 个 GPU direct cases 全部有效后，它们已更新为 `frozen`/`yes-frozen`，构成首版
denominator=5。五个单元均已完成 NPU/comparison：1 个 `BEHAVIOR_UNCHANGED`、3 个
`EXPECTED_PRODUCT_DIVERGENCE`、1 个 `NPU_REGRESSION`，正式闭环为 5/5。回归单元已进入
`regressions/known_issues.yaml`。

## 9. 统一 result schema

每个运行结果至少包含：

```text
schema_version / generated_at
acceptance_unit_id / variant_id
upstream_commit / source_test / tracking_mode
environment fingerprint
input dtype / shape / stride / dynamic / direction
NPU control: enabled | disabled | guarded | patched
execution_success
match expectation / match count / counter
FX before/after / signature
replacement / decomposition
lowering / scheduler / codegen
runtime_path: triton | extern | fallback | mixed
correctness
performance（通过门禁后）
first_divergence
root_cause
recommended_action
final_verdict
repair_status
```

结构合同分别位于 `schemas/npu_result.schema.json` 与
`schemas/comparison_result.schema.json`；当前结果位于 `results/current/<acceptance-unit>/`（首个历史目录
保留 case ID）。
`scripts/validate_comparison_data.py` 使用标准库交叉检查 manifest、环境指纹、仓库工件 hash、
variant 完整性和 formal verdict 门禁，不导入 `torch`。回归结论允许“数值正确，但目标命中合同失败”，
此时 execution 保留 failed 且 repair status 必须进入修复流程。

禁止只输出笼统 PASS/FAIL。`torch.compile` 成功但走 fallback 时，不得标为完全 native supported。

## 10. Upstream 检查

每次 PyTorch commit 变化后检查：

- implementation/registration source 是否变化；
- community test/helper 是否变化、删除或重命名；
- expected counter、replacement 或 guard 是否变化；
- adapter/extracted mapping 是否漂移。

输出至少区分：

```text
UPSTREAM_UNCHANGED
UPSTREAM_CHANGED
UPSTREAM_TEST_CHANGED
UPSTREAM_TEST_REMOVED
UPSTREAM_MAPPING_BROKEN
```

reference 行为变化时，先更新 upstream contract，不能直接判为 NPU regression。

## 11. GPU/reference baseline

GPU runner 必须先尝试直接执行 community 原生测例或原生 helper，确认原图和原断言是否在 GPU
reference 上命中。只有 direct 路径被设备常量、backend 注入或 artifacts 采集阻塞时，才进入
最小 adapter；adapter 不得改变图、shape、dtype、正负例语义或 expected match。

GPU runner 对每个 acceptance unit/variant 保存：

```text
metadata.json
reference_result.json
fx_before.txt
fx_after.txt
stdout.log
stderr.log
benchmark.json（功能门禁通过后）
```

T-076 首批执行计划包含 13 个 `direct` community cases：20 个 variants 中 14 个由动态 case
覆盖，3 个 generated-registration 分支保留为静态结构证据，3 个 NPU product gate 在 reference
侧标记为不适用。部分 community case 只提供 contract 证据，不强行绑定到不对应的 variant。

每个 case 使用 fresh process，并配置独立 `TORCHINDUCTOR_CACHE_DIR`、
`TORCH_COMPILE_DEBUG_DIR` 和 `TORCH_TRACE`。有效 reference 同时要求原生测试通过和计划要求的
FX before/after 已捕获；全部 skip、未发现测试或缺失必需 artifact 均为 invalid。社区测试内部
未打印的 counter 不能由 runner 从 return code 推造，只能记录 expected assertion 与“原断言通过”。

`reference_plan.yaml` 若增加 `adapter` 或 `extracted` case，必须同时登记原 direct blocker 和新
entrypoint。当前没有 direct GPU artifacts，因此 `adapters/` 不创建。人工命令和回传合同见
[`docs/REFERENCE_RUNNER_GPU.md`](docs/REFERENCE_RUNNER_GPU.md)。

baseline 先比较 previous reference 与 current reference：match、FX signature 或行为变化时标记
`UPSTREAM_CHANGED`。reference invalid、missing 或 mapping broken 时，NPU 结果只能保留为运行证据，
不能形成 compatibility verdict。

## 12. NPU 执行

NPU 对同一 manifest 运行，并按顺序采集：

1. backend control：enabled/disabled/guarded/patched；
2. 输入 FX 与 reference 是否可比；
3. pattern match 和 extra_check；
4. replacement/decomposition；
5. lowering；
6. Inductor IR/layout；
7. scheduler/fusion；
8. Triton experimental codegen/compile；
9. runtime path；
10. correctness；
11. performance。

所有测试从 `/home/z50063656/tmp` 启动，source `Pass/activate_pass.sh`，并使用 fresh process
隔离 experimental backend 和 pass-on/pass-off。

## 13. First-divergence 归因

建议枚举：

```text
upstream
config
graph
pattern
replacement
decomposition
lowering
scheduler
codegen
triton_compile
runtime
correctness
performance
none
```

修复最早出现分歧的层，不根据最终错误位置盲目加特殊分支。

GPU/reference 命中但 NPU 不命中时依次判断：explicit disable、config、device guard、输入图差异、
extra_check、dtype/shape/layout 条件，最后才使用 `UNKNOWN_NOT_MATCHED`。不以提高命中率为理由强开
不安全或无收益的优化。

## 14. 兼容性与变化分类

NPU 当前状态至少区分：

```text
EXPECTED_DISABLED / DISABLED_BY_NPU
NPU_NOT_MATCHED
LOWERING_UNSUPPORTED
SCHEDULER_UNSUPPORTED
CODEGEN_UNSUPPORTED
TRITON_COMPILE_FAIL
RUNTIME_FAIL
CORRECTNESS_FAIL
SUPPORTED_NATIVE
SUPPORTED_FALLBACK
SUPPORTED_NO_GAIN
SUPPORTED_BENEFICIAL
PERF_REGRESSION
```

跨版本变化至少区分：

```text
UPSTREAM_CHANGED
NPU_REGRESSION
NEWLY_SUPPORTED
PERF_IMPROVED
PERF_REGRESSED
BEHAVIOR_UNCHANGED
```

只有 reference contract 稳定而 NPU 行为变坏时，才能标记 `NPU_REGRESSION`。

## 15. Correctness 和性能门禁

correctness 失败时优先比较 pass OFF/ON，检查 transformation、decomposition、lowering 和 guard。
correctness 未通过，不运行或不发布性能收益结论。

性能阶段固定比较：

- pass OFF vs ON；
- kernel/task/launch 数；
- Triton/extern/fallback 路径；
- p50/p99 与首编；
- 峰值内存和必要的 layout 代价。

只有部分 variants 有收益时，进入 capability guard 评估，不全开或全关。

## 16. Repair 与 regression

每个失败 acceptance unit 进入 repair queue，记录 failure artifact、first divergence、root cause、
建议修复层和优先级。修复后必须用同一 community contract/variant 重跑完整门禁，并写入
`fixed_issues.yaml` 的 fixed commit、expected result 和 reference commit。

后续 PyTorch 或 torch_npu 升级时，对 fixed issues 自动执行 regression tracking。

## 17. Definition of Done

一个 acceptance unit 只有满足以下条件才闭环：

- upstream contract、community test 和 source commit 已冻结；
- registration/pattern evidence 与 variants 已审核；
- tracking mode 和 local deviation 已记录；
- GPU/reference baseline 有效；
- 当前 Pass 环境 NPU baseline 已建立；
- trigger、replacement、lowering、scheduler、codegen、runtime path 明确；
- correctness 明确；
- 性能已完成或有明确“不适用/未执行”理由；
- final verdict、failure layer 和 repair status 已写入 matrix；
- 版本间 diff 可计算；
- 修复项有 regression expectation。

## 18. T-074 迁移边界

T-074 的 207 行 candidate CSV 是可复用 inventory，不需要因术语校准立即重扫。当前 188 个
acceptance units 来自 source/name 归一化和少量显式 semantic group，属于第一版 heuristic；它已
正确排除部分 registry/hook，但还不能证明每个分组都等于一个 upstream contract。

因此当前处理是：保留 candidate 数据和 188/158 provisional 统计，先用本工作流人工审核；
只有审核发现需要调整 grouping/schema 时，再增量生成新版 acceptance-unit/manifest 产物，并保留
旧版本作为可追溯输入。

T-075 首批复核没有改变 5 个单元的数量，但修正了 variants 和测试证据角色；T-074 v1 未覆盖。
T-076 GPU reference 已完成并通过文本 handoff 复核，13 个 direct cases 全部有效，不创建 GPU
adapter。首个 NPU unit 已生成统一 comparison 并正式闭环；pad-mm 首个产品 baseline 也已完成，
下一执行项是继续同一单元的 original-aten、stride、exclusion cases。48 个 `no-test-found` 和
29 个 indirect 单元仍待人工审核，但不阻塞首批 5 个冻结单元的 NPU 验证。
