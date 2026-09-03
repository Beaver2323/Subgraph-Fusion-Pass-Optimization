# 当前状态与 2026-08-31 工作线校准结论

> 更新时间：2026-09-03 08:05 CST（UTC+08:00）
> 校准输入：`831需求变更.md`、`831TODO_triton_experimental_pass_tracker.md`、
> `831WORKFLOW_triton_experimental_pass_tracker.md`。
> 当前阶段：T-076/T-077 已完成；T-078 已静态审核 4 个新单元并准备 GPU runner；MM lowering 修复已验证、尚未合入。

## 1. 总结

8 月 31 日需求与 8 月 29 日主线方向一致，都要求从 T-056/T-074 的静态 inventory 转向
PyTorch community-native Inductor compatibility。需要修正的不是基础候选数据，而是任务主键、
事实源、TODO 顺序和仓库信息架构：

- 主键从 registration candidate 改为经过人工审核的 acceptance unit；
- 事实源从源码扫描优先改为 community test + GPU/reference baseline 优先；
- T-074 从“待验证 pass 清单”重新定位为 inventory 与 provisional mapping；
- 动态顺序从“逐个 NPU case”改为 mapping → GPU baseline → NPU → compare → repair；
- GPU 无 Agent，NPU 为控制节点；
- 历史 default/custom/feature-family 结果保留，但不计入新主线完成率。

在此基础上，T-075 已完成首批 5 个单元的 schema、manifest、pass map 和人工复核：单元数仍为
5，共 20 个 variants、13 个 community test 引用。T-076 的 13 个 direct cases 已全部 passed 且
reference valid，文本 handoff 的环境、FX signature 和关键文件哈希已复核；5 个单元进入冻结
denominator，GPU 侧没有 adapter。统一 NPU/comparison schema、全部 5 个单元结果和零 torch
导入交叉校验已落盘，正式闭环为 5/5。
`AU-post-grad-mm-plus-mm` 已在当前 Pass 环境进入 NPU：原生直接入口因
upstream `HAS_GPU` 不包含 NPU 而为 `NO_TESTS`，随后 case-specific adapter 按 manifest
允许范围注入 NPU 设备、`triton_experimental` backend 和目标专属负向断言，
4/4 输入分支全部有效，unit-level verdict 为 `BEHAVIOR_UNCHANGED`。
`AU-pad-mm-mm` 的 dynamic-M、original-aten、stride、exclusion 四个 case 已按原生优先完成。
adapter 均未绕过 `disable_pad_mm`，原图 correctness 与 stride 合同有效；单元级 verdict 为
`EXPECTED_PRODUCT_DIVERGENCE`。T-077 第二波 5 个单元已经人工映射为 11 个 direct cases、
17 个 variants；GPU 11/11 direct cases 与 17/17 variants 均有效，5 个单元已经冻结。NPU 5/5
单元已全部形成正式 comparison：Gumbel 命中一致；B2B、decompose-BMM、dynamic addmm 的正例因
upstream CUDA/XPU device guard 形成预期产品分歧；decompose-MM 在目标 pass 未命中的负例路径发现
small-mm pointwise 反向 lowering correctness 回归。本地候选 `dfbcc25` 仅对 NPU 禁用该启发式，
定向单测 2/2、MM 六合同 6/6 通过，当前状态是 verified-not-merged。

`AU-pad-mm-bmm` 与 `AU-pad-mm-addmm` 同样完成产品关闭基线校验，均为
`EXPECTED_PRODUCT_DIVERGENCE`。`AU-post-grad-addmm` 已完成运行态纠偏：测试实际加载的 Pass
site-packages 中 `disable_addmm_fusion=True`，所以 matrix/vector 从 GPU `2/4` 到安装态 NPU
`0/0` 是显式产品 gate，结论修正为 `EXPECTED_PRODUCT_DIVERGENCE`。P-018 独立 wheel 已在冻结
commit 上恢复正例 `2/4`，并保持四类负例 `0/0`，作为已验证候选单列。

性能阶段已按任务内收束，不另立 T-078 补债。T-076 的 experimental 同后端证据为
`mm_plus_mm` 与 P-018 addmm；三类 pad 由 `disable_pad_mm=True` 显式关闭，性能免测；旧 default
backend 强制性能只作不计入诊断。T-077 Gumbel 已在 experimental 下完成 OFF/ON 各三轮，
host p50/p99 改善 45.79%/45.75%，NPU Event p50/p99 改善 46.69%/46.50%，升级为
`PERF_IMPROVED`。其余四项没有按“显式关闭免测”处理，而是完成测试态最小 capability：B2B 的
4 正例 matcher 与 2 负例合同正确，但 12 个社区网格代表/边界点中融合模板获选为 0，正式记为
`CAPABILITY_REJECTED_NO_EFFECTIVE_TEMPLATE`；decompose-BMM/MM/dynamic-addmm 均能命中并正确运行，
但三轮 OFF/ON 分别形成 `PERF_REGRESSED`，因此保留现有 NPU guard。T-078 仅用于下一批新
acceptance units。

## 2. 工作线吻合性

| 主题 | 8 月 29 日已有工作 | 8 月 31 日要求 | 校准决定 |
| --- | --- | --- | --- |
| 主线 | community-native pass/pattern → tests → NPU | community tests → acceptance unit → GPU/NPU | 方向吻合，补齐 GPU/reference 与 tracker 层 |
| 静态 inventory | 203 主候选 + 4 控制行 | registration 只能辅助查漏 | 保留 207 行，不称为 Pass 数量 |
| 验收单元 | 188 个、158 provisional | contract 才是基本单位 | 保留当前版本，但必须人工复核后冻结 |
| 测试事实源 | 已映射 community tests，仍夹杂源码扫描中心表述 | community tests 为主要事实源 | README/TODO/WORKFLOW 全部改为 test-first |
| 动态测试 | 首批 5 个拟做 NPU 最小迁移 | 先 GPU baseline，再 NPU comparison | T-076 GPU/NPU/comparison 已全部完成，正式闭环 5/5 |
| 双机 | 旧流程主要围绕 NPU | GPU 无 Agent、NPU 有 Agent | 明确固定脚本与 artifacts 交接 |
| 历史结果 | 保留 old Benchmark/isolated venv | historical/archived evidence | 归档，不删除、不升级 verdict |

## 3. T-074 复核结论

### 3.1 可以继续复用的部分

- 203 条 `inherited-upstream-needs-dynamic-validation` 与 4 条显式关闭控制项的集合边界清楚；
- 207 行拥有唯一 `candidate_id`，旧 `record_id` 重复也已显式保留；
- candidate CSV 已区分 implementation、registration、config、community tests、coverage、
  historical evidence 和动态状态；
- `direct-trigger-test`、`indirect-regression-test`、`no-test-found` 的定义适合作为 mapping
  review 信号；
- registry container 和 extension hook 已被识别为不直接进入 provisional denominator；
- 所有动态状态保持 `not-run-current-Pass`，没有把静态映射伪装成 NPU verdict。

### 3.2 仍需审核的部分

当前 acceptance-unit ID 主要按 `source stem + normalized name` 生成，只对少数已知 semantic group
做显式合并。这个规则能提供第一版去重，但不能自动证明一个组就是一个 upstream optimization
contract：

- 同名 registration 可能生成多个具有独立 expected behavior 的 pattern variants；
- 多个 source/registration 可能共同实现一个 contract；
- 一个 community regression test 可能只做间接覆盖，不能证明具体 transformation；
- 仅靠测试名/token overlap 不能冻结 direct contract；
- 自动聚合中有 15 个多 candidate 单元，9 个还同时包含多个 pass/pattern 名称，需要优先审阅。

因此 188 个单元和 158 个 `yes-provisional` 只能描述 T-074 v1 的 heuristic 输出，不能作为
项目规模、成功率或剩余 Pass 总数。

### 3.3 是否重新生成 T-074 v1

T-075 首批复核没有重新运行或覆盖 T-074：

1. registration candidate inventory 没有因 8 月 31 日术语校准失效；
2. 直接重跑同一 heuristic 只会得到相同 188/158，不能解决 contract 定义问题；
3. 已单独创建 acceptance-unit schema/manifest，并人工复核首批 community
   test/contract/variant；
4. 后续审核发现合并、拆分或字段缺口时，再生成 T-074 v2，并保留 v1 作为输入证据。

当前数量因此**没有变化**：207 candidate rows、188 provisional units、158 provisional eligible、
5 个首批冻结单元、20 个 variants、13 个 community test 引用、冻结 denominator 5、正式闭环
5/5。未来人工审核
可以改变 acceptance-unit 数量，这是预期的可审计修正，不是数据回归。

## 4. 已修正的概念问题

| 旧表述风险 | 问题 | 新口径 |
| --- | --- | --- |
| “每个 pass 都有……” | 暗示 inventory 行与独立任务一一对应 | 每个冻结 acceptance unit 承载完整证据 |
| “203 个上游候选/Pass” | 容易把 registration candidate 当 Pass 总数 | 203 是主候选行，只用于 inventory |
| “188 个去重 Pass” | heuristic grouping 尚未证明 contract | 188 个 provisional acceptance units |
| “先逐个写 NPU case” | 绕过 community contract 和 reference baseline | 先 mapping/manifest/GPU，再 NPU |
| “community test 是映射字段之一” | 弱化主要事实源地位 | community test + GPU baseline 定义 expected behavior |
| “compile 成功即支持” | 可能实际走 extern/fallback | 必须记录 runtime path 与完整链路 |
| “正确即可测性能并宣称收益” | 未隔离 pass 与 fallback | correctness 门禁后做 fresh-process OFF/ON paired |

历史报告中的原始统计和当时表述不机械改写；它们通过 archive/index 明确标为 previous phase。

## 5. 仓库结构调整

根目录从多份互相重复的总览/计划收束为：

```text
README.md       当前定位与导航
TODO.md         唯一活动任务清单
WORKFLOW.md     唯一活动执行流程
docs/           当前状态、范围、指南、变更控制、历史与归档
report/         不可改写的实验事实和 T-074 数据
```

旧 README、旧 current status、旧 audit overview、检查点、计划和交接全部移动到
`docs/archive/`，没有删除。`docs/HISTORY.md` 保留成果索引，`report/README.md` 提供证据导航。

## 6. T-075/T-076 首批完成项与剩余边界

已完成：

- `upstream/manifest.schema.json` 与首批 `manifest.yaml`；
- 5 条 candidate → registration → test → acceptance unit 多对多映射；
- 首批 5 个 contract/variant 人工决策；
- T-074 v1 → T-075 的证据角色修正；
- 不导入 `torch` 的静态一致性校验；
- 13 个 community tests 的 direct reference plan 与批量执行器；
- 20 个 variants 的完整 reference 处置：14 个进入动态 case，3 个 registration-only，3 个
  NPU-only gate；
- 环境/commit 严格门禁、单 case 隔离、失败继续、FX before/after、稳定 signature 和结构化 summary；
- GPU 静态校验、整批/单 case 执行、打包与回传说明。
- GPU 13/13 direct valid、文本 handoff 与逐 case FX/result/inventory 哈希复核；
- 首批 5 个 acceptance units 冻结进入 denominator。
- `REF-mm-plus-mm-native` 已落盘统一 NPU/comparison 结果，正式 verdict 为 `BEHAVIOR_UNCHANGED`；
- `AU-pad-mm-mm`、`AU-pad-mm-bmm`、`AU-pad-mm-addmm` 已形成单元级 comparison，均分类为 `EXPECTED_PRODUCT_DIVERGENCE`。
- `AU-post-grad-addmm` 已纠偏为 `EXPECTED_PRODUCT_DIVERGENCE`；P-018 default-enable/live-opt-out 候选已完成精确上游合同复验。
- T-077 的 5 个单元、11 个 direct cases 和 17 个 variants 已完成 GPU/NPU 对照；MM 回归进入
  known issues，修复候选已验证。
- T-076、T-077 性能处置均完成；T-077 为 measured=4、capability-assessed=1、pending=0。机器可读结果位于
  `results/current/T-076/performance_summary.json` 与 `results/current/T-077/performance_summary.json`。

T-078 已从上述集合审核 1 个 `no-test-found` 与 3 个 indirect 单元，建立 12 个 direct cases 和
20 个 variants；其余 47 个 `no-test-found` 和 26 个 indirect 单元继续待审。仍未完成 T-077 MM 候选修复的
产品代码评审/合入。P-018 是否并入正式产品属于候选变更评审，不再作为 T-076 未闭环回归。

## 7. 下一条 Codex 任务

```text
T-077 修复支线：复核候选 `dfbcc25b76743ea6c1c5cd61b6b30f0a910148a6`，经授权后推送/合入
torch_npu，并用同一六变体合同做安装态回归。

T-078 新批次：4 个单元的 manifest/reference plan 已完成；下一步由 GPU 执行 12 个原生 community
cases，20/20 variants 有效后冻结 denominator，再进入 NPU `triton_experimental` comparison、
条件 repair 与 performance。
```

## 8. 当前环境边界

- NPU 新测试从 `/home/z50063656/tmp` 发起；GPU T-076 从 `/data/z50063656/tmp` 发起；
- GPU pull 后使用 `scripts/run_gpu_reference_task.sh --task T-076|T-077|T-078 --gpu ID`，脚本自动进入
  工作目录、激活环境、校验、运行、导出文本并维护 `latest`，不再人工查找 timestamp；
- NPU 控制节点动态任务使用 `/home/z50063656/Pass/activate_pass.sh` 激活 Conda `Pass`；GPU
  reference 使用 `z00824525`/sudo、A100/R550、CUDA 12.6.3、pip venv Python 3.12 和与冻结
  PyTorch commit 一致的 `/data/z50063656/envs/PassGPURef`；compat 当前未启用；
- 不在 PyTorch/torch_npu 源码树中 import `torch`；
- installed torch_npu wheel 与 `dist` 同名 wheel 哈希冲突仍未解除，不重装；
- T-055～T-073 的 Benchmark/isolated venv 结果保留原环境标签；
- T-076 GPU runner 阶段只修改 tracker 文档、schema、执行计划和 runner；本机无
  `nvidia-smi`，当时只完成 `torch_imported=0/gpu_executed=0` 静态校验。
- E-205 NPU 执行新增的仍只是 tracker case-specific adapter 和文档；没有修改
  PyTorch、torch_npu、Triton、Conda 环境或 wheel。
