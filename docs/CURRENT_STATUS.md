# 当前状态与 2026-08-31 工作线校准结论

> 更新时间：2026-09-11 18:44 CST（UTC+08:00）
> 校准输入：`831需求变更.md`、`831TODO_triton_experimental_pass_tracker.md`、
> `831WORKFLOW_triton_experimental_pass_tracker.md`。
> 当前阶段：T-076～T-086 原冻结范围共33个acceptance units有效；T-078 新增低精度
> regression 已在 NPU 修复，等待正确的 FP16 value=1 GPU 邻接 reference；T-084～T-086 的
> 5个单元已完成GPU、NPU功能与性能处置。T-076/T-077严格再认证单列。
> T-087～T-113保留38个跟踪ID，T-112去重后为37个独立单元；其中7个新reference已冻结，30个尚未冻结。
> 13批40个GPU cases已复核；7个新单元完成NPU功能、正式comparison和性能处置；
> T-081～T-113已无草案批次，所有未选候选均保留明确的合并/延期原因。

## 1. 总结

2026-09-11 已拉取 T-102 全部 11 片并复核；新包共 13 批、40 cases、36 个计划 ID、1226 份可恢复正文。
T-087 训练重排、T-088 两个单元、T-089/T-090 各一个单元已完成真实 NPU experimental 原合同、代码审查、性能图门禁和六臂计时：3改善、1中性、1混合。
T-087设备解析与T-096精度修复已获授权部署Pass，分别原例1/1+近邻4/4、原例3/3+功能对照7/7通过。前者无合法OFF免测；后者原七元素六臂性能PERF_REGRESSED，额外NPU数值域PARTIAL_ALIGNED。产品社区合入未执行。
当前40个GPU reference冻结、40个NPU/comparison、40项性能处置；[本轮完成/缺口/学习入口](../report/t087_t096_unblocked_completion_20260911.md)。
T-091 与 27 个 attention 合同需补精确目标归因；只读观察器已接入原生 GPU 入口，尚待 GPU 执行。
T-112 与 T-084 同合同，原文件保留，不重复计数。详见 [GPU复核](../report/gpu_incoming_review_20260911.md)、
[NPU步骤和学习报告](../report/t087_t090_npu_progress_20260911.md)。

2026-09-07 严格核验：24 个 GPU case 的关键正文已验哈希，19 个 case 的源码断言覆盖在声明
范围内通过；另有 2 项数值辅助 oracle 与 3 项输入梯度 oracle 缺口。24 份空 stdout 无须重传，
24 份非空 stderr 及全量 archive 仍待补；不能称原始日志已经重解析通过。本机已读取 19 份选定
NPU 原始结果，4 项性能汇总独立重算一致，但历史安装态/源码绑定和逐样本证据仍不足。
严格总状态 `pending=41、exempt=3`，表示分项总门禁而非 41 个文件缺失；原始 verdict 不改写。
详见 [本轮核验与最短补证路径](../report/t076_t077_history_reaudit_20260907.md)；
固定机器可读入口为 [最新审计](../results/audits/latest.json)。

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

T-078 原冻结范围的 4 个 post-grad 单元、12/12 原生 cases、20/20 variants 继续有效。
2026-09-08 覆盖复核新增 addcdiv FP16/BF16 两条 dtype-only GPU case，2/2 reference 有效；NPU
六进程三臂归因后 BF16 已通过位级/codegen 验证并以 `PERF_NEUTRAL` 保持启用。FP16 修复前的
三条 compiled 路径共同偏离 eager，根因为缺少除法、乘法后的 FP16 舍入；现已通过 NPU 专属
显式舍入 lowering 修复 `value!=1` 的原 FMA 图；`value=1` 已纠正为社区 bitwise 邻接，
预期不命中且 counter=0，不再注册 `add(div)` 重融合。此前 `value=1 counter=1` 的附加证据仅作
被纠正历史、不参与验收。2026-09-09 真机补测发现 FP16 `value=1` 的普通
`div -> add` compiled 丢失 quotient FP16 舍入，修复前有 1176/4096 mismatch、最大误差
0.015625。当前 NPU 后端修复在 div 与 add 之间恢复一次 FP16 round-trip，真机
bitwise 通过、mismatch=0、counter=0。该扩展状态为 `NPU fixed + GPU reference pending`。NPU
`triton_experimental` 上补齐 addcdiv FP32/BF16/FP16 pattern/lowering，并修复
partial amin→min 的 Triton Ascend NaN helper；两项均通过同合同回归。性能处置结果为 addcdiv
`PERF_NEUTRAL`、partial `PERF_MIXED`、addmm unfuse `PERF_REGRESSED`、baddbmm unfuse
`PERF_MIXED`。其中 addcdiv `PERF_NEUTRAL` 只适用于已有合法 OFF/ON 分母的 FP32/BF16；FP16
功能已修复，但修复前 OFF 不正确，故单列为无合法收益分母，不借用其他 dtype 结论。最终产品 gate 全局关闭 addmm unfuse，只关闭 baddbmm 的非默认 alpha/beta，默认
标量收益路径保留。详情见 [T-078 闭环报告](../report/t078_npu_completion_20260906.md)。

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

### 3.2 批次审核完成后仍保留的 heuristic 边界

当前 acceptance-unit ID 主要按 `source stem + normalized name` 生成，只对少数已知 semantic group
做显式合并。这个规则能提供第一版去重，但不能自动证明一个组就是一个 upstream optimization
contract：

- 同名 registration 可能生成多个具有独立 expected behavior 的 pattern variants；
- 多个 source/registration 可能共同实现一个 contract；
- 一个 community regression test 可能只做间接覆盖，不能证明具体 transformation；
- 仅靠测试名/token overlap 不能冻结 direct contract；
- 自动聚合中有 15 个多 candidate 单元，9 个还同时包含多个 pass/pattern 名称，需要优先审阅。

截至 2026-09-10，158 个 provisional eligible 已全部分配并完成逐批审核：71 个进入活动
manifest，87 个因重复、CPU/Fake/间接证据、显式关闭或缺少独立设备合同而保留明确延期理由。
旧表中的 48 个 `no-test-found` 和 29 个 indirect 是审核输入标签，不再表示仍有同数量的未审核任务。
30 条 registry/hook/容器类非计数记录也已按结构原因单列，不进入 GPU 分母；只有 upstream 新增
直接合同或 source/test mapping 漂移时才重开。

因此 188 个单元和 158 个 `yes-provisional` 仍只能描述 T-074 v1 的 heuristic 输出，不能作为
项目规模、成功率或剩余 Pass 总数；“审核完成”也不等于延期项已经获得设备支持。

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
- T-078 原冻结范围的 12/12 GPU 原生 cases、20/20 variants、4/4 NPU comparison 均有效；
  addcdiv BF16 与 FP16 value!=1 已验证，FP16 value=1 普通 div+add 已在 NPU 修复并位级通过，
  待 GPU 邻接 reference。addcdiv/partial 已验证范围的候选性能
  和最终 addmm/baddbmm 产品门禁均已落盘。正式数据位于
`results/current/T-078/` 与四个对应 acceptance-unit 目录。

T-079 已完成 4/4 units、4/4 社区方法和 14/14 variants 的 GPU/NPU 对照与性能处置。NPU 原生入口
均因 `HAS_GPU=false` 为 0 tests，最小适配保留原方法、数值和 counter/FileCheck 后全部 PASS。
cat-slice-cat、split→cat、cat→split 的 NPU Event p50 分别改善 18.41%、28.46%、13.03%，保留启用；
batch=1 bmm→mm 在社区 shape 和三个 sensitivity shape 的 Event p50 全部回退，已在
`triton_experimental` 增加 `disable_bmm_to_mm=true` 门禁，默认/显式重开双臂验证 PASS。正式数据位于
`results/current/T-079/` 与四个对应 acceptance-unit 目录。

T-080 已完成 3/3 units、13/13 GPU cases、13/13 NPU variants 的功能/结构处置。Constructor mover
功能一致且社区 length=32 的 Event p50/p99 变化为 +0.46%/-0.50%，判定 `PERF_NEUTRAL` 并保留启用。
Prepare-softmax 经测试态 generic guard 探针形成 online prim，但产品 `FALLBACK_LIST` 明确关闭 lowering，
因此为受控差异和 `PERF_EXEMPT`。Const-scatter 在修复两处完整图 codegen 缺口后，社区全尺寸
CrossEntropy backward 的 Event p50/p99 回退 10.50%/10.55%，allocated 增加 35.87%；已增加
NPU-only 可逆门禁并完成默认关闭/显式重开双臂验证。正式数据位于 `results/current/T-080/` 与三个
对应 acceptance-unit 目录。

T-078 已从上述集合审核并闭环 1 个 `no-test-found` 与 3 个 indirect 单元；T-080 又纠正 2 个
`no-test-found`。其余同类 coverage hint 已在 T-081～T-113 中完成选择或明确延期，不再列为
未审核队列。仍未完成 T-077 MM 候选修复的产品代码评审/合入。P-018 是否并入正式产品属于候选
变更评审，不再作为 T-076 未闭环回归。

## 7. 下一条 Codex 任务

2026-09-06：验收校验加固和零设备回归已落地，见
[修复记录](../report/tracker_validation_hardening_20260906.md)。完整后续草案见
[TASK_BACKLOG](TASK_BACKLOG.md)：125 个剩余 provisional eligible 单元暂列 33 批，其中 T-084～T-086
已审核并准备5个单元；
30 条非计数结构记录单列。该数量只覆盖旧 FX inventory，不含尚未独立建表的 lowering/template。

```text
T-077 修复支线：复核候选 `dfbcc25b76743ea6c1c5cd61b6b30f0a910148a6`，经授权后推送/合入
torch_npu，并用同一六变体合同做安装态回归。

T-078：原12/12 GPU reference、4/4 NPU comparison、条件 repair、候选性能和最终产品 gate有效；
FP16 value=1 普通 div+add 已修复并保持 addcdiv FMA counter=0；待 GPU 邻接 reference 回传。

T-079、T-080 均已正式闭环。T-080 保留 const-scatter CrossEntropy 的社区完整 benchmark；
prepare-softmax 因产品显式 lowering fallback 免测；constructor mover 从社区功能正例派生性能图。
T-081～T-083已完成2/2/3个社区合同的GPU运行：11/11 cases、24/24 variants有效，7个单元已冻结；
随后在`triton_experimental`完成7/7数值、命中和实际改图。T-083没有沿用world_size=1结论，另用真实
双rank HCCL验证3→1/2→1 collective。7个单元均完成全局互斥下的六臂性能处置。
CPU-only/fake-PG/间接覆盖或缺少直接测例的剩余候选按原批次保留 deferred。
活动矩阵共71个跟踪ID、70个独立单元：T-076～T-086的33个冻结单元均已有GPU、NPU/comparison与性能处置；
新单元中7个已功能/性能处置闭环（含两项Pass已部署修复）、28个待GPU精确归因、2个待GPU收件；T-112另保留为别名行，
不计新增分母。T-078另有1个FP16 value=1扩展
variant仍等待正确GPU邻接reference。

T-091～T-100本轮审计37个候选，只将4个可独立归属且有真实CUDA入口的合同放入活动矩阵：
stack axis规范化、A100 E8M0 log2-ceil边界替换、Conv-BN eval重参数化和Linear binary folding。
T-100属于旧inventory漏测例后的纠偏；T-098把同一个社区乘积中的inlined/decomposed合并计一次。
T-096的NPU路径原是generic CUDA guard遗漏导致的capability-pending，不是产品明确disable；已经先留存原生阻断，完成最小适配、NPU注册/数学修复评审和安装态回归。六个零ready批次只保留延期审计，不创建空run。

T-101～T-113完成52个候选的最终准备审核。T-102～T-107按编号保留27个真实CUDA SDPA
pattern合同；25～27的`disable_cuda=True`和XPU-only测试入口作为显式关闭证据，不绕过。
T-112原生CUDA `test_dedup_reduce_scatter` 已通过，但其`world_size=1`只证明
生成代码2个RS降为1个；与T-084同合同，独立NPU/性能入口停止重复执行，已有worker仅保留历史。
T-101与已闭环T-085合并；T-108～T-111为CPU量化/MKLDNN；T-113为具体fusion调度容器。
准备时保留28个ready ID、24个合并/延期，零草案；现按27个独立SDPA合同和1个别名补证计数。

T-084～T-086分别闭环1/3/1个单元。T-084真实2-rank HCCL为`PERF_IMPROVED`；T-085的
pointless-cumsum与partitioned-scatter回退，overlap因轮间高波动为`PERF_MIXED`；T-086
重入原位功能一致但尾延迟混合。pointless-cumsum已增加NPU专属默认关闭gate并真机确认。
完整代码框、调用链、适配和FX/IR/generated code见
[NPU闭环报告](../report/t084_t086_npu_function_performance_and_fix_20260910.md)。

T-078新上传的`BF16-value=1-text-handoff`实际只包含
`REF-addcdiv-fma-codegen-native`，证明CUDA codegen含`div_rn`和`tl.fma`，但没有执行
`REF-addcdiv-fma-fp16-value1-derived`。因此误命名包作为有效codegen邻接补证保留，FP16 value=1
GPU dtype/value缺口不关闭。
T-084～T-086 的功能和性能讲解、worker及实际设备结果均已进入当前71行矩阵；T-087～T-113
按本节的功能阶段/归因待补/待收件/重复合同分别展示；上述T-078
扩展variant仍独立等待GPU，不借用其他codegen邻接证据关闭。

T-076/T-077 的历史结论保留；严格再认证不能直接免除补证或同合同重验。新 1.3 review
分别可恢复 80/68 份关键正文，用于逐项 FX 学习与审计，但不是完整历史 archive。
仍须区分原始日志、数值/梯度断言覆盖、性能运行溯源是否齐全；缺项以独立审计记录为准，
不得用当前源码元数据补造历史运行事实。
```

## 8. 当前环境边界

- NPU 新测试从 `/home/z50063656/tmp` 发起；GPU T-076 从 `/data/z50063656/tmp` 发起；
- GPU pull 后使用 `scripts/run_gpu_reference_task.sh --task T-087 --gpu ID`（支持 T-076～T-113；
  T-085 使用 `--gpus ID1,ID2`），脚本自动进入
  工作目录、激活环境、校验、运行、导出 1.3 review handoff 与备用网页分片并维护 `latest`，
  不再人工查找 timestamp；默认 handoff 是单行 JSON，超过 96 KiB 时自动生成分片并打印应上传的
  manifest 路径；评审包恢复摘要/FX/关键 case 正文，其余 artifacts 保留 SHA256；
- NPU 控制节点动态任务使用 `/home/z50063656/Pass/activate_pass.sh` 激活 Conda `Pass`；GPU
  reference 使用 `z00824525`/sudo、A100/R550、CUDA 12.6.3、pip venv Python 3.12 和与冻结
  PyTorch commit 一致的 `/data/z50063656/envs/PassGPURef`；compat 当前未启用；
- NPU 原生入口、最小适配、命中/生效分层、根因定位、最小修复和回归的连续教学流程见
  `docs/NPU_FUNCTION_REPAIR_WORKFLOW.md`；T-079/T-080 均已有正式 runner、结果和产品处置；
- 不在 PyTorch/torch_npu 源码树中 import `torch`；
- installed torch_npu wheel 与 `dist` 同名 wheel 哈希冲突仍未解除，不重装；
- T-055～T-073 的 Benchmark/isolated venv 结果保留原环境标签；
- T-076 GPU runner 阶段只修改 tracker 文档、schema、执行计划和 runner；本机无
  `nvidia-smi`，当时只完成 `torch_imported=0/gpu_executed=0` 静态校验。
- E-205 NPU 执行新增的仍只是 tracker case-specific adapter 和文档；没有修改
  PyTorch、torch_npu、Triton、Conda 环境或 wheel。
