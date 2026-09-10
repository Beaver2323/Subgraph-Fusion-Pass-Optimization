# 后续批次与覆盖边界

> 更新时间：2026-09-10T23:48:08+08:00

机器清单见 `upstream/task_backlog.json`；本表由 `scripts/build_task_backlog.py` 生成。

T-074 的 188 个 provisional 单元中，活动 manifest 已接入 71 个；
未接入的 87 个候选中，87 个留在已审批次延期，12 个批次已确认零GPU-ready，其余保留 0 个草案批次；另有 30 条非计数结构记录已按原因单列，不进入GPU分母，只有源码或映射变化时才重开。

T-081～T-086 的已选12单元已完成原生GPU reference、NPU triton_experimental功能验证与性能处置；T-087～T-100有10个单元等待GPU；T-101～T-113从52个候选中准备28个GPU原生合同、明确延期24个，其中6个批次合法零GPU-ready。至此T-081～T-113全部完成准备/延期审核，保留原批次和旧ID，不重排。

| 草案任务 | 源码 family | 暂列单元数 | 状态 |
| --- | --- | ---: | --- |
| T-081 | `joint_graph` | 5 | 已闭环2，延期3 |
| T-082 | `joint_graph` | 4 | 已闭环2，延期2 |
| T-083 | `post_grad` | 5 | 已闭环3，延期2 |
| T-084 | `post_grad` | 5 | 已闭环1，延期4 |
| T-085 | `post_grad` | 5 | 已闭环3，延期2 |
| T-086 | `post_grad` | 5 | 已闭环1，延期4 |
| T-087 | `post_grad` | 4 | 已准备2，延期2；等GPU |
| T-088 | `split_cat` | 5 | 已准备2，延期3；等GPU |
| T-089 | `split_cat` | 5 | 已准备1，延期4；等GPU |
| T-090 | `split_cat` | 5 | 已准备1，延期4；等GPU |
| T-091 | `split_cat` | 5 | 已准备1，延期4；等GPU |
| T-092 | `split_cat` | 5 | 已审核0个GPU-ready，延期5；禁止空跑 |
| T-093 | `split_cat` | 3 | 已审核0个GPU-ready，延期3；禁止空跑 |
| T-094 | `pre_grad` | 5 | 已审核0个GPU-ready，延期5；禁止空跑 |
| T-095 | `pre_grad` | 1 | 已审核0个GPU-ready，延期1；禁止空跑 |
| T-096 | `misc_patterns` | 5 | 已准备1，延期4；等GPU |
| T-097 | `replace_random` | 4 | 已审核0个GPU-ready，延期4；禁止空跑 |
| T-098 | `efficient_conv_bn_eval` | 3 | 已准备1，延期2；等GPU |
| T-099 | `freezing_patterns` | 5 | 已审核0个GPU-ready，延期5；禁止空跑 |
| T-100 | `binary_folding` | 1 | 已准备1，延期0；等GPU |
| T-101 | `reduced_atomic_contention` | 1 | 已审核0个GPU-ready，延期1；禁止空跑 |
| T-102 | `fuse_attention` | 5 | 已准备5，延期0；等GPU |
| T-103 | `fuse_attention` | 5 | 已准备5，延期0；等GPU |
| T-104 | `fuse_attention` | 5 | 已准备5，延期0；等GPU |
| T-105 | `fuse_attention` | 5 | 已准备5，延期0；等GPU |
| T-106 | `fuse_attention` | 5 | 已准备4，延期1；等GPU |
| T-107 | `fuse_attention` | 5 | 已准备3，延期2；等GPU |
| T-108 | `quantization` | 5 | 已审核0个GPU-ready，延期5；禁止空跑 |
| T-109 | `quantization` | 4 | 已审核0个GPU-ready，延期4；禁止空跑 |
| T-110 | `mkldnn_fusion` | 5 | 已审核0个GPU-ready，延期5；禁止空跑 |
| T-111 | `mkldnn_fusion` | 5 | 已审核0个GPU-ready，延期5；禁止空跑 |
| T-112 | `fsdp` | 1 | 已准备1，延期0；等GPU |
| T-113 | `group_batch_fusion` | 1 | 已审核0个GPU-ready，延期1；禁止空跑 |

## 每批准备与验收标准

- 逐单元审核 contract、正负例和真实测试入口；允许合并/拆分，但保留旧 ID 映射。
- 性能先检索社区 benchmark；没有则记录功能例派生理由和精确输入/输出/梯度合同。
- 准备 manifest、reference plan、功能/性能讲解、目标级 OFF/ON worker 与零设备回归。
- 先 GPU 原生 reference；NPU triton_experimental 原生阻断后才评审最小适配。
- 显式产品关闭只保留关闭证据/已有测量并免测；generic guard 进入能力评估，不能伪造 ON。
- 功能/命中/正确性通过再做独立进程 OFF/ON 性能；修复及性能归属本批 T。

## 当前覆盖边界

| 注册/执行层 | 当前证据 | 后续要求 |
| --- | --- | --- |
| FX register_graph_pattern / register_replacement | T-074 全部候选来自 fx_passes；部分已人工复核 | 按优化合同审核，不能按装饰器数量计数 |
| PatternMatcherPass / pass_dict | 部分容器/调度项在 inventory | 关联具体优化，结构记录不独立冒充性能单元 |
| register_lowering / ATen→IR | 已有 MM 回归等下游链路证据 | 独立注册清单尚缺；不可声称完整覆盖 |
| template / choice / autotune | 已有 B2B/GEMM 局部选模证据 | 独立候选注册清单尚缺；需与已有 FX 合同去重 |

`inductor-extension` 是旧 inventory 的分类标签，不证明 scheduler/codegen/lowering 全量覆盖。
后两层先补 inventory 与去重关系，再确认新增批次编号；当前草案数量不是全项目最终 T 数量。
