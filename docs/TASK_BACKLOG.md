# 后续批次与覆盖边界

> 更新时间：2026-09-06T02:15:00+08:00

机器清单见 `upstream/task_backlog.json`；本表由 `scripts/build_task_backlog.py` 生成。

T-074 的 188 个 provisional 单元中，活动 manifest 已接入 21 个；
剩余 137 个 provisional eligible 单元暂分 33 批，另有 30 条非计数结构记录待审。

这些 T 是待审核草案，**不是 GPU-ready**，不进入冻结分母。仅修正 constructor mover 的 cuda→gpu 名称映射，不重写 T-074 原始证据。

| 草案任务 | 源码 family | 暂列单元数 | 状态 |
| --- | --- | ---: | --- |
| T-081 | `joint_graph` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-082 | `joint_graph` | 4 | 功能映射、性能来源与 worker 待准备 |
| T-083 | `post_grad` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-084 | `post_grad` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-085 | `post_grad` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-086 | `post_grad` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-087 | `post_grad` | 4 | 功能映射、性能来源与 worker 待准备 |
| T-088 | `split_cat` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-089 | `split_cat` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-090 | `split_cat` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-091 | `split_cat` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-092 | `split_cat` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-093 | `split_cat` | 3 | 功能映射、性能来源与 worker 待准备 |
| T-094 | `pre_grad` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-095 | `pre_grad` | 1 | 功能映射、性能来源与 worker 待准备 |
| T-096 | `misc_patterns` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-097 | `replace_random` | 4 | 功能映射、性能来源与 worker 待准备 |
| T-098 | `efficient_conv_bn_eval` | 3 | 功能映射、性能来源与 worker 待准备 |
| T-099 | `freezing_patterns` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-100 | `binary_folding` | 1 | 功能映射、性能来源与 worker 待准备 |
| T-101 | `reduced_atomic_contention` | 1 | 功能映射、性能来源与 worker 待准备 |
| T-102 | `fuse_attention` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-103 | `fuse_attention` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-104 | `fuse_attention` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-105 | `fuse_attention` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-106 | `fuse_attention` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-107 | `fuse_attention` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-108 | `quantization` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-109 | `quantization` | 4 | 功能映射、性能来源与 worker 待准备 |
| T-110 | `mkldnn_fusion` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-111 | `mkldnn_fusion` | 5 | 功能映射、性能来源与 worker 待准备 |
| T-112 | `fsdp` | 1 | 功能映射、性能来源与 worker 待准备 |
| T-113 | `group_batch_fusion` | 1 | 功能映射、性能来源与 worker 待准备 |

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
