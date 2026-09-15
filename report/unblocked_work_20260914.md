# 2026-09-14 不依赖下一轮 GPU 的工作

> 更新时间：2026-09-15 01:30 CST（UTC+08:00）；继续执行中，不把阶段进度当全量完成。

最新增量见[9月15日剩余任务](remaining_work_20260915.md)：17已证实为15的推理注册别名，无需独立GPU补测；
71跟踪ID、69独立单元、已完成仍47。16/29补充执行器准备完成待GPU；22设备边界串行排队。
本文后续70/23等计数是上一轮时点记录，不再作为当前去重分母。

## 1. 新收件

已快进到 `2c95aa3`，保留所有本地未提交修改。T-098 10片、T-100 2片；还原校验分别通过2250和890份评审正文。
两批原生各1个方法、零skip，冻结源码clean；T-098原run为 `reference-20260911T100136+0800-7mip1lr_`，
T-100为 `reference-20260911T101110+0800-txchnvg2`。这是旧run重压缩回传，不是9月14日重测。

随后再次快进到 `16ecd84`，收到T-076单包历史日志、T-077三片历史日志和T-078正确FP16 value=1单例。
24份历史stderr已全部按旧inventory校验、重解析成功；T-078单例已与原NPU修复对照并关闭邻接缺口。
历史T-076/T-077完整再认证仍pending，不重复要求已收到的日志。

## 2. 当前工作范围

- 新收到的 Conv-BN 与 Linear binary-folding 原社区 NPU 功能和后续合法性能。
- 已确认精确目标的24个 attention 单元；16/17/29留在映射审核，不用邻接编号替代。
- T-091 独立 OFF/ON 图及性能。
- 新结果、适配/问题报告和完整 FX/IR/output_code 归档。

## 3. 执行记录

T-102 pattern 1原生入口为 `_FailedTest` / 缺少CUDA专用测试类，不等于pattern不支持。
先保留原件，再最小设备适配。新适配器保持原partialmethod、训练/推理、dtype和数值断言；
CUDA生成代码符号断言映射为AST验证的实际NPU attention调用，明确计为断言适配。

| 单元 | 实际完成 | 尚未完成 |
|---|---|---|
| T-091 stack normalization | 原社区功能、精确OFF/ON、六臂性能、正式验收；PERF_MIXED | 无本合同执行待办；不外推模型收益 |
| T-100 Linear binary folding | 原方法176次数值比较、160正/16负、六臂性能；PERF_IMPROVED | 默认开关扩展属独立产品评审，本轮未改 |
| T-098 Conv-BN | 原默认失败已归档；四进程/三种子定位HF32与折叠舍入 | 对齐TF32-OFF意图的完整原方法复验；默认模式仍不能写已修复 |
| T-102 pattern1 | 安装态训练匹配失败；隔离注册候选原方法20次比较、8次精确改写通过 | 邻接/负例及注册入口评审、安装态授权与回归；未签性能 |
| T-104 pattern13 | 原方法、独立OFF/ON、六臂性能；PERF_REGRESSED | 无本合同执行待办；未改默认配置 |
| T-105 pattern18、T-106 pattern23/24 | 原方法、独立OFF/ON、六臂性能及逐图复核完成；两改善一回退 | 仅24数学路径保留PARTIAL_ALIGNED；不外推其他域或改默认配置 |
| T-105 pattern19 | FP32原合同、独立数值OFF/ON、六臂计时；PERF_REGRESSED | half+FP32 mask编译失败及原例无数值oracle明确保留，PARTIAL_ALIGNED |
| 其他18个已归因attention | 3原例通过、14训练合同未通过、22号安装态数值失败；每例有原件和报告 | 15标量适配后OFF NaN、20OFF数值、21邻接接替分别阻断；22号隔离候选原例和三个邻接通过，未部署 |
| 16/17/29 | 已完成已有输入为何命中其他编号的源码复核 | 精确本编号GPU合同；不盲重跑相同输入 |

T-091 host p50/p99改善-1.23%/+9.26%，Event -3.48%/-14.42%，两时钟尾部相反。
T-100 host p50/p99改善10.36%/3.21%，Event 13.62%/11.88%。均为社区功能图派生子图，非完整模型。
T-104 pattern13 host改善-6.77%/-7.55%，Event -5.61%/-4.15%，两时钟均回退。
18号host改善26.19%/8.67%、Event31.75%/26.55%；23号host29.02%/27.48%、Event36.50%/34.49%；
24号host-52.34%/-48.75%、Event-66.76%/-64.39%。19号FP32数学路径host时延增加38.38%/45.00%、
Event增加49.33%/51.64%。当前正式reference/comparison/性能处置各47，独立计划总数70。

学习入口：

- [T-091逐层讲解](../results/current/T-091/stack-normalization_讲解.md)
- [T-100逐层讲解](../results/current/T-100/linear-binary-folding_讲解.md)
- [T-104 pattern13功能/性能输入和真实生成代码讲解](../results/current/T-104/pattern-13_讲解.md)
- [T-105 pattern18讲解](../results/current/T-105/pattern-18_讲解.md)、[T-106 pattern23讲解](../results/current/T-106/pattern-23_讲解.md)、[pattern24讲解](../results/current/T-106/pattern-24_讲解.md)
- [T-105 pattern19：FP32计时与half缺口](../results/current/T-105/pattern-19_讲解.md)
- [T-106 pattern22：修复候选代码、完整原例和邻接验证](../issues/REF-sfdp-pattern-22-native/修复候选验证报告.md)
- [T-104 pattern15：入口修正后OFF真实数值失败](../issues/REF-sfdp-pattern-15-native/性能OFF数值失败.md)
- [T-098精度根因及必要调用栈](../issues/REF-efficient-conv-bn-eval-product-native/根因分析.md)
- [attention训练注册根因](../issues/REF-sfdp-pattern-1-native/根因分析.md)及[隔离验证](../issues/REF-sfdp-pattern-1-native/修复验证报告.md)
- [16/17/29映射解释](gpu_target_mapping_16_17_29_20260914.md)

## 4. 性能边界

旧attention准备worker的OFF关闭整轮joint_graph，不能据此签单pattern收益；当前禁止使用它直接benchmark。
准备worker已改为仅去掉本编号注册，并初始化原registration的empty样例输入；尚未获功能门禁的编号仍禁止直接benchmark。
T-091已按目标级OFF实测；T-100仅切换Linear folding配置并核验folded_op。没有把明确产品关闭路径强行启用。

补充复核发现inference注册会把dropout置零，因此3/4/6/7/9/12/28改为极低非零dropout的训练前向准备图，
尚待实际目标/数值门禁。详见[性能执行阶段与社区设计依据](attention_performance_contract_review_20260914.md)。

T-091/T-100的仓库离线复核入口已经可用，不读取原机器tmp：

```bash
python scripts/review_t091_t100_completion.py --check-current
```

该命令只重算和验哈希，`device_execution=false`描述复核器自身不启动设备，不否定归档中实际NPU执行。

## 5. 仍依赖 GPU 的内容

T-078 FP16 value=1邻接和T-076/T-077历史日志已经补齐；仍需GPU的是16/17/29的精确本编号reference、
T-076/T-077剩余archive/增强oracle（后者要按新run记录，不能伪造历史断言）。NPU/性能历史源码溯源缺口不是GPU重跑能修补的。
没有自动修改产品安装态，没有推送或社区合入。

24个attention合计：8原例通过、15训练合同未通过（包含pattern 1安装态）、1个数值失败。
通过的是13/15/18/19/20/21/23/24；19/20原例无数值oracle；18容器比较记录已复验为12个Tensor叶子。
14修正符号断言后推理数值通过，但训练未命中；22则在第一组输出出现真实大幅超差。
性能预检直接compile内部search_fn触发的Dynamo MOD_SKIPLIST已局部用社区dont_skip_tracing包装解决：
保留fullgraph、相同运算及精确OFF/ON断言；13/18/19/23/24已完成六臂计时并归档复核。最初8项失败已独立归档，不算性能结果。
22号已通过修改4处生成代码尺寸的三种子对照确认广播错误；通用codegen隔离候选通过完整原例及21/23/24三个邻接，
共21个Tensor比较、9次精确改写。独立设备边界与部署评审仍待完成，未改安装态。
详见[22号数值失败与调用栈](../issues/REF-sfdp-pattern-22-native/数值失败分析.md)。
校验器统计也已纠正：当前`fully_covered_units=47`只计登记原合同已闭环记录；矩阵其余23个独立原合同显示`base-contract-not-yet-closed`，T-112显示不计数别名，不冒充设备完成。额外域限制仍以PARTIAL_ALIGNED明确保留。

新增学习入口：[FP16 value=1 GPU/NPU真实生成代码对照](../issues/REF-addcdiv-fma-fp16-value1-derived/GPU补证与NPU修复对照.md)。

T-098此前一小时限额的HF32-off轮次主动中断；35组合/70次数值比较的部分记录保留，不生成PASS/性能门禁。
2026-09-15已在19号六臂结束后启动12600秒限额的完整112组合复验，使用同一项目设备锁；
本轮原件为`/home/z50063656/tmp/t098-native-22v6s_le/adapter/`，进度文件不是最终结论。
