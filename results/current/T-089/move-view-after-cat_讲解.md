# T-089：将 view 移到 cat 之后 功能与性能报告

> 更新时间：2026-09-11 15:25 CST（UTC+08:00）

## 1. Pattern 意图与代码

把多个切片 view 的拼接改为先拼接再 view，减少重复视图处理；保留 clone 多用户关系。

```python
# 文件：torch/_inductor/fx_passes/split_cat.py:2935
# 语义示意：cat([view(p0), ..., view(p6)]) -> view(cat([p0, ..., p6]))
```

## 2. GPU / NPU 行为对照

GPU：原 [7,8,96]、clone 与嵌套 cat 的社区合同通过。

NPU：精确 handler 改图且原数值通过。实际 OFF 为 3 次 NPU cat extern；ON 为 1 个 Triton 切片/复制核加 2 次 NPU cat extern，并非所有 cat 都被消除。

NPU 后端为 `triton_experimental`。原生入口先记录跳过/不生成测试类，再做设备 harness 适配；没有放宽社区数值断言或覆盖冻结源码。

原例步骤、最小适配和失败记录见 [适配报告](../../../issues/REF-move-view-after-cat-aten-native/适配报告.md)。

## 3. 性能测例来源和测法

冻结社区中未找到精确切换本目标的原生 benchmark。本次复用社区功能图和原 shape/dtype 派生子图性能测例；不是社区原生性能测试，也不是完整模型端到端。

每臂新进程，OFF1/ON1/ON2/OFF2/OFF3/ON3，预热 10 次、每臂 100 次。每臂求 p50/p99，再取三臂中位数；编译首调用单列，不纳入稳态。记录同步 host 与 NPU Event 两套时钟、峰值 allocated/reserved。

训练单元计时包含前向+反向，不含 SGD；其他单元计时涵盖社区子图的一次调用。性能前已核验原合同、精确目标 FX、全图编译和设备 placement。cat/matmul_backward 的图内 NPU extern 与 CPU/图外 fallback 分开计数。

## 4. 本次结果

结论：`PERF_IMPROVED`；保留现有默认配置，不凭一次局部图收益全局开启。

| 时钟 | p50 改善 | p99 改善 |
| --- | ---: | ---: |
| host_ms | 7.23% | 4.19% |
| event_ms | 6.48% | 9.45% |

负数表示回退。若臂间 p50 波动比超过 1.5，按高方差 PERF_MIXED，不声称稳定收益。

原始样本、显存、编译时间及逐臂 FX/IR/output_code 索引见 [性能汇总](performance_summary.json)；[功能与门禁原件](functional/move-view-after-cat.json)。

## 5. 修复/适配边界

这五个单元使用现有 NPU 产品实现即可运行；本轮没有把“测试 harness 适配”称为“原产品不支持、修复后支持”。
若某个 GPU/NPU kernel 不同，按后端 lowering 差异明确保留，不仅凭数值通过宣称全栈相同。
