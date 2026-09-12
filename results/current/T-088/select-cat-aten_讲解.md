# T-088：连续 select + cat 转 view 功能与性能报告

> 更新时间：2026-09-11 15:25 CST（UTC+08:00）

## 1. Pattern 意图与代码

完整连续切片又拼回同一维度时，用 view 代替复制；不完整或不连续切片不改。

```python
# 文件：torch/_inductor/fx_passes/split_cat.py:1907
view_node = graph.call_function(torch.ops.aten.view.default, args=(node_input, cat_node.meta['val'].shape))
```

## 2. GPU / NPU 行为对照

GPU：原生四输出合同通过；只有完整连续分支目标命中。

NPU：四输出与数值保持，完整连续分支改成 view；部分分支继续 NPU cat extern。

NPU 后端为 `triton_experimental`。原生入口先记录跳过/不生成测试类，再做设备 harness 适配；没有放宽社区数值断言或覆盖冻结源码。

原例步骤、最小适配和失败记录见 [适配报告](../../../issues/REF-select-cat-aten-native/适配报告.md)。

## 3. 性能测例来源和测法

冻结社区中未找到精确切换本目标的原生 benchmark。本次复用社区功能图和原 shape/dtype 派生子图性能测例；不是社区原生性能测试，也不是完整模型端到端。

每臂新进程，OFF1/ON1/ON2/OFF2/OFF3/ON3，预热 10 次、每臂 100 次。每臂求 p50/p99，再取三臂中位数；编译首调用单列，不纳入稳态。记录同步 host 与 NPU Event 两套时钟、峰值 allocated/reserved。

训练单元计时包含前向+反向，不含 SGD；其他单元计时涵盖社区子图的一次调用。性能前已核验原合同、精确目标 FX、全图编译和设备 placement。cat/matmul_backward 的图内 NPU extern 与 CPU/图外 fallback 分开计数。

## 4. 本次结果

结论：`PERF_IMPROVED`；保留现有默认配置，不凭一次局部图收益全局开启。

| 时钟 | p50 改善 | p99 改善 |
| --- | ---: | ---: |
| host_ms | 10.14% | 11.76% |
| event_ms | 11.78% | 11.22% |

负数表示回退。若臂间 p50 波动比超过 1.5，按高方差 PERF_MIXED，不声称稳定收益。

原始样本、显存、编译时间及逐臂 FX/IR/output_code 索引见 [性能汇总](performance_summary.json)；[功能与门禁原件](functional/select-cat-aten.json)。

## 5. 修复/适配边界

这五个单元使用现有 NPU 产品实现即可运行；本轮没有把“测试 harness 适配”称为“原产品不支持、修复后支持”。
若某个 GPU/NPU kernel 不同，按后端 lowering 差异明确保留，不仅凭数值通过宣称全栈相同。
