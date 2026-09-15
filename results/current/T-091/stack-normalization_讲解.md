# T-091：stack 规范化的功能与性能对照

> 更新时间：2026-09-14 21:07 CST（UTC+08:00）。实际后端：`triton_experimental`；已完成本合同功能/性能处置。

## Pattern 意图与代码

```python
# PyTorch/torch/_inductor/fx_passes/split_cat.py:383，语义示意，具体分支见冻结源码
# normalize_stack_default 读取 stack 的输入和 dim，检查并重建规范调用节点。
torch.stack([x, y], axis=1)  # 社区输入；捕获图时 axis 已转为 dim
torch.stack([x, y], dim=1)   # handler边界图两侧均为dim，不能把关键字转换归功于本handler
# test/inductor/test_split_cat_fx_passes.py::test_stack_normalization_axis_kwarg
# 两个 [4,4] FP32 输入；保留原数值比较。
```

GPU 原方法 1/1、零 skip，通过只读观察器确认 `normalize_stack_default` 实际重建 FX。
NPU 原入口因 GPU 测试条件无法执行，先保存阻断，再最小设备适配；原方法和数值断言均通过。
具体适配见[issue](../../../issues/REF-stack-axis-normalization-native/适配报告.md)。

```text
原社区方法 → torch.compile → pre_grad.normalization_pass
 → normalize_stack_default → 后续stack分解/布局处理 → NPU cat + view → 原输出断言
```

## 合法 OFF/ON 与实际设备行为

两臂都启用 `normalization_pass`，OFF 只移除目标注册，保留其余 11 条；ON 精确 handler=1、图变化=1，OFF=0。
后端在导入前选择，功能两臂及性能六臂各用新进程。`fullgraph=True`，无 CPU 转移；
这里的 NPU `aten.cat` 是图内设备调用，不是 CPU fallback，也不是新生成的 Triton 融合核。

```python
# issues/REF-stack-axis-normalization-native/evidence/benchmark-20260914T210110/
# stack-normalization/{off1,on1,...}/debug/torch_compile_debug/run_*/torchinductor/model*/output_code.py
buf0 = torch.ops.aten.cat.default([arg0_1, arg1_1], 1)
# 后续 reinterpret/view 得到 [4,2,4]；两臂最终算子相同。
```

## 性能结果与边界

复用社区功能图派生微基准；没有把它称为社区原生 benchmark 或完整模型端到端。
三轮顺序 OFF1/ON1/ON2/OFF2/OFF3/ON3，每臂预热10、采样100；编译耗时另记。

| 时钟 | 指标 | OFF ms | ON ms | 改善率 |
|---|---|---:|---:|---:|
| 同步 host | p50 | 0.38063 | 0.38530 | -1.23% |
| 同步 host | p99 | 0.56998 | 0.51718 | +9.26% |
| NPU Event | p50 | 0.26423 | 0.27342 | -3.48% |
| NPU Event | p99 | 0.29169 | 0.33376 | -14.42% |

结论 `PERF_MIXED`：中位延迟近似中性，尾部方向冲突，不能宣称稳定收益。
本例目标确实生效，但更晚阶段生成相同算子，因此不应仅凭一次尾延迟改善建议修改默认开关。

[功能复核](functional/stack-normalization.json)、[原始采样与性能汇总](performance_summary.json)、
[六臂完整 FX/IR/codegen 清单](../../../issues/REF-stack-axis-normalization-native/evidence/benchmark-20260914T210110/inventory.json)。
