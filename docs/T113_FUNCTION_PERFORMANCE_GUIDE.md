# T-113 功能与性能测例讲解

> 更新时间：2026-09-10 22:55:21 CST（UTC+08:00）
> 状态：0 个GPU-ready单元，1 个明确延期。

NPU功能、修复验证和性能统一使用 `triton_experimental`，并在导入`torch`/`torch_npu`前选后端；OFF/ON每臂使用新进程。

## 功能测例

```python
# torch/_inductor/fx_passes/group_batch_fusion.py:1679
def group_batch_fusion_passes(graph, pre_grad=True, fusion_options=None):
    fusions = generate_fusion_from_config(...)  # 调度具体fusion
    for rule in fusions:
        rule.apply(graph)
```

这是调度容器，不是一个独立数学改写。映射测试实际归属于batch-linear、cat-linear或纯subset选择helper；具体fusion应各自形成合同，容器不能再计一个功能/性能分母。

## 性能测例

本批没有通过真实GPU目标合同的独立单元，因此不派生OFF/ON worker、不执行设备性能，也不制造NPU ON路径。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-group-batch-fusion` | `deferred-structural-dispatcher-concrete-fusions-counted-separately` | group_batch_fusion_passes只按配置实例化并调度具体fusion；映射测例分别属于batch-linear、cat-linear或纯subset helper，容器不独立计功能/性能分母。 |
