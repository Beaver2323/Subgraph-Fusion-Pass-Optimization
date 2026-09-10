# T-101 功能与性能测例讲解

> 更新时间：2026-09-10 22:55:21 CST（UTC+08:00）
> 状态：0 个GPU-ready单元，1 个明确延期。

NPU功能、修复验证和性能统一使用 `triton_experimental`，并在导入`torch`/`torch_npu`前选后端；OFF/ON每臂使用新进程。

## 功能测例

```python
# torch/_inductor/fx_passes/reduced_atomic_contention.py
def replacement(index, index_size, src, dim_size):
    # partitioned scatter optimization；真实合同已归入T-085
    ...
```

旧索引把同名内层`replacement`错误关联到四条pattern-matcher注册机制测试。当前实现、GPU/NPU证据和性能处置均已由T-085覆盖，因此这里合并旧ID，不重复计数。

## 性能测例

本批没有通过真实GPU目标合同的独立单元，因此不派生OFF/ON worker、不执行设备性能，也不制造NPU ON路径。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-reduced-atomic-contention-replacement` | `merged-existing-coverage-t085-stale-symbol-mapping` | 当前源码的replacement属于partitioned scatter优化，已由T-085完成原生GPU、NPU功能和性能闭环；旧索引四条pattern-matcher基础设施测试不执行该优化。 |
