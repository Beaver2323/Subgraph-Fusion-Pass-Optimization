# T-111 功能与性能测例讲解

> 更新时间：2026-09-10 22:55:21 CST（UTC+08:00）
> 状态：0 个GPU-ready单元，5 个明确延期。

NPU功能、修复验证和性能统一使用 `triton_experimental`，并在导入`torch`/`torch_npu`前选后端；OFF/ON每臂使用新进程。

## 功能测例

```python
# torch/_inductor/fx_passes/mkldnn_fusion.py:41
if torch._C._has_mkldnn:
    mkldnn = torch.ops.mkldnn
# packed convolution/linear/rnn与unary/binary lowerings
```

MKLDNN fusion面向CPU（部分XPU）packed算子。旧索引中的CUDA convolution测例走通用卷积/select_algorithm，不会调用这里的MKLDNN fusion，故不能作为GPU reference。

## 性能测例

本批没有通过真实GPU目标合同的独立单元，因此不派生OFF/ON worker、不执行设备性能，也不制造NPU ON路径。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-mkldnn-fusion-register-binary-unary-maybe-inplace-fusion-lowering` | `deferred-mkldnn-cpu-only-or-mismapped` | 源码受MKLDNN可用性与CPU/XPU packed op约束；列出的CUDA卷积测试走通用select_algorithm，不执行MKLDNN fusion，其他测例为CPU。 |
| `AU-mkldnn-fusion-register-hardtanh-fusion-lowering` | `deferred-mkldnn-cpu-only-or-mismapped` | 源码受MKLDNN可用性与CPU/XPU packed op约束；列出的CUDA卷积测试走通用select_algorithm，不执行MKLDNN fusion，其他测例为CPU。 |
| `AU-mkldnn-fusion-register-leaky-relu-fusion-lowering` | `deferred-mkldnn-cpu-only-or-mismapped` | 源码受MKLDNN可用性与CPU/XPU packed op约束；列出的CUDA卷积测试走通用select_algorithm，不执行MKLDNN fusion，其他测例为CPU。 |
| `AU-mkldnn-fusion-register-unary-fusion-lowering` | `deferred-mkldnn-cpu-only-or-mismapped` | 源码受MKLDNN可用性与CPU/XPU packed op约束；列出的CUDA卷积测试走通用select_algorithm，不执行MKLDNN fusion，其他测例为CPU。 |
| `AU-mkldnn-fusion-reshape-linear-reshape` | `deferred-mkldnn-cpu-only-or-mismapped` | 源码受MKLDNN可用性与CPU/XPU packed op约束；列出的CUDA卷积测试走通用select_algorithm，不执行MKLDNN fusion，其他测例为CPU。 |
