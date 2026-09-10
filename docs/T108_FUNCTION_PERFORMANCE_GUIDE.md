# T-108 功能与性能测例讲解

> 更新时间：2026-09-10 22:55:21 CST（UTC+08:00）
> 状态：0 个GPU-ready单元，5 个明确延期。

NPU功能、修复验证和性能统一使用 `triton_experimental`，并在导入`torch`/`torch_npu`前选后端；OFF/ON每臂使用新进程。

## 功能测例

```python
# torch/_inductor/fx_passes/quantization.py
quantized_decomposed = torch.ops.quantized_decomposed
# qconv/qlinear/woq注册最终生成MKLDNN/quantized CPU lowering
```

这些候选是CPU量化/MKLDNN路径。没有原生CUDA目标命中合同，也没有可合法转成NPU `triton_experimental` 的同一实现；因此不是‘社区开启却漏测’，而是本GPU→NPU审计范围不适用。

## 性能测例

本批没有通过真实GPU目标合同的独立单元，因此不派生OFF/ON worker、不执行设备性能，也不制造NPU ON路径。

## 延期候选

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `AU-quantization-qcat` | `deferred-cpu-mkldnn-quantization-only` | 该入口生成quantized_decomposed/MKLDNN CPU lowering；映射测例为空或只走CPU/MKLDNN，不能证明CUDA/NPU合同。 |
| `AU-quantization-qconv` | `deferred-cpu-mkldnn-quantization-only` | 该入口生成quantized_decomposed/MKLDNN CPU lowering；映射测例为空或只走CPU/MKLDNN，不能证明CUDA/NPU合同。 |
| `AU-quantization-qconv-binary` | `deferred-cpu-mkldnn-quantization-only` | 该入口生成quantized_decomposed/MKLDNN CPU lowering；映射测例为空或只走CPU/MKLDNN，不能证明CUDA/NPU合同。 |
| `AU-quantization-qlinear` | `deferred-cpu-mkldnn-quantization-only` | 该入口生成quantized_decomposed/MKLDNN CPU lowering；映射测例为空或只走CPU/MKLDNN，不能证明CUDA/NPU合同。 |
| `AU-quantization-qlinear-binary` | `deferred-cpu-mkldnn-quantization-only` | 该入口生成quantized_decomposed/MKLDNN CPU lowering；映射测例为空或只走CPU/MKLDNN，不能证明CUDA/NPU合同。 |
