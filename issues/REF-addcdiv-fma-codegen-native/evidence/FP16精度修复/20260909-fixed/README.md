# FP16 value=1 修复后证据

> 生成时间：2026-09-09 18:05:53 CST（UTC+08:00）  
> 设备：Ascend 910B2  
> 后端：`triton_experimental`

本目录保存 T-078 FP16 `torch.addcdiv(..., value=1)` 普通 `div -> add` 路径的修复后
原件。这个分支不属于 addcdiv FMA pattern，因此正确结构仍是：

```text
aten.div.Tensor -> aten.add.Tensor
addcdiv_fma_fused = 0
```

修复前 quotient 以 FP32 计算值直接进入 add，与 NPU eager 先物化 FP16 div 结果的
舍入边界不同：

```python
tmp3 = tmp1 / tmp2
tmp4 = tmp0 + tmp3
```

修复后生成代码在 div 与 add 之间恢复 FP16 round-trip：

```python
tmp3 = (tmp1 / tmp2)
tmp4 = tmp3.to(tl.float16)
tmp5 = tmp4.to(tl.float32)
tmp6 = tmp0 + tmp5
```

实测结果：`bitwise_equal=true`、`mismatch_count=0`、`max_abs_error=0`、
`addcdiv_fma_fused=0`、显式 FP16 round 为 1 次，无 `tl.fma` 和 `div_rn`。

证据导航：

- `fp16_source_fix_summary.json`：4 个 value 加 2 个 guard 的六进程汇总；
- `evidence_manifest.json`：本目录证据的字节数、SHA256 和原始运行目录；
- `value-one/arm_result.json`：value=1 结构、精度和 codegen 断言；
- `value-one/fx_graph_readable.py` 与 `fx_graph_transformed.py`：修复后 FX 前后图；
- `value-one/ir_pre_fusion.txt` 与 `ir_post_fusion.txt`：修复后 Inductor IR；
- `value-one/output_code.py`：最终 Triton Ascend 代码；
- `bf16-value-one/arm_result.json`：BF16 邻接回归，确认非 FP16 路径未被改变。

修复前对照位于同级 `20260909-current/value-one-failure/`。详细诊断见
`issues/REF-addcdiv-fma-codegen-native/根因分析.md` 和 `FP16精度修复报告.md`。
