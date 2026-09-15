# T-104 pattern 13：三维 BMM attention 的真实图对照

> 复核时间：2026-09-14 23:22 CST（UTC+08:00）；NPU原例运行：`t104-native-cw5by_le`。

## 1. 本轮结论

原社区方法在当前安装态 `triton_experimental` 下 **1/1、零skip**；精确
`_sfdp_pattern_13_half_inference` 改写一次，FP16输出比较一次通过。
这不是产品修复：原生入口因CUDA测试类定义门禁不能发现测试，完成设备/生成代码符号断言适配后即可通过。
没有修改安装包、没有使用隔离注册候选。

原例只测推理，**不覆盖训练、反向或非零dropout**。独立OFF/ON微图和六臂性能现已完成，
`PERF_REGRESSED`，正式计一个单元；[详细输入、容差、代码、计时与结果](../../results/current/T-104/pattern-13_讲解.md)。

## 2. Pattern 与社区测例

```python
# 冻结 PyTorch 8e86e0a，torch/_inductor/fx_passes/fuse_attention.py:341
def _sfdp_pattern_13(query, key, value, dropout_p):
    attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
    attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
    return torch.bmm(attn_weight, value)

# 同文件:347，节选；counter在原函数中递增。
def _sfdp_replacement_13(query, key, value, dropout_p):
    return _scaled_dot_product_attention(
        query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0),
        dropout_p=dropout_p, scale=1.0,
    ).squeeze(0)
```

意图是将 `BMM → softmax → BMM` 换成SDPA。三维输入补一个最外层维度以符合四维attention接口，
输出再去掉该维度；这里原图没有除以 `sqrt(d)`，因此replacement明确传 **`scale=1.0`**，不能错用默认缩放。

```python
# test/inductor/test_fused_attention.py:885，原方法参数节选
tensor_shape = (4, 8, 16)
self._check_common(
    dot_prod_attention, check_train=False, args1=args,
    has_dropout=True, override_check_equal=True,
    atol=1e-2, rtol=1e-2,
)
# 同文件:1855，GPU方法通过partialmethod固定 dtype=torch.half。
```

虽然方法体写 `dropout(p=0.5, training=training)`，`check_train=False`只运行`training=False`，
所以实际dropout关闭。这是原例合同，不是为NPU临时删除dropout。原方法在NPU保留同样shape、dtype和容差，
比较 **NPU compiled 与 NPU eager**，不是让NPU数值逐位等于GPU。

## 3. 图与生成代码

必要调用链：

```text
SDPAPatternRewriterGpuTests.test_sdpa_rewriter_13_gpu
  -> 原 _test_sdpa_rewriter_13(dtype=half)
  -> _check_common(training=False) -> torch.compile(fullgraph=True)
  -> AOT/Inductor joint_graph -> pattern 13 replacement
  -> SDPA对应设备实现 -> Inductor生成调用 -> 真实设备 -> 原数值断言
```

NPU只读观察器保存的目标图，先有`permute + bmm + float32 softmax展开 + bmm`，
后变为`unsqueeze × 3 + npu_fusion_attention_v3 + squeeze`。这证明发生了图替换，不只是进入handler。
完整原件见[目标改写前后](evidence/t104-native-cw5by_le/adapter/target-observations/3783983-ReplacementPatternEntry-0001/)。

```python
# GPU原件：evidence/gpu-20260914T191834/debug/torch_compile_debug/
# run_2026_09_14_19_19_38_952140-pid_1670285/torchinductor/model__0_inference_0.0/output_code.py
# 以下省略reinterpret_tensor参数展开，调用与scale来自原件。
buf0 = torch.ops.aten._scaled_dot_product_flash_attention.default(q4d, k4d, v4d, scale=1.0)
```

```python
# NPU原件：evidence/t104-native-cw5by_le/adapter/debug/torch_compile_debug/
# run_2026_09_14_22_31_27_567614-pid_3783983/torchinductor/model__0_inference_0.0/output_code.py
# 同样仅省略视图参数与其余默认参数，不是新增可执行测例。
buf0 = torch.ops.npu.npu_fusion_attention_v3.default(
    q4d, k4d, v4d, 4, 'BNSD', scale=1.0, keep_prob=1.0,
)
```

| 检查项 | GPU | NPU |
|---|---|---|
| 输入 | FP16，三份 `[4,8,16]` | 相同原方法合同 |
| 精确目标 | pattern 13 half inference | 同编号、一次图改写 |
| 底层调用 | ATen Flash Attention | NPU Fusion Attention v3 |
| 输出 | 原社区eager比较通过 | 原社区eager比较通过 |
| 训练/梯度 | 此原例未覆盖 | 此原例未覆盖 |
| 性能 | 本次是功能reference | 派生同后端OFF/ON：host p50回退6.77%，Event p50回退5.61% |

NPU生成的`Runner.call`在`torch.npu.utils.device(0)`中直接调用NPU算子，输出通过view返回；
未见CPU转移或CPU算子调用。文件顶部定义`empty_strided_cpu`等通用别名不等于实际使用CPU。
这是对该生成图的人工审查，不把简单字符串扫描的“零告警”当全局无fallback证明。

## 4. 证据与下一步

- [安装态原件与执行器快照](evidence/t104-native-cw5by_le/adapter/)；[GPU原件](evidence/gpu-20260914T191834/)。
- [原生阻断与最小适配说明](适配报告.md)；[安装态基线复核](安装态基线复核.md)。
- 已只删除本编号注册构造OFF，保留132个其他条目；OFF没有邻接attention接替，ON精确命中一次。
- 性能使用注册输入的派生微图，已另过数值/精确目标/生成代码门禁；原社区输入和派生输入/容差分别记录。
