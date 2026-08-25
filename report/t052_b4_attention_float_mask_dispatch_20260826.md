# T-052：B4 attention float-mask dispatcher 分流报告

## 结论

pattern 5、21、29 的 matcher 命中后重新展开为 BMM + Triton，不是 pass 未触发，也不是 NPU
attention lowering 丢失。根因是它们携带一般 additive float mask，而当前 torch_npu SDPA 的两个
vendor 分支都只接受 bool mask 或无 mask；不满足时按设计调用
`aten::_scaled_dot_product_attention_math`。

无 mask 的 `_sfdp_pattern_30_half_inference` 对照在 fresh cache 中 exact/总 fusion counter 均为
1，最大绝对误差 `0.0009765625`，最终为单个 `npu_fusion_attention_v3` /
`aclnnFlashAttentionScore`。这与 pattern 29 的 BMM + Triton 结果形成直接对照，确认 float
additive mask 是当前代表输入的主分流因素。

## 源码链路

PyTorch replacement 明确保留 additive mask 语义：

- pattern 5：`attn_mask.to(dtype=query.dtype)`，scale 为 `1/inv_scale`；
- pattern 21：T5 float mask 转 query dtype，scale 为 1；
- pattern 29：BERT-like `_safe_softmax` float mask，scale 为 `inv_scale²`；
- pattern 30：与 29 相近，但 `attn_mask=None`。

torch_npu 的
`third_party/op-plugin/op_plugin/ops/opapi/ScaledDotProductAttentionKernelNpuOpApi.cpp` 中：

1. 第一条 FlashAttention 分支要求 `attn_mask.dtype()==bool` 或无 mask；
2. 第二条 FusedInferAttention 分支同样要求 bool/None，并附加 inference、shape、dtype 条件；
3. 两条都不满足时，将 bool mask转换为 math bias，调用 `_scaled_dot_product_attention_math`。

因此 pattern 5/21/29 的 arbitrary float bias 不能无条件转成 bool：0/1、有限负偏置、`-inf` 和
连续位置 bias 的语义都不同。直接转换会把“加性偏置”错误改成“允许/屏蔽”。

## 动态对照

| family | mask | exact counter | 数值 | 最终路径 |
|---|---|---:|---:|---|
| pattern 5 | fp16 additive | 1 | max error 0.0009765625 | 2 BMM + Triton math |
| pattern 21 | fp32 additive | 1 | max error 0.001953125 | 2 BMM + Triton math |
| pattern 29 | fp16 additive | 1 | max error 0.0009765625 | 2 BMM + Triton math |
| pattern 30 | None | 1 | max error 0.0009765625 | 1 vendor FlashAttention |

pattern 30 证据为
`results/t052_b4_attention_pattern30_fresh_20260826/result.json`。第一次未设置 fresh cache，虽然
generated code 是 vendor attention，但 counter 来自已缓存编译，不能用于 exact trigger；第二次
只打开 debug 仍复用 cache，counter 为 0。第三次同时设置独立 Inductor/Triton cache 才是有效
结果。三个目录均保留，不删除中性方法记录。

## 处理建议

- 当前保持 math fallback；它是安全 capability 分流，不标 `unsupported`。
- 不手写完整 attention 来复制已有数学路径。下一步先做 pattern 5 pass-on/pass-off paired，判断
  “先融合成 SDPA 再 math decomposition”相对原图是否有端到端或资源收益。
- 若未来考虑 vendor 路径，必须先证明某个严格子域的 float mask可无损映射到 bool mask，或 vendor
  `pse` 能完整承接 additive bias；需要覆盖广播、有限 bias、0、`-inf`、NaN、dtype、dynamic 和
  backward，再做 paired 性能。没有这些证据不得扩大 gate。
