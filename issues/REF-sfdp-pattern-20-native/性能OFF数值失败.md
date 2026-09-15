# Pattern 20：不能把精度错误的OFF用于性能对照

> 更新时间：2026-09-14 23:30 CST（UTC+08:00）。状态：额外FP16注册微图OFF数值失败，ON未运行，未计时。

原社区方法`t105-native-jq9qae5o`通过结构/执行合同，但它没有数值oracle。
此次注册微图在**OFF编译输出与同输入NPU eager**比较中失败；不得改写原例为“精度已通过”。
[失败原件、完整FX/IR/output_code与执行器快照](evidence/functional-20260914T232619+0800/)已入库。

```python
# torch/_inductor/fx_passes/fuse_attention.py:588，原search主体节选
q = query.permute([0, 2, 1, 3]).div(inv_scale)
k = key.permute([0, 2, 1, 3])
scores = q @ k.transpose(-2, -1)
attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
return torch.nn.functional.dropout(
    torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), dropout_p
) @ v
```

Q/K/V是连续FP16 `[2,4,8,16]`，mask是注册的FP16 `[2,4]`，按社区0/1下三角设计初始化。
每个mask行至少一个允许位置；inference dropout=0。不是随机全遮挡行用例，也不是性能采样噪声。

```text
t102_t107_attention_performance_worker.py:325
  → assert_close(actual, expected), atol=0.002, rtol=0.2
AssertionError: Tensor-likes are not close!
Mismatched elements: 544 / 1024 (53.1%)
Greatest absolute difference: nan at index (0, 0, 1, 0)
```

实际OFF输出代码包含Q缩放/转置、两次NPU bmm和多个Triton softmax展开kernel；
其中也有`extract_slice`，但尚未逐kernel定位，不能直接认定与22号根因相同。
ON没有运行，不能说融合后修复或加速。后续先诊断错误OFF，再按同合同复验；
保留原种子、输入域、容差和原件，不用重抽输入、替换后端或NaN相等选项制造通过。
