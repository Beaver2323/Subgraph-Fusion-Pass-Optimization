# Pattern 15：标量入口修正后的 OFF 数值失败

> 更新时间：2026-09-15 00:25 CST（UTC+08:00）。状态：真实设备失败已归档，尚未定位首个错误 kernel；未运行 ON 或计时。

这次失败不是上次 `exact=0` 的同义表述。此前零维 Tensor scale 被合法 guard 拒绝；
仅将具名 `inv_scale` 在 compile 外还原为 Python 数值后，新进程 OFF 就出现 NaN。
原社区方法此前通过仍保留；这里失败的是注册输入派生的独立 FP16 微图。

## 输入、步骤与实际调用栈

物理 NPU 5，`triton_experimental` 在导入前选择；Q/K/V 为 FP16 `[2,4,8,16]`，
mask 为 FP16 `[2,4]` 的 0/1 下三角，每一行至少有一个可见元素，`inv_scale=2.0`。
仅移除本编号注册，保留整轮 joint 和其他优化；未修改产品或数值容差。

```python
# 冻结 torch/_inductor/fx_passes/fuse_attention.py:388—397（逻辑节选）
q, k, v = query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)
scores = (q @ k.transpose(-2, -1)) / inv_scale
mask = (attn_mask == 0).view(bs, 1, 1, k_len).expand_as(scores)
return torch.softmax(scores.masked_fill(mask, -float("inf")), dim=-1) @ v
```

```text
# evidence/functional-20260915T000318+0800/off/stderr.log
t102_t107_attention_performance_worker.py:350 main
 -> assert_close:109
 -> torch.testing.assert_close(actual, expected, rtol=0.2, atol=2e-3)
AssertionError: Tensor-likes are not close!
Mismatched elements: 400 / 1024 (39.1%)
Greatest absolute difference: nan at index (0, 0, 1, 0)
```

已逐读本次 FX 与 `Runner.call`：执行顺序为 Q/K 转置物化 → FP16 bmm →
四个 softmax/mask 相关 Triton kernel → V 转置物化 → FP16 bmm，未生成 FA。

```python
# 本轮 output_code.py，Runner.call，以下仅保留核心顺序
extern_kernels.bmm(..., out=buf2)
triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_2.run(...)
triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_3.run(...)
triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_4.run(...)
triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_5.run(...)
extern_kernels.bmm(..., out=buf8)
```

[完整 FX、IR、生成代码、日志、启动参数和执行器快照](evidence/functional-20260915T000318+0800/)
均已保留。执行器在数值断言处终止，没有成功 `result.json`，不补造通过记录。

## 尚缺什么

```python
# 本轮 output_code.py:303—304，第一处行最大值相关 kernel
_es_full0 = tl.load(in_ptr0 + 4*x2 + _es_lane0, x2mask).to(tl.float32)
tmp0 = extract_slice(_es_full0, [0,0,0], [real_block_x2,real_block_x1,1], [1,1,1])
# 加载不含 x1；实际该外维应为 singleton，而切片写成 real_block_x1。
```

这里已经找到与 22 号类似的广播形状不一致；随后 exp-sum kernel 还包含独立的 strided-slice 路径。
需逐 kernel 比较中间量，确认首处分歧及是否存在越界、广播或归约错误。
22 号已证实的 select-load 问题只能作为排查线索，不能据相似症状直接套用修复或认定同根因。
OFF 未过门禁，不运行 ON 性能，不把未测记成中性、退化或免测；修复后重新独立验证双方。

[此前标量入口适配及其代码解释](性能入口适配分析.md)保留，便于区分“测试入口问题”和后续真实数值问题。
