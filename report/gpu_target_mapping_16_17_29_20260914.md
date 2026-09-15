# 16 / 17 / 29：原例通过为什么仍不能冻结本编号

> 更新时间：2026-09-15 01:30 CST（UTC+08:00）。仅复核已有 GPU 工件和冻结源码，本报告没有新增设备执行。

最新处置：[17已核验默认注册去重、16/29补测准备](remaining_work_20260915.md)。
17不再保持独立待命中状态，归并15；下面“保持待映射”的表格是9月14日历史分析。

## 结论

不是让用户反复重跑同一命令：现有输入会消去区分目标编号的运算。原包有效，但只能作为邻接行为证据。
保留原始收件、编号、哈希和历史记录；三行继续不计独立冻结分母，不用其他编号冒充。

| 待核编号 | 已观察目标 | 原因 | 当前处置 |
|---|---|---|---|
| 16 | 14 / 5 | CUDA选的是inference入口，training=False，dropout成为恒等 | 作为无dropout邻接证据；不擅自打开CUDA训练路径 |
| 17 | 15 | 原方法显式check_train=False，dropout同样消失 | 原例不能证明dropout专属17；保持待映射 |
| 29 | 30 | 原图构造全零mask，add(mask)被消去，变为无mask形式 | 需要独立评审非平凡mask输入的最小GPU适配，不盲重跑原例 |

## 16 / 17 的调用链与代码

```python
# PyTorch/test/inductor/test_fused_attention.py:1872
test_sdpa_rewriter_16_inference_gpu = functools.partialmethod(
    TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16,
    check_train=False, override_check_equal=True,
)
# 同文件:1120，17原方法
self._check_common(dot_prod_attention, check_train=False, has_dropout=True)
# 同文件:_check_common:81、95
for training in [False, True] if check_train else [False]:
    dropout_arg = [training] if has_dropout else []
```

```text
原CUDA partialmethod → _check_common只走training=False
 → F.dropout(..., training=False)恒等
 → 无dropout的14/5或15实际替换 → 原测试通过
```

16 的 CUDA replacement 另有保留数学路径的专属逻辑，不能把 ROCm 的训练入口移植为 CUDA 产品结论。
17 的 `has_dropout=True, override_check_equal=False` 还会让 `_check_common` 不比较输出，
原通过主要证明编译/命中边界，不能额外声明数值和梯度对齐。

## 29 的代码与语义

```python
# PyTorch/test/inductor/test_fused_attention.py:1663，29原例
attn_mask = torch.zeros(1, 1, query.size(1), key.size(1),
                        device=query.device, dtype=query.dtype)
attn_weight = (q @ k_t) + attn_mask
attn_weight = torch.ops.aten._safe_softmax(attn_weight, -1)
return attn_weight @ v
# torch/_inductor/fx_passes/fuse_attention.py:876、906
# pattern29: scaled QK + mask → safe_softmax → matmul
# pattern30: scaled QK        → safe_softmax → matmul
```

移除全零加法后实际是30。应先评审将mask改为非平凡输入是否仍符合29的search/replace、广播及dtype合同，
再提供独立GPU补充入口；不能静默修改原社区方法或当前计划快照。

## 证据入口

[T-105逐例目标观察](../results/current/T-105/gpu_reference_review.json)、
[T-107逐例目标观察](../results/current/T-107/gpu_reference_review.json)、
[本轮总复核](gpu_observer_review_20260914.md)。
这里完成的是“判明原映射为何不足”，不是三项新reference已通过。
