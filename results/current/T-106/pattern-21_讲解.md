# T-106 pattern 21：功能通过、单pattern性能归因受限

> 更新时间：2026-09-15 17:21 CST（UTC+08:00）。用户已接受最终处置，不计性能收益或默认关闭免测。

## 意图与代码

21号识别T5样式加性mask attention，用scale=1.0的SDPA替换显式matmul-softmax-matmul。
22号计算主体相同，但还返回转置K/V；两者仍是不同的输出合同，不因本次处置合并分母。

```python
# 冻结PyTorch8e86e0a，torch/_inductor/fx_passes/fuse_attention.py:628、654
# 语义节选：注册/改写在joint FX阶段，早于lowering和kernel选择。
def pattern21(q, k, v, mask):
    return attention(q, k, v, mask)
def pattern22(q, k, v, mask):
    return attention(q, k, v, mask), permuted_k, permuted_v
```

## GPU与NPU功能对照

GPU冻结reference确认本编号命中，原社区方法通过且零skip。
NPU `triton_experimental` 安装态完整原方法通过：2次Tensor比较、2次精确改写，
无候选注入、不改数值容差。代码符号断言按实际NPU数学SDPA展开适配，
不是CUDA融合kernel，因此标记`PARTIAL_ALIGNED`，不外推未测dtype/shape/梯度。
原件、实际FX/IR/output_code均在[安装态原例目录](../../../issues/REF-sfdp-pattern-21-native/evidence/t106-native-6t1pel2i/adapter/)。

## 性能测例与为什么不计时

性能输入来自同编号half-inference社区注册样例的派生微图，不是模型端到端。
只删除21号注册、保留整轮joint和其他注册后，真实具名观察显示22号接替改写。

```text
# 原件：issues/REF-sfdp-pattern-21-native/evidence/functional-20260915T112553+0800/off/
单独移除21注册 → _sfdp_pattern_22_half_inference真实改图
→ eager数值比较通过 → RuntimeError: OFF仍命中fuse_attention
→ 退出码1，未进入计时，未执行ON
```

这里的“OFF”已不是未优化对照，不能把两者时间差归因于21号。
删除22号来制造对照会改变问题为模式组收益，不属于当前批准的单pattern处置。
详见[失败栈与目标FX](../../../issues/REF-sfdp-pattern-21-native/性能归因阻断.md)。

## 用户接受的最终统计

| 项目 | 本号贡献 |
| --- | --- |
| 功能/comparison与最终处置完成 | +1 |
| 性能无法独立归因 | +1 |
| 性能实测 / 性能收益 / 默认关闭免测 | 各+0 |

状态`PERF_NOT_INDEPENDENTLY_ATTRIBUTABLE`，不提供p50/p99或加速比。
[确认记录](../../../issues/REF-sfdp-pattern-21-native/最终处置确认.md)与[机器处置原件](functional/pattern-21.json)分开留存；
结项不抹掉归因限制，也不影响22号已测PERF_REGRESSED结论。
