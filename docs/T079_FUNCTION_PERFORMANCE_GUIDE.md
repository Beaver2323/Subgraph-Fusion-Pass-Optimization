# T-079 功能与性能测例讲解

> 更新时间：2026-09-04 09:08 CST（UTC+08:00）
> 状态：4 个功能合同和 4 个性能合同已准备，等待 GPU reference。
> NPU 固定后端：`triton_experimental`；其他后端历史数据不计入 verdict。

## 1. batch=1 的 bmm 降为 mm（`AU-joint-graph-bmm-to-mm`）

代码位置：`torch/_inductor/fx_passes/joint_graph.py:979`。

```python
# torch/_inductor/fx_passes/joint_graph.py:979
def bmm_to_mm(match, mat1, mat2):
    def repl(a, b):
        return torch.mm(a.squeeze(0), b.squeeze(0)).unsqueeze(0)
    if check_device(...) and mat1.meta["val"].shape[0] == 1:
        match.replace_by_example(repl, [mat1, mat2])
```

功能测例 `test_bmm_to_mm` 使用 `[1,16,8]@[1,8,32]` 验证正例生成 `mm`，再用 batch=3 验证
仍保留 `bmm`。它同时说明 pass 的边界：这里只消除“虚假的 batch 维”，不是把任意 bmm 改成 mm。

社区没有性能 benchmark。性能主测先原样复用小 shape；考虑小算子易受 launch 噪声影响，可增加
batch 始终为 1、仅放大 M/N/K 的 sensitivity 网格，但必须和社区 shape 分栏，不能冒充社区数据。

## 2. cat→slice→cat 折叠（`AU-post-grad-cat-slice-cat`）

代码位置：`torch/_inductor/fx_passes/post_grad.py:1089`。

```python
# torch/_inductor/fx_passes/post_grad.py:1089
def cat_slice_cat(match, cat_input, size, dim=1):
    first, *rest = cat_input
    if size >= 0 and statically_known_leq(size, first.get_size()[dim]):
        return L[aten.cat]([first, *rest, L[aten.slice](first, dim, 0, size)], dim)
    return original_two_cat_fallback(...)
```

功能测例使用 `[2,32]` 与 `[2,16]` 输入。slice 终点在第一个输入内时可以直接复用 first，减少一次
中间 cat；终点越界或为负时必须走原始结构。这里“pattern 命中 handler”不等于优化生效，结果中
必须通过 FX/IR 分辨优化分支与 fallback。

性能从合法社区正例派生，测 compiled 端到端，并记录 cat 数、kernel/task 数和中间写入；两个
fallback 分支只做功能性 guard。

## 3. split_with_sizes→cat 消除（`AU-post-grad-splitwithsizes-cat-replace`）

代码位置：`torch/_inductor/fx_passes/post_grad.py:1731`。

```python
# torch/_inductor/fx_passes/post_grad.py:1731
def splitwithsizes_cat_replace(match, input_):
    return input_
```

功能测例把 `[2,32]` 按 `[8,24]` 分开再同维、同序拼回，正例应直接返回输入。缺少一个 getitem、
cat 维不同或 getitem 重排都会改变结果，必须拒绝消除。

性能没有社区 benchmark，复用正例做端到端 OFF/ON；重点不是某个 kernel 变快，而是 split/cat 图
节点与中间量消失，因此 Event、kernel/task 数和内存需要一起看。

## 4. cat→split_with_sizes 消除（`AU-post-grad-cat-splitwithsizes-replace`）

代码位置：`torch/_inductor/fx_passes/post_grad.py:1786`。

```python
# torch/_inductor/fx_passes/post_grad.py:1786
def cat_splitwithsizes_replace(match, input_):
    return input_
```

功能测例把宽度 2、3、5 的三个输入 cat，再按 `[2,3,5]` 拆回。只有 cat 无额外用户、split 维、
数量与边界完全相同才能返回原输入 tuple。cat 多用户、异维、数量或边界不同分别是独立负例。

性能从正例派生，测整个编译图并核对 alias/stride；负例不参与性能。它与上一单元方向相反、guard
不同，不能合并成一个 acceptance unit 或共用一个“命中”结论。

## 5. 结果判读

| 证据 | 回答的问题 |
| --- | --- |
| eager/compiled + 负例 | 语义和 guard 是否正确 |
| counter、FX/IR、generated code | 是否真的执行目标改写，而非 fallback |
| NPU Event p50/p99 | device steady-state 是否受益 |
| kernel/task 数与显存 | 图消除是否减少启动和中间量 |
| host 与首次编译 | 调度尾延迟、编译代价是否恶化 |

机器可读细节见 `upstream/t079_performance_plan.yaml`。正式性能结论只能来自 fresh-process、同源码、
同输入、同 `triton_experimental` backend 的目标级 OFF/ON。
