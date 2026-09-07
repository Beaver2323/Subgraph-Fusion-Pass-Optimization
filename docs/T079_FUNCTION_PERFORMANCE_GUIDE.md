# T-079 功能与性能测例讲解

> 更新时间：2026-09-07 20:50 CST（UTC+08:00）
> 状态：GPU/NPU 功能、性能处置与产品门禁全部完成；4/4 units、14/14 variants 正式收口。
> NPU 固定后端：`triton_experimental`；其他后端历史数据不计入 verdict。

NPU 的逐步操作、证据判定、最小适配和修复门禁统一见
[`NPU_FUNCTION_REPAIR_WORKFLOW.md`](NPU_FUNCTION_REPAIR_WORKFLOW.md)。本页解释 T-079 各单元合同，
逐单元执行合同见 [`T079_T080_NPU_FUNCTION_REPAIR_PLAN.md`](T079_T080_NPU_FUNCTION_REPAIR_PLAN.md)；
NPU 一键入口为 `scripts/run_t079_npu_all.sh`，性能入口为 `scripts/run_t079_performance.sh`。

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

GPU 实测：原生方法 1/1 通过。batch=1 的可读 FX 已是 `squeeze→mm→unsqueeze`，batch=3 仍为
`aten.bmm`。由于该改写发生在 joint graph，当前 debug 的 readable/transformed 捕获点都在改写之后，
所以正例前后签名相同不是“未生效”；命中结论还由社区方法对生成代码中 `mm/bmm` 的断言共同支撑。

社区没有性能 benchmark。性能主测先原样复用小 shape；考虑小算子易受 launch 噪声影响，可增加
batch 始终为 1、仅放大 M/N/K 的 sensitivity 网格，但必须和社区 shape 分栏，不能冒充社区数据。

NPU 结果：最小适配的原社区合同 1/1 PASS，证明 batch=1 能生成 mm、batch=3 保留 bmm；但目标级
6 臂测试中社区 shape 和三个 sensitivity shape 的 Event p50 分别回退 12.46%、31.58%、18.79%、
16.97%。因此 `triton_experimental.config.disable_bmm_to_mm=true`，最终默认产品路径保留 bmm；显式
关闭门禁后仍能恢复 mm。修复前后代码、调用栈与 `output_code.py` 见
`issues/REF-bmm-to-mm-native/{根因分析.md,修复验证报告.md}`。

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

GPU 实测：原生方法 1/1 通过，三个分支都从 `cat→slice→cat` 变为
`torch__inductor_fx_passes_post_grad_cat_slice_cat(...)` handler 节点；合法分支使用 first width=32、
`size=19`，两个 guard 分支分别是 first width=8、`size=19` 和 `size=-1`。FX 说明结构进入 handler，
真正是否折叠由 handler guard 与社区断言判定，不能把三个 handler 节点都记成优化生效。

NPU 结果：1/1 PASS；三条原 counter/node 判据均通过。目标级 Event p50/p99 改善
18.41%/16.88%，保留启用，无产品修复。

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

GPU 实测：原生方法 1/1 通过。完整同序正例的 `split/getitem/cat` 被替换为
`splitwithsizes_cat_replace(input_=arg0)`；缺片、异维和重排三个分支在 transformed FX 中仍保留
原始 `split_with_sizes/getitem/cat`，正负例边界清晰。

NPU 结果：1/1 PASS；正例 `[1,4]`、三个负例均 `[0,0]`。目标级 Event p50/p99 改善
28.46%/26.46%，保留启用，无产品修复。

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

GPU 实测：原生方法 1/1 通过。同边界、同维且 cat 单用户的正例变为
`cat_splitwithsizes_replace(input_=[arg0,arg1,arg2])`；cat 多用户、异维、异数量、异边界四个负例均保留
`cat/split_with_sizes`。这 5 张图与社区计数/数值断言共同覆盖 5 个 variants。

NPU 结果：1/1 PASS；正例 `[1,2]`、四个负例均 `[0,0]`。目标级 Event p50/p99 改善
13.03%/31.43%，保留启用，无产品修复。

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

GPU handoff 复核见 `report/t079_gpu_reference_review_20260907.md`；正式 NPU/GPU 对照位于
`results/current/AU-*/{npu_result.json,comparison_result.json}`，任务级性能与产品处置位于
`results/current/T-079/`。所有 NPU 数字均来自同源码、同输入、fresh-process、
`triton_experimental` 的目标级 OFF/ON。

逐问题阅读入口：

- `issues/REF-bmm-to-mm-native/`：适配、性能回退根因、产品门禁修复验证和合入描述；
- `issues/REF-cat-slice-cat-native/`：入口适配、命中/真正折叠边界与性能证据；
- `issues/REF-splitwithsizes-cat-native/`：split→cat 消除合同；
- `issues/REF-cat-splitwithsizes-native/`：cat→split 消除合同。

这些报告允许重复必要代码框与调用链，使单篇可独立阅读；机器可判定字段仍以 `results/current/` 为准。
