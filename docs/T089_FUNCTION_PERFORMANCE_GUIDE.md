# T-089 功能与性能测例讲解

> 更新时间：2026-09-10 23:14:54 CST（UTC+08:00）
> 状态：1 个单元已准备，等待原生 GPU reference；4 个候选延期。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-089 \
  --gpu 2 \
  --wait-gpu
```

## AU-split-cat-move-view-after-cat

```python
# torch/_inductor/fx_passes/split_cat.py:2935
def move_view_after_cat(match, *args, **kwargs):
    # before: split -> getitem -> reshape，每个结果再cat
    # after:  输入（必要时permute）先整体reshape
    view_node = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(permute_node, list(cat_node.meta["val"].shape)),
    )
```

该 pattern 将七路逐片 view + cat 收束成整体 view，减少图节点与中间操作；只有分片完整、索引连续、
view覆盖所有分片且维度合同成立时才改写。社区功能测例使用 `[7,8,96]` GPU 输入，同时保留一个
`clone` 多用户分支，断言 `normalization_aten_pass=4`、目标 counter=1 和 compiled/eager 一致。

性能测例完整复用该图和 shape；normalization 固定 ON，只省略/加入
`move_view_after_cat_aten_pass`。这是目标函数端到端，不是模型端到端；NPU 使用
`triton_experimental`，新进程两臂功能通过并签 gate 后才运行六臂。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t087_t090_performance.py \
  --task T-089 \
  --phase functional \
  --device npu
```

## 延期与阶段纠正

`merge_unbind_stack`、`merge_unbind_stack_aten`、`mutate_cat_node` 缺直接 GPU 目标合同。
`move_reshape_out_of_split_stack` 虽在同一源码文件，实际注册进 `pre_grad_fusion_options`，不是
post-grad；其社区入口还是 CPU，必须在后续 pre-grad 批次重建合同，不能借本单元结果代替。
