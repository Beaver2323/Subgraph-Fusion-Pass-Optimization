# T-088 功能与性能测例讲解

> 更新时间：2026-09-10 23:14:54 CST（UTC+08:00）
> 状态：2 个单元已准备，等待原生 GPU reference；3 个 CPU-only 候选延期。

GPU 使用 `inductor-default` 原生社区入口，NPU 动态验证和性能只允许 `triton_experimental`。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-088 \
  --gpu 2 \
  --wait-gpu
```

## AU-split-cat-merge-select-cat-aten

```python
# torch/_inductor/fx_passes/split_cat.py:1907
@register_graph_pattern(..., pass_dict=construct_pattern_matcher_pass(
    "select_cat_aten_pass"
))
def merge_select_cat_aten(match, *args, **kwargs):
    # 同一输入、同一dim、完整连续indices的select+cat可直接view
    view_node = graph.call_function(
        torch.ops.aten.view.default, args=(node_input, cat_node.meta["val"].shape)
    )
```

意图是消掉完整连续 `select` 后再 `cat` 的拆装；非完整、非连续或不同输入的分支保留。社区功能测例
`TestSplitCatAten.test_select_cat_post_grad` 在两个 `[1024,6,128]` GPU tensor 上断言
`normalization_aten_pass=4`、`select_cat_aten_pass=1`，并比较四个输出。

性能测例复用完整社区函数；normalization 固定 ON，只省略/加入 `select_cat_aten_pass`。worker 对
`merge_select_cat_aten` 注册 handler 精确插桩，ON 必须既调用又改图。

## AU-split-cat-merge-split-cat-aten

```python
# torch/_inductor/fx_passes/split_cat.py:1803
def merge_split_cat_aten(match, *args, **kwargs):
    threshold = config.post_grad_fusion_options[
        "split_cat_aten_pass"
    ].get("threshold_to_cat", 10)
    # 连续片段达到阈值时用原输入或slice替换多路getitem
```

意图是将 `split -> 多个连续getitem -> cat` 收束为原输入或单个 slice，减少拆分/拼接开销。正例
输入 `x/y=[1024,128]`、`z=[1024,32]`，阈值 5，目标 counter 为 1；singular 负例只拼一个分片，
counter 必须为 0。两项都是真实 GPU 社区测例。

性能只测正例完整函数，negative 在计时前门禁；normalization 固定 ON，唯一变量是目标 pass。
没有找到社区同后端 OFF/ON benchmark，因此明确标记为“从社区功能测例派生的目标子图”，不能外推为
模型端到端收益。

## 延期候选

`merge_getitem_cat`、`merge_split_squeeze`、`merge_splits` 当前社区入口使用 CPU tensor；保留其结构
价值，但不计 GPU 分母，也不先测 NPU 性能。

## NPU 功能入口

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t087_t090_performance.py \
  --task T-088 \
  --phase functional \
  --device npu
```
