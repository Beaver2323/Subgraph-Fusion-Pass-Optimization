# T-090 功能与性能测例讲解

> 更新时间：2026-09-10 23:14:54 CST（UTC+08:00）
> 状态：1 个单元已准备，等待原生 GPU reference；4 个 normalization 候选延期。

NPU功能、修复验证和性能固定使用`triton_experimental`，并在导入torch前选择后端。

## GPU 一键运行

```bash
cd /data/z50063656/tmp

bash \
  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \
  --task T-090 \
  --gpu 2 \
  --wait-gpu
```

## AU-split-cat-normalize-cat-default-aten

本节同时说明功能测例与性能测例的来源、边界和精确归因方法。

```python
# torch/_inductor/fx_passes/split_cat.py:1959
def normalize_cat_default_aten(match, *args, **kwargs):
    cat_dim = get_arg_value(cat_node, 1, "dim")
    if cat_dim is None:
        cat_dim = cat_node.kwargs.get("axis", 0)
    if cat_dim < 0:
        cat_dim += cat_node.meta["val"].dim()
    new_cat_node = graph.call_function(
        torch.ops.aten.cat.default, args=(tensors,), kwargs={"dim": cat_dim}
    )
```

它不改变数学计算，而是把 positional/`axis`/负 dim 统一成明确的非负 `dim` kwargs，为后续
split-cat pattern 提供稳定图形。改写发生在 post-grad、lowering/codegen/autotune 之前。

原生 `test_split_cat_post_grad` 是真实 GPU 组合测例：`normalization_aten_pass=5`，随后
`split_cat_aten_pass=1`，并检查数值与参数。因为 pass registry 里还包含其他 normalization handler，
GPU 回传后必须人工看 FX；NPU worker进一步对 `normalize_cat_default_aten` 精确插桩。这个结果只覆盖
cat，不外推到 clamp/detach/reshape。

性能图复用社区第一段 `x/y=[1024,128]`、`z=[1024,32]`；split-cat 固定 ON，唯一变量为
normalization pass。无社区单 handler benchmark，因此属于社区功能图派生目标子图。

```bash
cd /home/z50063656/tmp

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t087_t090_performance.py \
  --task T-090 \
  --phase functional \
  --device npu
```

## 延期候选

非 ATen `normalize_cat_default` 以及 `normalize_clamp_default`、`normalize_detach_default`、
`normalize_reshape_default` 没有可单独归因的直接 GPU 目标合同；当前不进入分母和性能测试。
