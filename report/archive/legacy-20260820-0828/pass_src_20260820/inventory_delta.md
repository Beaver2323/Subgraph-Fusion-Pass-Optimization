# Pass/src 与旧 Dynamo 清单差异

## 基线

- 旧清单：`/home/z50063656/Dynamo/pytorch` + `/home/z50063656/Dynamo/torch_npu`，194 条。
- 新清单（旧扫描同口径）：`/home/z50063656/Pass/src/pytorch` + `/home/z50063656/Pass/src/torch_npu`，189 条。
- 当前扩展清单：在新源码基线上补全后端和生成式注册，共 251 条。

## 统计差异

除 `npu-ext` 外，所有阶段和机制统计相同。新清单少 5 个 `extension-function`，`npu-specific` 由 40 变为 35。

缺少的五条为：

1. `_replace_legacy_npu_scatter_nd_update_pass`
2. `run_pre_grad_custom_pass`
3. `run_post_grad_custom_pre_pass`
4. `_batch_embedding_fusion_pass`
5. `_pad_slice_fold_pass`

它们都来自：

```text
third_party/torchair/torchair/experimental/_inductor_npu_ext/
python/inductor_npu_ext/passes/inductor_custom_passes.py
```

## 原因

`Pass/src/torch_npu` 的 `third_party/torchair/torchair` 子模块当前未初始化，`git submodule status` 前缀为 `-`，目录内不存在上述 Python 源码。主干 PyTorch、torch_npu FX pass 记录没有新增或删除。

## 扩展清单新增项

相对 189 条同口径清单，当前增加：

- 8 个 DVM/MLIR backend graph pass/subpass；
- 33 个生成式 pattern family（30 个 SDPA、3 个 pad-mm）；
- 10 个直接 `register_replacement`；
- 3 个函数式 `register_graph_pattern(...)(handler)`；
- 7 个 freezing/binary wrapper pattern；
- 1 个 `_disable_pad_mm_pass` 控制 gate。

这些扩展用于修正旧扫描器的覆盖缺口，不表示 2026-08-20 的源码比旧源码新增了 62 个 pass。

## 判定

该差异属于源码物料不完整，不是 NPU pass 支持度回退。是否将 torchair extension 纳入最终范围，需要在环境稳定后确认项目实际 backend 是否依赖该子模块；确认前保留旧五条为 `source-unavailable`，不能标记 `unsupported`。
