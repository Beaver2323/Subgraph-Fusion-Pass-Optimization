# Inductor Pass NPU Audit

> This is a source inventory plus a runtime measurement contract. Source evidence is not a performance claim.

- PyTorch source: `/home/z50063656/Dynamo/pytorch`
- torch_npu source: `/home/z50063656/Dynamo/torch_npu`
- Records: **194**

## Scope and verdicts

The inventory covers upstream `torch/_inductor/fx_passes`, NPU `torch_npu/_inductor/fx_passes`, the NPU Triton experimental FX pass, and the `inductor-npu-ext` custom pass entry points. It includes pipeline observers, pattern entries, custom NPU passes, and scheduler/extension hooks. `static_status` only chooses the next validation route:

- `generic-pipeline` / `generic-needs-validation`: no direct backend blocker found.
- `needs-npu-validation`: CUDA, CPU-only, or collective assumptions are visible.
- `backend-sensitive`: fallback/Triton/template/autotune behavior can change with backend.
- `npu-specific`: implementation is in torch_npu and must be checked for correctness and performance.

## Summary

| Dimension | Count |
|---|---:|
| Total records | 194 |
| Stage `inductor-extension` | 74 |
| Stage `joint_graph` | 13 |
| Stage `npu-custom` | 3 |
| Stage `npu-ext` | 5 |
| Stage `npu-pattern` | 1 |
| Stage `npu-triton-experimental` | 3 |
| Stage `post_grad` | 69 |
| Stage `pre_grad` | 26 |
| Mechanism `extension-function` | 25 |
| Mechanism `npu-custom-pass` | 27 |
| Mechanism `pattern-entry` | 76 |
| Mechanism `pattern-registry` | 25 |
| Mechanism `pipeline` | 41 |
| Static status `backend-sensitive` | 2 |
| Static status `generic-needs-validation` | 101 |
| Static status `generic-pipeline` | 41 |
| Static status `needs-npu-validation` | 10 |
| Static status `npu-specific` | 40 |

## Runtime result contract

A pass is `available` only when its triggering case compiles and its output matches eager within the dtype-specific tolerance. It is `fast` only when the paired NPU benchmark records CANN version, SoC, warmup, sample count, mean, stdev, p50, p99, and peak memory. Any compile/fallback/accuracy error is recorded separately; SKIP is never PASS.

The executable probe is `run_npu_probe.py`. Launch it from `/home/z50063656/tmp` with an installed torch_npu and a visible NPU. The current host has no visible NPU, so this checkout cannot supply runtime numbers.

## High-priority follow-up

1. `mm_plus_mm`: the current NPU patch narrows the shared pattern predicate because NPU lowering is not implemented. First replacement candidate is a fused two-matmul-plus-add backend kernel; validate against standalone matmul and the CATLASS path before enabling the pass.
2. NPU custom passes in `ascend_custom_passes/`: run graph-level correctness tests before performance attribution; their benefit is often kernel-count reduction rather than a single handwritten Triton op.
3. Triton/template/fallback-sensitive records: compare `TORCHINDUCTOR_NPU_BACKEND=triton`, `ascendc`, and `mlir` independently in fresh processes. A pass should not be enabled globally based on one backend's result.
4. Collectives, DDP/FSDP, and multi-stream scheduler passes require distributed tests and must not be inferred from a single-device microbenchmark.

## Full inventory

| Stage | Name | Mechanism | Source | Signals | Static status |
|---|---|---|---|---|---|
| inductor-extension | `apply_gumbel_max_trick` | pattern-entry | `torch/_inductor/fx_passes/apply_gumbel_max_trick.py:26` | - | generic-needs-validation |
| inductor-extension | `b2b_gemm_pass` | pattern-registry | `torch/_inductor/fx_passes/b2b_gemm.py:36` | - | generic-needs-validation |
| inductor-extension | `b2b_gemm_handler` | pattern-entry | `torch/_inductor/fx_passes/b2b_gemm.py:605` | - | generic-needs-validation |
| inductor-extension | `decompose_bmm` | pattern-entry | `torch/_inductor/fx_passes/decompose_mem_bound_mm.py:228` | - | generic-needs-validation |
| inductor-extension | `decompose_addmm` | pattern-entry | `torch/_inductor/fx_passes/decompose_mem_bound_mm.py:247` | - | generic-needs-validation |
| inductor-extension | `decompose_mm` | pattern-entry | `torch/_inductor/fx_passes/decompose_mem_bound_mm.py:271` | - | generic-needs-validation |
| inductor-extension | `efficient_conv_bn_eval_graph_transform_inlined` | pattern-entry | `torch/_inductor/fx_passes/efficient_conv_bn_eval.py:158` | dynamic_shape | generic-needs-validation |
| inductor-extension | `efficient_conv_bn_eval_graph_transform_decomposed` | pattern-entry | `torch/_inductor/fx_passes/efficient_conv_bn_eval.py:273` | - | generic-needs-validation |
| inductor-extension | `efficient_conv_bn_eval_graph_transform` | pattern-entry | `torch/_inductor/fx_passes/efficient_conv_bn_eval.py:373` | - | generic-needs-validation |
| inductor-extension | `binary_folding_pass` | pattern-registry | `torch/_inductor/fx_passes/freezing_patterns.py:34` | - | generic-needs-validation |
| inductor-extension | `freezing_passes` | extension-function | `torch/_inductor/fx_passes/freezing_patterns.py:37` | mkldnn | needs-npu-validation |
| inductor-extension | `unnecessary_dtype_convert` | pattern-entry | `torch/_inductor/fx_passes/freezing_patterns.py:308` | - | generic-needs-validation |
| inductor-extension | `_get_dedup_rs_pass` | extension-function | `torch/_inductor/fx_passes/fsdp.py:101` | - | generic-needs-validation |
| inductor-extension | `dedup_reduce_scatter` | pattern-registry | `torch/_inductor/fx_passes/fsdp.py:109` | - | generic-needs-validation |
| inductor-extension | `_get_dedup_rs_pass._` | pattern-entry | `torch/_inductor/fx_passes/fsdp.py:152` | - | generic-needs-validation |
| inductor-extension | `group_batch_fusion_passes` | extension-function | `torch/_inductor/fx_passes/group_batch_fusion.py:1679` | - | generic-needs-validation |
| inductor-extension | `patterns` | pattern-registry | `torch/_inductor/fx_passes/micro_pipeline_tp.py:27` | - | generic-needs-validation |
| inductor-extension | `micro_pipeline_tp_pass` | extension-function | `torch/_inductor/fx_passes/micro_pipeline_tp.py:1299` | collective | needs-npu-validation |
| inductor-extension | `grouped_gemm_pass` | extension-function | `torch/_inductor/fx_passes/mkldnn_fusion.py:208` | mkldnn | needs-npu-validation |
| inductor-extension | `_register_unary_fusion_lowering.fn` | pattern-entry | `torch/_inductor/fx_passes/mkldnn_fusion.py:474` | - | generic-needs-validation |
| inductor-extension | `_register_leaky_relu_fusion_lowering.fn` | pattern-entry | `torch/_inductor/fx_passes/mkldnn_fusion.py:494` | - | generic-needs-validation |
| inductor-extension | `_register_hardtanh_fusion_lowering.fn` | pattern-entry | `torch/_inductor/fx_passes/mkldnn_fusion.py:542` | - | generic-needs-validation |
| inductor-extension | `_register_binary_unary_fusion_lowering.fn` | pattern-entry | `torch/_inductor/fx_passes/mkldnn_fusion.py:772` | - | generic-needs-validation |
| inductor-extension | `_register_binary_unary_maybe_inplace_fusion_lowering.fn` | pattern-entry | `torch/_inductor/fx_passes/mkldnn_fusion.py:854` | dynamic_shape, memory_alias | generic-needs-validation |
| inductor-extension | `_register_weight_pack_pass` | extension-function | `torch/_inductor/fx_passes/mkldnn_fusion.py:1403` | mkldnn, dynamic_shape | needs-npu-validation |
| inductor-extension | `qconv` | pattern-entry | `torch/_inductor/fx_passes/quantization.py:426` | - | generic-needs-validation |
| inductor-extension | `qlinear` | pattern-entry | `torch/_inductor/fx_passes/quantization.py:539` | - | generic-needs-validation |
| inductor-extension | `qlinear_binary` | pattern-entry | `torch/_inductor/fx_passes/quantization.py:598` | - | generic-needs-validation |
| inductor-extension | `qconv_binary` | pattern-entry | `torch/_inductor/fx_passes/quantization.py:772` | memory_alias | generic-needs-validation |
| inductor-extension | `qmaxpool2d` | pattern-entry | `torch/_inductor/fx_passes/quantization.py:920` | - | generic-needs-validation |
| inductor-extension | `qcat` | pattern-entry | `torch/_inductor/fx_passes/quantization.py:1066` | - | generic-needs-validation |
| inductor-extension | `qreshape` | pattern-entry | `torch/_inductor/fx_passes/quantization.py:1112` | dynamic_shape | generic-needs-validation |
| inductor-extension | `woq_int8` | pattern-entry | `torch/_inductor/fx_passes/quantization.py:1273` | - | generic-needs-validation |
| inductor-extension | `_build_pattern_pass` | extension-function | `torch/_inductor/fx_passes/reduced_atomic_contention.py:565` | - | generic-needs-validation |
| inductor-extension | `partitioned_scatter_optimization` | pattern-registry | `torch/_inductor/fx_passes/reduced_atomic_contention.py:571` | - | generic-needs-validation |
| inductor-extension | `partitioned_scatter_optimization_pass` | extension-function | `torch/_inductor/fx_passes/reduced_atomic_contention.py:621` | - | generic-needs-validation |
| inductor-extension | `patterns` | pattern-registry | `torch/_inductor/fx_passes/replace_random.py:21` | - | generic-needs-validation |
| inductor-extension | `replace_random_passes` | extension-function | `torch/_inductor/fx_passes/replace_random.py:59` | - | generic-needs-validation |
| inductor-extension | `fuse_seed_creation_pass` | pipeline | `torch/_inductor/fx_passes/replace_random.py:65` | - | generic-pipeline |
| inductor-extension | `fuse_offset_creation_pass` | pipeline | `torch/_inductor/fx_passes/replace_random.py:68` | - | generic-pipeline |
| inductor-extension | `fuse_offset_creation_pass` | extension-function | `torch/_inductor/fx_passes/replace_random.py:74` | - | generic-needs-validation |
| inductor-extension | `fuse_seed_creation_pass` | extension-function | `torch/_inductor/fx_passes/replace_random.py:121` | - | generic-needs-validation |
| inductor-extension | `replace_random` | pattern-entry | `torch/_inductor/fx_passes/replace_random.py:185` | cuda | needs-npu-validation |
| inductor-extension | `replace_randint` | pattern-entry | `torch/_inductor/fx_passes/replace_random.py:245` | - | generic-needs-validation |
| inductor-extension | `construct_pattern_matcher_pass` | extension-function | `torch/_inductor/fx_passes/split_cat.py:100` | - | generic-needs-validation |
| inductor-extension | `normalize_split_default` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:229` | - | generic-needs-validation |
| inductor-extension | `remove_split_with_size_one` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:241` | dynamic_shape | generic-needs-validation |
| inductor-extension | `normalize_unbind_default` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:284` | - | generic-needs-validation |
| inductor-extension | `normalize_cat_default` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:321` | dynamic_shape | generic-needs-validation |
| inductor-extension | `normalize_stack_default` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:383` | - | generic-needs-validation |
| inductor-extension | `normalize_squeeze_default` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:429` | - | generic-needs-validation |
| inductor-extension | `normalize_reshape_default` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:469` | dynamic_shape | generic-needs-validation |
| inductor-extension | `normalize_clamp_default` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:498` | dynamic_shape | generic-needs-validation |
| inductor-extension | `normalize_detach_default` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:531` | dynamic_shape | generic-needs-validation |
| inductor-extension | `merge_splits` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:599` | - | generic-needs-validation |
| inductor-extension | `merge_split_squeeze` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1307` | - | generic-needs-validation |
| inductor-extension | `merge_unbind_stack` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1373` | - | generic-needs-validation |
| inductor-extension | `simplify_split_cat` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1438` | - | generic-needs-validation |
| inductor-extension | `merge_getitem_cat` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1523` | memory_alias | generic-needs-validation |
| inductor-extension | `mutate_cat_node` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1630` | dynamic_shape | generic-needs-validation |
| inductor-extension | `normalize_split_default_aten` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1707` | dynamic_shape | generic-needs-validation |
| inductor-extension | `normalize_split_with_size_default_aten` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1757` | dynamic_shape | generic-needs-validation |
| inductor-extension | `merge_split_cat_aten` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1803` | - | generic-needs-validation |
| inductor-extension | `merge_select_cat_aten` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1907` | dynamic_shape, memory_alias | generic-needs-validation |
| inductor-extension | `normalize_cat_default_aten` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:1959` | dynamic_shape | generic-needs-validation |
| inductor-extension | `merge_unbind_stack_aten` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:2013` | - | generic-needs-validation |
| inductor-extension | `split_cat_to_slices` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:2369` | - | generic-needs-validation |
| inductor-extension | `unbind_cat_to_view` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:2452` | dynamic_shape, memory_alias | generic-needs-validation |
| inductor-extension | `split_stack_to_cats` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:2626` | - | generic-needs-validation |
| inductor-extension | `unbind_stack_to_slices` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:2699` | dynamic_shape, memory_alias | generic-needs-validation |
| inductor-extension | `move_reshape_out_of_split_stack` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:2797` | dynamic_shape | generic-needs-validation |
| inductor-extension | `move_view_after_cat` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:2935` | dynamic_shape, memory_alias | generic-needs-validation |
| inductor-extension | `replace_einsum_to_pointwise` | pattern-entry | `torch/_inductor/fx_passes/split_cat.py:3037` | - | generic-needs-validation |
| inductor-extension | `AscendCustomPostPass` | extension-function | `torch_npu/_inductor/fx_passes/graph_match_pass.py:5` | - | npu-specific |
| joint_graph | `early_patterns` | pattern-registry | `torch/_inductor/fx_passes/joint_graph.py:49` | - | generic-needs-validation |
| joint_graph | `patterns` | pattern-registry | `torch/_inductor/fx_passes/joint_graph.py:50` | - | generic-needs-validation |
| joint_graph | `joint_custom_pre_pass` | pipeline | `torch/_inductor/fx_passes/joint_graph.py:718` | - | generic-pipeline |
| joint_graph | `remove_noop_ops` | pipeline | `torch/_inductor/fx_passes/joint_graph.py:725` | - | generic-pipeline |
| joint_graph | `constant_fold_uniform_value` | pipeline | `torch/_inductor/fx_passes/joint_graph.py:728` | - | generic-pipeline |
| joint_graph | `joint_custom_post_pass` | pipeline | `torch/_inductor/fx_passes/joint_graph.py:763` | - | generic-pipeline |
| joint_graph | `fix_iota_device` | pattern-entry | `torch/_inductor/fx_passes/joint_graph.py:788` | cuda, mkldnn | needs-npu-validation |
| joint_graph | `pointless_convert` | pattern-entry | `torch/_inductor/fx_passes/joint_graph.py:842` | - | generic-needs-validation |
| joint_graph | `pointless_view` | pattern-entry | `torch/_inductor/fx_passes/joint_graph.py:918` | dynamic_shape, memory_alias | generic-needs-validation |
| joint_graph | `pointless_view_pair` | pattern-entry | `torch/_inductor/fx_passes/joint_graph.py:936` | dynamic_shape | generic-needs-validation |
| joint_graph | `pointless_permute_pair` | pattern-entry | `torch/_inductor/fx_passes/joint_graph.py:957` | - | generic-needs-validation |
| joint_graph | `bmm_to_mm` | pattern-entry | `torch/_inductor/fx_passes/joint_graph.py:979` | cuda, dynamic_shape | needs-npu-validation |
| joint_graph | `scatter_upon_const_tensor` | pattern-entry | `torch/_inductor/fx_passes/joint_graph.py:1203` | dynamic_shape, memory_alias | generic-needs-validation |
| npu-custom | `run_register_pre_custom_passes` | extension-function | `torch_npu/_inductor/fx_passes/ascend_custom_passes/__init__.py:13` | - | npu-specific |
| npu-custom | `run_register_post_custom_passes` | extension-function | `torch_npu/_inductor/fx_passes/ascend_custom_passes/__init__.py:28` | - | npu-specific |
| npu-custom | `register_custom_pass` | extension-function | `torch_npu/_inductor/fx_passes/ascend_custom_passes/register_custom_pass.py:28` | - | npu-specific |
| npu-ext | `_replace_legacy_npu_scatter_nd_update_pass` | extension-function | `third_party/torchair/torchair/experimental/_inductor_npu_ext/python/inductor_npu_ext/passes/inductor_custom_passes.py:22` | npu, memory_alias | npu-specific |
| npu-ext | `run_pre_grad_custom_pass` | extension-function | `third_party/torchair/torchair/experimental/_inductor_npu_ext/python/inductor_npu_ext/passes/inductor_custom_passes.py:45` | - | npu-specific |
| npu-ext | `run_post_grad_custom_pre_pass` | extension-function | `third_party/torchair/torchair/experimental/_inductor_npu_ext/python/inductor_npu_ext/passes/inductor_custom_passes.py:51` | - | npu-specific |
| npu-ext | `_batch_embedding_fusion_pass` | extension-function | `third_party/torchair/torchair/experimental/_inductor_npu_ext/python/inductor_npu_ext/passes/inductor_custom_passes.py:56` | dynamic_shape | npu-specific |
| npu-ext | `_pad_slice_fold_pass` | extension-function | `third_party/torchair/torchair/experimental/_inductor_npu_ext/python/inductor_npu_ext/passes/inductor_custom_passes.py:525` | - | npu-specific |
| npu-pattern | `npu_fusion_attention_graph` | extension-function | `torch_npu/_inductor/fx_passes/pattern_match/npu_fusion_attention_graph.py:160` | - | npu-specific |
| npu-triton-experimental | `_elide_int_float_int_roundtrip_pass` | extension-function | `torch_npu/_inductor/triton_experimental/fx_passes.py:25` | npu, dynamic_shape, memory_alias | npu-specific |
| npu-triton-experimental | `_install_elide_int_float_int_pass` | extension-function | `torch_npu/_inductor/triton_experimental/fx_passes.py:378` | npu | npu-specific |
| npu-triton-experimental | `_disable_addmm_fusion_pass` | extension-function | `torch_npu/_inductor/triton_experimental/fx_passes.py:534` | npu | npu-specific |
| post_grad | `reorder_for_locality` | pipeline | `torch/_inductor/fx_passes/post_grad.py:196` | - | generic-pipeline |
| post_grad | `post_grad_custom_pre_pass` | pipeline | `torch/_inductor/fx_passes/post_grad.py:209` | - | generic-pipeline |
| post_grad | `remove_profiler_ops` | pipeline | `torch/_inductor/fx_passes/post_grad.py:230` | - | generic-pipeline |
| post_grad | `respecialize_current_device` | pipeline | `torch/_inductor/fx_passes/post_grad.py:236` | - | generic-pipeline |
| post_grad | `post_grad_custom_pre_pass` | pipeline | `torch/_inductor/fx_passes/post_grad.py:242` | - | generic-pipeline |
| post_grad | `remove_noop_ops` | pipeline | `torch/_inductor/fx_passes/post_grad.py:245` | - | generic-pipeline |
| post_grad | `remove_assert_ops` | pipeline | `torch/_inductor/fx_passes/post_grad.py:246` | - | generic-pipeline |
| post_grad | `partitioned_scatter_optimization` | pipeline | `torch/_inductor/fx_passes/post_grad.py:254` | - | generic-pipeline |
| post_grad | `fuse_ddp_communication` | pipeline | `torch/_inductor/fx_passes/post_grad.py:286` | - | generic-pipeline |
| post_grad | `post_grad_custom_post_pass` | pipeline | `torch/_inductor/fx_passes/post_grad.py:301` | - | generic-pipeline |
| post_grad | `chain_random_ops_ordering` | pipeline | `torch/_inductor/fx_passes/post_grad.py:306` | - | generic-pipeline |
| post_grad | `stable_sort` | pipeline | `torch/_inductor/fx_passes/post_grad.py:310` | - | generic-pipeline |
| post_grad | `move_constructors_to_cuda` | pipeline | `torch/_inductor/fx_passes/post_grad.py:312` | - | generic-pipeline |
| post_grad | `decomp_comms` | pipeline | `torch/_inductor/fx_passes/post_grad.py:337` | - | generic-pipeline |
| post_grad | `dedup_reduce_scatters` | pipeline | `torch/_inductor/fx_passes/post_grad.py:344` | - | generic-pipeline |
| post_grad | `bucket_reduce_scatters` | pipeline | `torch/_inductor/fx_passes/post_grad.py:357` | - | generic-pipeline |
| post_grad | `bucket_all_reduce` | pipeline | `torch/_inductor/fx_passes/post_grad.py:369` | - | generic-pipeline |
| post_grad | `bucket_all_gathers` | pipeline | `torch/_inductor/fx_passes/post_grad.py:389` | - | generic-pipeline |
| post_grad | `overlap_scheduling` | pipeline | `torch/_inductor/fx_passes/post_grad.py:447` | - | generic-pipeline |
| post_grad | `replace_collectives_with_low_contention` | pipeline | `torch/_inductor/fx_passes/post_grad.py:458` | - | generic-pipeline |
| post_grad | `reinplace_inplaceable_ops` | pipeline | `torch/_inductor/fx_passes/post_grad.py:464` | - | generic-pipeline |
| post_grad | `fix_auto_functionalized_dtype_views` | pipeline | `torch/_inductor/fx_passes/post_grad.py:469` | - | generic-pipeline |
| post_grad | `decompose_triton_kernel_wrapper_functional` | pipeline | `torch/_inductor/fx_passes/post_grad.py:473` | - | generic-pipeline |
| post_grad | `decompose_auto_functionalized` | pipeline | `torch/_inductor/fx_passes/post_grad.py:476` | - | generic-pipeline |
| post_grad | `decompose_scan_to_while_loop` | pipeline | `torch/_inductor/fx_passes/post_grad.py:479` | - | generic-pipeline |
| post_grad | `decompose_map_to_while_loop` | pipeline | `torch/_inductor/fx_passes/post_grad.py:482` | - | generic-pipeline |
| post_grad | `graph_pass` | pattern-registry | `torch/_inductor/fx_passes/post_grad.py:525` | - | generic-needs-validation |
| post_grad | `decompose_map_to_while_loop._` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:532` | mkldnn, dynamic_shape | needs-npu-validation |
| post_grad | `graph_pass` | pattern-registry | `torch/_inductor/fx_passes/post_grad.py:718` | - | generic-needs-validation |
| post_grad | `decompose_scan_to_while_loop._` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:725` | mkldnn, dynamic_shape | needs-npu-validation |
| post_grad | `mm_plus_mm` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1016` | - | generic-needs-validation |
| post_grad | `pointless_cumsum_replacement` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1039` | dynamic_shape, memory_alias | generic-needs-validation |
| post_grad | `cat_slice_cat` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1089` | fallback | backend-sensitive |
| post_grad | `graph_pass` | pattern-registry | `torch/_inductor/fx_passes/post_grad.py:1485` | - | generic-needs-validation |
| post_grad | `decompose_triton_kernel_wrapper_functional._` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1492` | - | generic-needs-validation |
| post_grad | `graph_pass` | pattern-registry | `torch/_inductor/fx_passes/post_grad.py:1579` | - | generic-needs-validation |
| post_grad | `decompose_auto_functionalized._` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1586` | - | generic-needs-validation |
| post_grad | `decompose_auto_functionalized._` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1613` | - | generic-needs-validation |
| post_grad | `splitwithsizes_cat_replace` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1731` | - | generic-needs-validation |
| post_grad | `cat_splitwithsizes_replace` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1786` | - | generic-needs-validation |
| post_grad | `reciprocal_sqrt_to_rsqrt` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1800` | - | generic-needs-validation |
| post_grad | `unfuse_bias_add_to_pointwise` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1905` | cuda, dynamic_shape | needs-npu-validation |
| post_grad | `unfuse_bias_baddbmm_to_pointwise` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:1955` | - | generic-needs-validation |
| post_grad | `addmm` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:2033` | - | generic-needs-validation |
| post_grad | `_fuse_addcdiv_to_fma` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:2078` | triton | backend-sensitive |
| post_grad | `reuse_partial` | pattern-entry | `torch/_inductor/fx_passes/post_grad.py:2119` | - | generic-needs-validation |
| post_grad | `fold_four_op_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:217` | - | npu-specific |
| post_grad | `fold_cast` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:265` | - | npu-specific |
| post_grad | `fold_cat` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:293` | - | npu-specific |
| post_grad | `fold_clone` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:350` | - | npu-specific |
| post_grad | `fold_detach` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:388` | - | npu-specific |
| post_grad | `fold_expand` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:407` | dynamic_shape | npu-specific |
| post_grad | `fold_reduce` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:449` | dynamic_shape | npu-specific |
| post_grad | `fold_sink_view` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:480` | dynamic_shape, memory_alias | npu-specific |
| post_grad | `fold_slice` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:586` | - | npu-specific |
| post_grad | `fold_squeeze` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:606` | - | npu-specific |
| post_grad | `fold_to_copy` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:639` | - | npu-specific |
| post_grad | `view_fold_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:703` | dynamic_shape, memory_alias | npu-specific |
| post_grad | `fold_where` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:753` | - | npu-specific |
| post_grad | `fold_redundant_ops` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:779` | dynamic_shape, memory_alias | npu-specific |
| post_grad | `cat_to_view_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:940` | dynamic_shape, memory_alias | npu-specific |
| post_grad | `repeat_to_expand_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:1172` | dynamic_shape | npu-specific |
| post_grad | `fold_iota_arithmetic_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:1433` | - | npu-specific |
| post_grad | `broadcast_const_mask_compress` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:1572` | - | npu-specific |
| post_grad | `masked_add_compose_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:1727` | - | npu-specific |
| post_grad | `bool_cast_mul_to_where_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:1871` | dynamic_shape, memory_alias | npu-specific |
| post_grad | `sign_diff_hamming_fuse_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:2005` | - | npu-specific |
| post_grad | `batch_embedding_fusion_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:2547` | dynamic_shape | npu-specific |
| post_grad | `fused_matmul_relu_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:2821` | npu, memory_alias | npu-specific |
| pre_grad | `apply_gumbel_max_trick_pass` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:48` | - | generic-needs-validation |
| pre_grad | `efficient_conv_bn_eval_pass` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:52` | - | generic-needs-validation |
| pre_grad | `fuse_split_linear_add_pass` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:56` | - | generic-needs-validation |
| pre_grad | `fuse_chunk_squeeze_cat_pass` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:59` | - | generic-needs-validation |
| pre_grad | `remove_reshape_pass` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:62` | - | generic-needs-validation |
| pre_grad | `normalization_pass_aten` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:67` | - | generic-needs-validation |
| pre_grad | `merge_splits_pass_aten` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:68` | - | generic-needs-validation |
| pre_grad | `split_cat_pass_aten` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:69` | - | generic-needs-validation |
| pre_grad | `unbind_stack_pass_aten` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:70` | - | generic-needs-validation |
| pre_grad | `merge_getitem_cat_pass_aten` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:71` | - | generic-needs-validation |
| pre_grad | `merge_stack_tahn_unbind_pass_aten` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:74` | - | generic-needs-validation |
| pre_grad | `mutate_cat_pass_aten` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:77` | - | generic-needs-validation |
| pre_grad | `remove_split_with_size_one_pass_aten` | pattern-registry | `torch/_inductor/fx_passes/pre_grad.py:78` | - | generic-needs-validation |
| pre_grad | `group_batch_fusion_passes` | pipeline | `torch/_inductor/fx_passes/pre_grad.py:379` | - | generic-pipeline |
| pre_grad | `efficient_conv_bn_eval_pass` | pipeline | `torch/_inductor/fx_passes/pre_grad.py:415` | - | generic-pipeline |
| pre_grad | `apply_gumbel_max_trick_pass` | pipeline | `torch/_inductor/fx_passes/pre_grad.py:418` | - | generic-pipeline |
| pre_grad | `pre_grad_custom_pass` | pipeline | `torch/_inductor/fx_passes/pre_grad.py:423` | - | generic-pipeline |
| pre_grad | `linear_permute_fusion` | pipeline | `torch/_inductor/fx_passes/pre_grad.py:466` | - | generic-pipeline |
| pre_grad | `permute_linear_fusion` | pipeline | `torch/_inductor/fx_passes/pre_grad.py:468` | - | generic-pipeline |
| pre_grad | `permute_matmul_fusion` | pipeline | `torch/_inductor/fx_passes/pre_grad.py:470` | - | generic-pipeline |
| pre_grad | `remove_identity` | pipeline | `torch/_inductor/fx_passes/pre_grad.py:477` | - | generic-pipeline |
| pre_grad | `fuse_conv_bn` | pipeline | `torch/_inductor/fx_passes/pre_grad.py:479` | - | generic-pipeline |
| pre_grad | `cat_slice_cat_fold_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:65` | dynamic_shape | npu-specific |
| pre_grad | `pad_slice_fold` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:157` | - | npu-specific |
| pre_grad | `dtype_optimal_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:844` | dynamic_shape | npu-specific |
| pre_grad | `fusion_attention_v3_pass` | npu-custom-pass | `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py:916` | npu | npu-specific |
