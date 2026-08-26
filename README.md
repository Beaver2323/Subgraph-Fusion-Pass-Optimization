# Inductor Pass NPU Audit

This directory is the audit, evidence, and report area for the local PyTorch
and torch_npu source trees. The project working directory is
`/home/z50063656/Pass`. T-011, T-023, T-029/T-031, T-036, T-038, T-040, T-042,
T-043, T-046, T-047/P-012, T-054/P-013, and the current-source-only
P-014/P-016/P-017/P-018 contain pre-registered torch_npu source changes; all other
audit prototypes stay in this directory, and every product
change is governed by `change_control.md`.

## Project state

The P-013/T-054 default-backend work remains archived in
[PAUSED_CHECKPOINT_20260826_P013.md](PAUSED_CHECKPOINT_20260826_P013.md). The
user has resumed the task with `triton_experimental` as the target backend while
keeping the existing Benchmark runtime unchanged. Read
[triton_experimental_migration_20260826.md](triton_experimental_migration_20260826.md)
before running another test or proposing a source change. The verified P-013
wheel and its P-012 rollback wheel remain preserved under `artifacts/`.

T-055 now records 12/12 successful experimental entry cases and one real
experimental-to-default decomposition reload failure. P-014's one-entry erfc
cleanup passes source re-entry and bidirectional backend-switch validation, but
the installed wheel is intentionally unchanged while an unrelated, uninstalled
shared `triton.py` diff is present. See
[report/t055_triton_experimental_enable_20260826.md](report/t055_triton_experimental_enable_20260826.md).
T-056's independent routing/config/feature inventory is now available at
[report/t056_triton_experimental_inventory_20260826.md](report/t056_triton_experimental_inventory_20260826.md);
its CSV artifacts live under `report/triton_experimental_20260826/`. T-057 then
proved backend state leakage and an unsafe default int-float-int elision. The
Float32 ON/OFF pair isolates a one-integer numerical error, and all three tested
float dtypes expose output alias changes. P-016 now disables that pass by default
in source while preserving explicit opt-in; source gating passes, but the installed
wheel remains unchanged because the shared tree contains unrelated uninstalled
changes. See
[report/t057_backend_state_isolation_20260826.md](report/t057_backend_state_isolation_20260826.md)
and
[report/t057_int_float_int_boundary_20260826.md](report/t057_int_float_int_boundary_20260826.md).
T-057 also found that the installed experimental GELU override ignores
`approximate="none"`; P-017 restores the upstream erf/CDF/PDF contract in current
source. Its source contract probe and six FP32/FP16/BF16 NPU source-overlay cases
pass, with `none` generating `libdevice.erf` and `tanh` retaining `tl.sigmoid`.
The installed wheel remains P-013 while unrelated shared-tree changes are present.
See
[report/t057_gelu_approximate_20260826.md](report/t057_gelu_approximate_20260826.md).
T-058 has now completed the first experimental addmm gate cohort. Restoring the
two pristine upstream checks in a fresh audit worker changes `mm + Triton add`
into one extern addmm; three interleaved rounds improve median p50 by 22.17% and
p99 by 14.05% with unchanged allocated peak. Eleven follow-up capability cases
and a second unaligned performance cohort also pass. P-018 now enables fusion in
current source with a reversible explicit opt-out; its source gate passes, while
the installed wheel and host-tail retest remain pending.
See
[report/t058_experimental_addmm_gate_20260826.md](report/t058_experimental_addmm_gate_20260826.md).
T-059 has completed the installed experimental permute-gather audit. The T5-like
representative shape improves median device p50/p99 by 8.14%/8.99%, and six
BF16/FP32/dynamic/guard cases pass. The tradeoff is one extra copy kernel,
1,560,576 B more peak allocation, higher forced-fresh compile time, and one
5.00725 ms host p99 outlier. The default remains enabled with host-tail, memory,
and clean-launcher monitoring; no source or handwritten Triton change was needed.
See
[report/t059_experimental_permute_gather_20260826.md](report/t059_experimental_permute_gather_20260826.md).
T-060 has now isolated `rsplit_outer` reachability. Native scalar sum reaches
the existing two-kernel path, while representative RMSNorm dweight and outer
reductions receive `ReductionHint.DEFAULT` and are rejected by the
INNER/OUTER-only gate. After document-first validation, P-019 now locally
accepts DEFAULT in the existing gate. The existing partial+combine codegen is
correct and the real source overlay improves median device p50/p99 by
29.93%/29.94% across three rounds, at a 393,728 B peak-allocation and
forced-fresh compile-time cost. The current target suite is 6/6: P-019 covers
the DEFAULT positive plus small-r, non-sum, wide-x, OUTER_TINY, fused-output,
and nested guards, while T-061 adds downcast-memo invalidation; source-overlay
static/dynamic and small-r device checks also pass. The wheel remains
unbuilt/uninstalled, and no handwritten Triton replacement is needed. See
[report/t060_experimental_rsplit_outer_20260827.md](report/t060_experimental_rsplit_outer_20260827.md).
T-061 then found a correctness boundary in int64 lowering: in-range pointwise
and in-place cases are exact, but a four-value boundary cohort fails 4096/4096
with ±2^32 wraparound. An audit-only ATen mul fallback restores exact results;
P-020 now tracks a dtype-aware fallback design, and a handwritten Triton int64
replacement is rejected because the target vector compute path lacks int64.
See [report/t061_experimental_int64_boundary_20260827.md](report/t061_experimental_int64_boundary_20260827.md).

This task currently uses the user-specified shared Benchmark runtime; it does
not yet have a separate environment. Source commits, wheel identity, device
isolation, and the audit-only launcher workaround are recorded explicitly.
Static inventory and the first P0 dynamic/function/performance checkpoint are complete:

- Work resumed from the user-requested 2026-08-21 checkpoint. The preserved
  baseline is [PAUSED_CHECKPOINT_20260821.md](PAUSED_CHECKPOINT_20260821.md),
  and the resumed evidence starts at `change_control.md:E-030`. T-014 through
  T-026 have now closed standalone correctness, dtype/layout/dynamic/backward
  semantics, default-off wheel integration, fresh-process paired performance,
  memory root cause, workspace/tile alternatives, and the pad family capability
  and performance audit. The different-K result is conditionally beneficial but
  stays default-off; pad mm/bmm/addmm are functionally executable after a
  test-only device-gate bypass but regress 65–121% at p50, so the gate remains.
  The next main batch is P1; the matching fresh-launcher environment remains an
  explicit T-023 follow-up rather than being hidden by the audit compiler shim.
  P1 B2 is now complete: T-027 through T-046 cover 85/85 FX tests and fresh NPU
  positive/negative/alias/IEEE cases for all twenty-seven custom passes. Alias auditing rejected the
  original direct-input `fold_reduce`/`cat_to_view` rewrites. The final wheel
  disables the performance-regressed `fold_reduce` rewrite and keeps an
  alias-safe `cat_to_view` clone that reduces tasks 3→1 and allocated peak by
  4,195,840 B while remaining latency-neutral. The second cohort identifies
  `fold_cast`/`fold_clone`/`fold_detach` as partial-reachability cases and closes
  `fold_cat` as `supported-beneficial`: p50/p99 improve 10.14%/10.32%, tasks
  drop 2→1, and allocated peak falls by 2,097,664 B. The third cohort verifies
  four more reached transforms and classifies `fold_to_copy` as
  reachability-neutral; `fold_where` is functionally sound but performance
  neutral (p50/p99 +1.16%/+3.12%, tasks and peak memory unchanged). The fourth
  cohort found zero-numeric-error alias defects in `cat_slice_cat_fold_pass` and
  `pad_slice_fold`; conservative source guards now preserve observable storage/
  stride semantics while retaining safe positive rewrites. The rebuilt wheel,
  60/60 FX tests, and 6/6 NPU variants pass. T-037 then closes both as
  `supported-beneficial`: p50 improves 24.00%/31.35%, tasks fall 2→1/3→1,
  and pad-slice allocated peak falls by 10,485,248 B. T-038 then narrows three
  dtype/index/mask rewrites after finding observable dtype, Inf/overflow, and
  broadcast-shape counterexamples. The rebuilt wheel passes 67/67 source and
  installed FX tests plus 9/9 NPU functional workers. T-039 closes their paired
  performance: safe `dtype_optimal_pass` and iota downcast improve p50 by
  52.06%/55.78%; equal-shape mask compression improves only 0.30% and remains
  `supported-neutral`. T-040 then found float signed-zero and `0*Inf/NaN`
  counterexamples in two remaining mask arithmetic passes, restricted both to
  exact-zero integer/bool dtypes, rebuilt the source wheel, and closed 76/76 FX
  plus 9/9 NPU functional workers. T-041 then closes 24/24 paired workers:
  masked-add and sign-Hamming are neutral, bool-cast view-chain improves
  p50/p99 by 36.30%/39.90%, while its direct rewrite regresses p99 by 19.02%.
  T-042 adds a non-empty view-chain guard, rebuilds/installs the wheel, and
  passes 76/76 source/installed FX plus 3/3 NPU functional workers. The
  direct/float paths now stay unchanged and the view-chain scope is beneficial.
  T-043 through T-046 then repair batch-embedding slice/dtype/storage semantics,
  attention schema/meta handling, and duplicate PRE execution. Batch fusion improves
  P50 by 23.50%/43.90% and reduces tasks 9→3/13→3, but increases allocated peak and
  first compile, so it is resource-beneficial rather than globally beneficial.
  Legacy→v3 attention on 910B2 regresses P50/P99 by 4.85%/31.72% with the same
  single vendor kernel; the final wheel disables that replacement on non-A5 devices.
  Fused matmul+relu remains A5-only and is correctly not applicable on the available 910B2.
  T-047/T-048 now close the eight B3 DVM/MLIR records. P-012 makes the DVM
  sum/expand subpasses self-contained and the rebuilt wheel passes 6/6 source
  and installed contract tests, 15/15 DVM graph-fusion NPU tests, and 32/32
  DVM-backend tests. Aggregate DVM fusion improves P50/P99 by 24.20%/39.93%
  with equal allocated peak and much shorter first compile, so it is
  `supported-beneficial`. K=1 is already decomposed upstream; expand remains
  unreachable through the current partition capability; full MLIR remains
  blocked by the missing `torch_mlir` dependency.
  T-049/T-052 verify exact matcher/numerics/codegen for eight representative
  B4 attention families. Pattern 1/13 reach vendor attention directly, 18/28
  retain auxiliary Triton kernels, and 5/21/29 re-expand to BMM + Triton.
  T-050 closes pattern 1's static fp16 inference scope as
  `supported-beneficial`: P50/P99 improve 46.70%/44.26%, tasks fall 4→1, and
  additional allocated peak falls 87.31%. No product source changed in B4.
  T-051 then closes pattern 13 as `supported-neutral-resource-beneficial`:
  P50 improves only 0.99%, while tasks fall 3→1, first compile improves
  91.37%, and additional allocated peak falls 87.31%.
  T-052 traces 5/21/29 re-expansion to the vendor bool/None mask gate and
  confirms no-mask pattern 30 reaches one vendor attention task.
  T-053/T-054 close pattern 5: the rewrite regresses P50 by 103.23% and expands
  tasks 3→8; the P-013 exact NPU guard restores the original graph, improves
  P50 by 50.28%, and returns tasks to 3. Pattern 1/13/21 regressions pass.

- Do not modify PyTorch, torch_npu, or Triton functional source before the
  exact proposal, rollback boundary, and verification plan are recorded in
  `change_control.md`. T-011 is the first approved and verified exception.
- Do not treat static evidence or skipped probes as NPU availability/performance results.
- Record every proposed source or Triton change in `change_control.md` before implementation.
- Launch every test from `/home/z50063656/tmp`; never import torch while the
  current directory is inside a torch_npu source tree.

The current dynamic runtime is Conda `benchmark-py311`, activated by
`/home/z50063656/Benchmark/env.sh`: Python 3.11.15, PyTorch
`2.14.0a0+git8e86e0a` as an editable source install from
`/home/z50063656/Benchmark/pytorch-upstream`, source-built torch_npu
`2.14.0a0+git83cc452` installed from the local `dist` wheel with `--no-deps`,
Triton runtime 3.2.0, CANN 9.0.1, and 8 Ascend 910B2 devices. Runtime NPU tests
pass, but T-022 found that a fresh Triton host launcher cannot be compiled by
this exact mixed header contract without an audit-only shim; product validation
must not rely on that shim. The installed T-046 source wheel SHA256 is
`beee993d4c803ed72d26284dcdc06eac97cedaf450a54398ec11285d2711d54b`;
the previous P-010 wheel remains preserved as
`artifacts/torch_npu_t043_before_t046_attention_b2_gate.whl` for rollback. The
current P-013 wheel SHA256 is
`3909fd649d777b8dfd393342da0ff2b88c5cce2ef219f0d103d063af4c2d4989`;
the P-012 rollback wheel is `artifacts/torch_npu_t053_before_p013.whl`;
the P-011 rollback wheel is
`artifacts/torch_npu_t046_before_t047_p012.whl`.

Read [current_status_and_background.md](current_status_and_background.md) before
choosing a wheel or source build and before restarting any probe.
For a one-page separation of successful, rejected, neutral, and environment
attempts, read [outcome_index.md](outcome_index.md).

For a source-backed, from-scratch learning path, start with
[inductor_pass_npu_beginner_guide.md](inductor_pass_npu_beginner_guide.md). It
connects the compile pipeline, PyTorch/torch_npu source entry points, P0
evidence, matrix workflow, and the next implementation stages.

For the exact project scope and source edit map, read
[task_scope_and_code_map.md](task_scope_and_code_map.md). The current source
inventory is under `report/pass_src_20260820/`; the root-level report is the
older `/Dynamo` snapshot and is retained for comparison.

P0 gate cases are specified in [p0_case_design.md](p0_case_design.md). The next
66 P1 records (NPU custom, DVM/MLIR, and attention) are routed in
[p1_batch_design.md](p1_batch_design.md).

P0 current behavior has been executed for 20 positive/negative backend pairs;
all compile and compare correctly. Three pad families remain product-unsupported
because upstream `pad_mm.py` admits only CUDA/XPU; T-025/T-026 proved their
static fp16 graphs are functionally executable after a test-only bypass, but
their p50 regresses 72.65%/65.31%/120.63%. The representative addmm and
`mm_plus_mm` function matrix is 16/16 correct. In paired performance runs,
default addmm exceeds the 10% p50 threshold in all 8 representative configs;
experimental `mm_plus_mm` exceeds it in 6/8, while transposed and dynamic are
supported but performance-neutral. The semantic matrix confirms addmm bias and
dtype guards, mm_plus_mm shape guards, and isolated backward gradients. T-011
fixed the torch_npu `strict_sum` lowering interface, rebuilt and installed the
source wheel with `--no-deps`, and closed vector/full-bias backward plus neighbor
regressions. Addmm is now `supported-beneficial`; T-023 has moved different-K
`mm_plus_mm` to `conditional-supported-beneficial` through a default-off NPU
template with extern fallback. See
[p0_sweep_function_matrix_20260820.md](report/p0_sweep_function_matrix_20260820.md)
and
[p0_sweep_performance_20260820.md](report/p0_sweep_performance_20260820.md), then
[p0_semantic_matrix_20260821.md](report/p0_semantic_matrix_20260821.md).
The different-K pass-on/pass-off baseline is in
[t012_mmplus_different_k_baseline_20260821.md](report/t012_mmplus_different_k_baseline_20260821.md):
shape-A is -0.30% and unaligned is +2.53% at p50, so the current fallback is
performance-neutral. The follow-up kernel breakdown is in
[t013_mmplus_different_k_profile_20260821.md](report/t013_mmplus_different_k_profile_20260821.md):
two in-step gaps plus add leave a 17.68%/16.02% theoretical ceiling, which
allows a standalone microprototype but not a source integration. T-014 through
T-019 show that the 128³ candidate is one-task and covers fp16/bf16/fp32,
true transposed stride, dynamic replay, and backward semantics. T-020 paired
benchmarking records p50 improvements of 11.20% for bf16, 15.25% for fp32,
and 12.70% for fp16 transposed; the earlier fp16 contiguous points improve
15.60%/17.12%. The isolated large rerun improves median p50 by 11.58%, but is
held because of one-round regression, high variance/tails, and candidate
additional peak growth to 5.90 MB. T-021 then selected a torch_npu-owned,
NPU-only, default-off template choice that always retains the current extern
fallback. T-022 re-profiled three large tiles: device p50 is
29.27/24.33/14.41 μs, but stable paired p50 gains are only
6.64%/6.97%/7.55%, so large is now `supported-neutral-hold`. Its steady memory
decomposition shows candidate allocated peak exceeds baseline by one 655,360 B
logical output; the older 5.90 MB value included long-lived first outputs. See the
[T-014–T-016 report](report/t014_t016_mmplus_different_k_candidate_20260821.md),
[T-017–T-019 coverage report](report/t017_t019_mmplus_different_k_coverage_20260821.md),
the [T-020 extended benchmark](report/t020_mmplus_different_k_extended_benchmark_20260821.md),
the [T-021 integration design](report/t021_mmplus_different_k_integration_design_20260821.md),
the [T-022 large decomposition](report/t022_mmplus_different_k_large_profile_20260821.md),
the [T-023 integration report](report/t023_mmplus_different_k_integration_20260821.md),
the [T-024 workspace audit](report/t024_mmplus_different_k_workspace_20260821.md),
and the [T-025/T-026 pad audit](report/t025_t026_pad_family_20260821.md).
T-023 shape-A/unaligned integrated p50 improves 15.29%/18.04%, but candidate
peak allocated is 270,336 B above baseline because Triton Ascend allocates
65,536 B workspace for each of six blocks. T-024 found no configuration that
passes both the strict memory and task-duration gates. The template therefore
remains default-off with a 131072-element output cap; the formal matrix verdict
is conditional until a matching no-shim launcher environment is verified.

The first P1 B2 checkpoint is in
[T-028](report/t028_p1_b2_npu_compile_20260821.md) and the
[T-029/T-030/T-031 closure report](report/t029_t030_b2_alias_fix_performance_20260824.md).
The important distinction is explicit: `fold_reduce -> clone` was a correct but
performance-regressed intermediate attempt; `cat_to_view -> clone` is retained
as `supported-neutral-resource-beneficial`. Three other positives were removed
before their target pass and remain reachability-neutral, not attributed wins.
The second cohort is in the
[T-032 compile report](report/t032_b2_redundancy_compile_20260824.md) and
[T-033 fold_cat performance report](report/t033_fold_cat_performance_20260824.md).
The third cohort is in the
[T-034 compile report](report/t034_b2_view_copy_compile_20260824.md) and
[T-035 fold_where performance report](report/t035_fold_where_performance_20260824.md).
The fourth cohort's alias defects, conservative source fix, rebuilt wheel, and
functional closure are in the
[T-036 layout alias report](report/t036_b2_layout_alias_fix_20260825.md); its
three-round paired performance is in the
[T-037 layout performance report](report/t037_layout_pass_performance_20260825.md).
The fifth cohort's dtype/index/mask semantic boundary and rebuilt-wheel closure
are in the
[T-038 semantic report](report/t038_dtype_index_mask_semantic_fix_20260825.md);
its three-round paired performance and two-beneficial/one-neutral decision are
in the
[T-039 performance report](report/t039_dtype_index_mask_performance_20260825.md).
The remaining mask arithmetic IEEE fix, rebuilt-wheel verification, and 9/9
NPU functional closure are in the
[T-040 report](report/t040_mask_hamming_semantic_fix_20260825.md).
Their four-case paired performance is in the
[T-041 report](report/t041_mask_hamming_performance_20260825.md), and the final
bool view-chain capability/wheel closure is in the
[T-042 report](report/t042_bool_view_guard_integration_20260825.md). The final
batch-embedding, fused-matmul-relu, and attention-v3 cohort—including the B2
performance rejection gate—is in the
[T-043–T-046 report](report/t043_t046_b2_composite_passes_20260825.md).
The B3 DVM/MLIR source map, P-012 fix, 47 NPU tests, reachability findings,
and paired aggregate performance are in the
[T-047/T-048 report](report/t047_t048_b3_dvm_mlir_20260826.md).
The B4 scale contract, seven representative pattern/codegen smokes, invalid
isolation attempts, and first attention paired result are in the
[T-049/T-050 report](report/t049_t050_b4_attention_first_20260826.md).
Pattern 13's paired latency/resource split is in the
[T-051 report](report/t051_b4_attention_pattern13_performance_20260826.md).
Pattern 5's math re-expansion regression and verified P-013 guard are in the
[T-053 report](report/t053_b4_attention_pattern5_performance_20260826.md) and
[T-054 report](report/t054_b4_attention_pattern5_guard_20260826.md).
The float-mask dispatcher root cause and pattern-30 control are in the
[T-052 report](report/t052_b4_attention_float_mask_dispatch_20260826.md).

## 1. Generate the full source inventory

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/inductor_pass_npu_audit/audit_passes.py \
  --pytorch-root /home/z50063656/Pass/src/pytorch \
  --torch-npu-root /home/z50063656/Pass/src/torch_npu \
  --output /home/z50063656/Pass/inductor_pass_npu_audit/report/pass_src_20260820
```

The JSON contains one record per pipeline observer, pattern entry, registered
NPU custom pass, and scheduler/extension hook. The Markdown report is a human
review index. Static status values are routing hints only.

## 2. Validate P0 gate behavior

The P0 probe is the first dynamic entry point. Its orchestrator never imports
torch; each case/backend pair runs in a fresh worker:

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/inductor_pass_npu_audit/run_p0_gate_probe.py \
  --output /home/z50063656/Pass/inductor_pass_npu_audit/report/pass_src_20260820/p0_probe \
  --backends default,triton_experimental --debug
```

Add `--benchmark --warmup 10 --runs 100` only after correctness and generated
graph evidence are healthy. For a single-pass A/B result, also use the
family-specific `--target-pass-modes current,disabled` control in fresh
workers; ordinary current-only timings remain diagnostic.

For the P0 coverage expansion, select only the target family and request an
explicit sweep. The following command creates fresh current/disabled workers,
alternates their order across three rounds, and performs a real dynamic replay:

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/inductor_pass_npu_audit/run_p0_gate_probe.py \
  --output /home/z50063656/Pass/inductor_pass_npu_audit/results/p0_sweep_example \
  --cases mm_plus_mm_positive \
  --backends triton_experimental \
  --dtypes float16 \
  --shape-profiles shape_a \
  --layouts contiguous \
  --dynamic-modes dynamic \
  --target-pass-modes current,disabled \
  --rounds 3 --benchmark --warmup 10 --runs 100
```

The full parameter contract and the non-Cartesian cohort plan are documented
in [p0_case_design.md](p0_case_design.md#p0-覆盖扩展参数).

## 3. Run broader representative probes

Run only from `/home/z50063656/tmp` and use a fresh process per backend:

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/inductor_pass_npu_audit/run_npu_probe.py \
  --output /home/z50063656/Pass/inductor_pass_npu_audit/report \
  --backends triton,ascendc,mlir --warmup 10 --runs 100
```

The probe records compile/first-run latency, steady-state mean/stdev/p50/p99,
peak allocated NPU memory, eager-vs-compiled correctness, and the pass names
observed by `GraphTransformObserver` and `PatternMatcherPass`. A missing NPU is
recorded as `skip`; it is never reported as a pass.

`run_npu_probe.py` is an earlier broad draft. Its backend list and cases must be
reconciled with the P0 evidence before it becomes the full-matrix runner.

## 4. Acceptance rules

For every pass/case/backend pair:

1. `available`: compilation succeeds and outputs pass dtype-specific eager comparison.
2. `fast`: the same fresh baseline/candidate run records CANN version, SoC,
   warmup, sample count, mean, stdev, p50, p99, and peak memory. Use the
   project's paired baseline rule; do not compare historical logs.
3. `fallback`: report the exact fallback operator and generated-code evidence.
4. `unsupported`: report the exception and smallest reproducer.

Handwritten Triton should be considered only after the generated graph shows a
real backend gap or a measured kernel-count/latency problem. For each candidate,
keep a reference implementation, add a correctness test, and compare against
the native NPU op/AscendC/CATLASS implementation before enabling a pass.

## Historical baseline limitation

The earlier probe, run outside `Benchmark/env.sh`, saw `torch 2.14.0.dev20260805+cpu`,
`torch_npu 2.14.0+git06101a0`, no visible NPU, and no installed Triton module.
Those values are historical diagnostics only and are not a baseline for the
formal Conda `Pass` environment.
