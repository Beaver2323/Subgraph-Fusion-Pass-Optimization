# T-074 社区原生 pass/pattern 与上游测试索引（2026-08-29）

> 原始生成时间：2026-08-29
> 术语复核时间：2026-08-31 17:50 CST（UTC+08:00）
> 状态：T-074 v1 静态 inventory/provisional mapping；不是冻结的 Pass 或 acceptance-unit 分母。

## 结论

已完成第一版静态索引生成与首批人工回填：T-056 的 203 条主候选全部进入索引，
另纳入 3 条 pad 和 1 条 addmm 显式关闭的上游控制项，因此候选表共 207 行。
本轮没有导入 `torch`，没有运行 NPU、`torch.compile` 或性能测试，也没有修改
PyTorch、torch_npu、Triton、环境或 wheel。

207 行当前聚合为 188 个去重验收单元。其中 158 个标为
`yes-provisional`，30 个注册容器/扩展 hook 标为不进入分母。这个 158 **不是冻结分母**：
除首批 5 个单元外，其余自动映射仍需逐项人工审阅，不能据此宣称完成率。

## 2026-08-31 术语与工作流复核

- 203 条主候选和 4 条控制项是 registration/inventory candidate 行，不称为 207 个 Pass；
- 188 个单元来自 source/name 归一化和少量显式 semantic group，是 heuristic provisional
  grouping，不自动等于 188 个 upstream optimization contracts；
- community test 是 contract 的主要事实源，registration/pattern evidence 用于辅助 coverage；
- 一个 registration 可能展开多个需分别验证的 variants，多个 registration 也可能共同实现
  一个 acceptance unit；
- 当前 candidate CSV 可以继续复用，不需要因术语变化立即重扫；
- 先冻结 acceptance-unit schema 并人工复核 mapping，发现合并/拆分后再生成 v2，保留 v1；
- 当前数量不变：207 candidate rows、188 provisional units、158 provisional eligible、
  5 个首批人工 mapping、正式 tracker 闭环 0。

## 输入边界与索引修正

| 集合 | 行数 | 说明 |
| --- | ---: | --- |
| `primary-203` | 203 | `inherited-upstream-needs-dynamic-validation` 主候选 |
| `explicitly-disabled-control-4` | 4 | pad mm/bmm/addmm 与 add+mm -> addmm |
| 总计 | 207 | 两个集合分开统计 |

T-056 的 `record_id` 不是源码行级唯一键：203 条主候选只有 197 个唯一 ID。
T-074 保留历史 `record_id`，另生成包含 source/line/name 摘要的唯一 `candidate_id`。
重复项如下：

| record_id | 重复行 |
| --- | --- |
| `9563f69c084d` | torch/_inductor/fx_passes/quantization.py:1208 `woq_int8`; torch/_inductor/fx_passes/quantization.py:1273 `woq_int8` |
| `ea5af77d58bb` | torch/_inductor/fx_passes/post_grad.py:525 `graph_pass`; torch/_inductor/fx_passes/post_grad.py:718 `graph_pass`; torch/_inductor/fx_passes/post_grad.py:1485 `graph_pass`; torch/_inductor/fx_passes/post_grad.py:1579 `graph_pass` |
| `ebd427f76b31` | torch/_inductor/fx_passes/post_grad.py:1586 `decompose_auto_functionalized._`; torch/_inductor/fx_passes/post_grad.py:1613 `decompose_auto_functionalized._` |
| `fbeedced3149` | torch/_inductor/fx_passes/post_grad.py:209 `post_grad_custom_pre_pass`; torch/_inductor/fx_passes/post_grad.py:242 `post_grad_custom_pre_pass` |

## 初始测试覆盖映射

下表是静态发现结果，不是 NPU verdict。`direct` 来自明确符号引用、测试名契约或
人工映射；`indirect` 只表示更大回归用例可追踪到相关入口；`no-test-found` 表示当前
冻结源码中尚未建立可追溯测试，不表示 NPU 不支持。自动生成项均带
`generated-needs-human-review`。

| 口径 | direct-trigger-test | indirect-regression-test | no-test-found |
| --- | ---: | ---: | ---: |
| 207 条候选/控制行 | 120 | 33 | 54 |
| 188 个去重单元 | 111 | 29 | 48 |
| 158 个 provisional 分母单元 | 99 | 19 | 40 |

### 按源码文件分布

| source | rows | direct | indirect | no-test-found |
| --- | ---: | ---: | ---: | ---: |
| `torch/_inductor/fx_passes/apply_gumbel_max_trick.py` | 1 | 1 | 0 | 0 |
| `torch/_inductor/fx_passes/b2b_gemm.py` | 2 | 2 | 0 | 0 |
| `torch/_inductor/fx_passes/binary_folding.py` | 1 | 0 | 0 | 1 |
| `torch/_inductor/fx_passes/decompose_mem_bound_mm.py` | 3 | 3 | 0 | 0 |
| `torch/_inductor/fx_passes/efficient_conv_bn_eval.py` | 3 | 3 | 0 | 0 |
| `torch/_inductor/fx_passes/freezing_patterns.py` | 7 | 2 | 2 | 3 |
| `torch/_inductor/fx_passes/fsdp.py` | 3 | 0 | 1 | 2 |
| `torch/_inductor/fx_passes/fuse_attention.py` | 30 | 30 | 0 | 0 |
| `torch/_inductor/fx_passes/group_batch_fusion.py` | 1 | 1 | 0 | 0 |
| `torch/_inductor/fx_passes/joint_graph.py` | 15 | 11 | 1 | 3 |
| `torch/_inductor/fx_passes/micro_pipeline_tp.py` | 2 | 2 | 0 | 0 |
| `torch/_inductor/fx_passes/misc_patterns.py` | 5 | 1 | 0 | 4 |
| `torch/_inductor/fx_passes/mkldnn_fusion.py` | 12 | 4 | 3 | 5 |
| `torch/_inductor/fx_passes/pad_mm.py` | 3 | 3 | 0 | 0 |
| `torch/_inductor/fx_passes/post_grad.py` | 47 | 24 | 13 | 10 |
| `torch/_inductor/fx_passes/pre_grad.py` | 22 | 9 | 10 | 3 |
| `torch/_inductor/fx_passes/quantization.py` | 9 | 1 | 2 | 6 |
| `torch/_inductor/fx_passes/reduced_atomic_contention.py` | 4 | 1 | 0 | 3 |
| `torch/_inductor/fx_passes/replace_random.py` | 8 | 2 | 1 | 5 |
| `torch/_inductor/fx_passes/split_cat.py` | 29 | 20 | 0 | 9 |

## 首批 5 个已人工回填单元

| pass/pattern | set | upstream test | 历史证据边界 |
| --- | --- | --- | --- |
| `addmm_pattern` | `explicitly-disabled-control-4` | test/inductor/test_pad_mm.py::PadMMTest.test_pad_addmm_dyn_m<br>test/inductor/test_pad_mm.py::PadMMTest.test_addmm_beta_zero_mismatched_bias_skips_padding | `historical-performance-rejected-await-community-port` |
| `mm_pattern` | `explicitly-disabled-control-4` | test/inductor/test_pad_mm.py::PadMMTest.test_pad_mm_dyn_m<br>test/inductor/test_pad_mm.py::PadMMTest.test_exclude_padding<br>test/inductor/test_pad_mm.py::PadMMTest.test_original_aten_preserved_pad_mm | `historical-performance-rejected-await-community-port` |
| `bmm_pattern` | `explicitly-disabled-control-4` | test/inductor/test_pad_mm.py::PadMMTest.test_pad_bmm_dyn_b<br>test/inductor/test_pad_mm.py::PadMMTest.test_no_autocast_in_pad_bmm_joint_graph_pass | `historical-performance-rejected-await-community-port` |
| `mm_plus_mm` | `primary-203` | test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_mm_plus_mm | `historical-complete-await-current-Pass-community-port` |
| `addmm` | `explicitly-disabled-control-4` | test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_addmm<br>test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_addmm_symbolic_scalar | `historical-isolated-wheel-beneficial-await-current-Pass-port` |

首批映射的结论边界：

- `mm_plus_mm` 已对应社区 `test_mm_plus_mm` 的同 K/different K 正负触发合同，
  旧 Pass 动态证据可继续复用，但仍需在当前 installed baseline 上最小复核。
- pad mm/bmm/addmm 已对应 `test_pad_mm.py` 的正例、跳过/排除和 counter/codegen
  合同；旧性能否决保持历史有效，不等于当前社区用例已经在 Pass 环境迁移。
- add+mm -> addmm 已对应 `test_addmm` 与 symbolic-scalar 负例；P-018 的收益来自
  独立 wheel，尚不能升级为当前 Pass 产品已开启。

## 产物

- `t074_build_upstream_test_index.py`：标准库静态生成器，不导入 `torch`。
- `upstream_pass_test_index_20260829/candidate_test_index.csv`：207 行源码
  候选/控制项映射，包含注册、config、社区测试、device、NPU 状态和证据边界。
- `upstream_pass_test_index_20260829/acceptance_units.csv`：188 个 provisional 单元及
  provisional 分母角色。

## 下一步

1. 执行 T-075：先定义 acceptance-unit manifest/schema，再复核首批 5 个单元的 contract、
   variants、tracking mode 和 denominator 状态。
2. 按单元审核 48 个 `no-test-found` 和 29 个 indirect，同时保留其对应的 54/33 条 candidate
   行；无法确认时标记 `needs-review`，不得猜测。
3. mapping 收敛后生成 manifest-driven GPU/reference runner；GPU baseline 有效后才进行 NPU
   `triton_experimental` comparison。
4. NPU 动态测试继续遵守 Pass 环境、fresh process、wheel 身份和 correctness-before-performance
   门禁；当前所有动态字段保持 `not-run-current-Pass`。
