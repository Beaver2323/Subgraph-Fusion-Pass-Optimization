# Inductor Pass NPU 调研任务与代码地图

## 任务最终要交付什么

本任务的目标不是简单统计源码中的 pass，也不是把每个 pass 都改成 Triton。最终交付应是一张覆盖 NPU 的可审计矩阵：每个 pass 都有适用范围、最小触发图、是否实际触发、正确性、fallback/codegen 证据、性能数据和最终处理结论。

最终判定分为：

| 判定 | 含义 |
|---|---|
| `not-applicable` | CPU/MKLDNN/CUDA 专用，NPU 不触发是正确行为 |
| `environment-blocked` | 尚未进入 pass，工具链或运行环境阻塞 |
| `unsupported` | pass 已触发，但替换图/lowering/codegen 在 NPU 失败 |
| `supported-neutral` | 可用，但没有稳定性能收益 |
| `supported-beneficial` | 可用且有可重复的稳态或端到端收益 |
| `conditional-supported-beneficial` | 可用且有收益，但环境、显存或能力边界尚不允许默认启用 |
| `supported-regression` | 可用但性能、内存或编译开销明显回退 |

对于 `unsupported` 和 `supported-regression`，再形成替代实现提案。提案应先比较原图、vendor op、CATLASS、AscendC 和 Triton，不默认选择手写 Triton。

## Inductor pass 位于编译链的哪里

```text
Python 模型
  -> TorchDynamo 捕获 FX 图
  -> AOTAutograd / functionalization
  -> pre_grad passes
  -> joint_graph passes
  -> post_grad passes
  -> lowering 到 Inductor IR
  -> scheduler / fusion / codegen
  -> NPU kernel 与 runtime
```

因此一个失败需要分层定位：

1. pattern 是否注册并匹配；
2. NPU capability gate 是否允许；
3. 替换后的 op 是否有 NPU lowering；
4. scheduler/codegen 是否能承接；
5. 最终是否走 NPU kernel，而不是 graph break 或 CPU fallback；
6. 编译成本、kernel 数、稳态延迟和内存是否改善。

## 主要源码入口

### PyTorch 通用 pass

| 层级 | 入口 | 主要职责 |
|---|---|---|
| pre-grad | `src/pytorch/torch/_inductor/fx_passes/pre_grad.py:336` | 在未完全 functionalize 的 Torch IR 上做规范化和融合 |
| joint graph | `src/pytorch/torch/_inductor/fx_passes/joint_graph.py:699` | 在联合前后向图上做 pattern、常量折叠和随机算子处理 |
| post-grad | `src/pytorch/torch/_inductor/fx_passes/post_grad.py:177` | functionalized 图上的主要 pattern/fusion/通信与重排 pass |
| 配置入口 | `src/pytorch/torch/_inductor/config.py` | pass 开关、自定义 pre/post pass、shape padding 等配置 |
| pattern 基础设施 | `src/pytorch/torch/_inductor/pattern_matcher.py` | pattern 注册、匹配和计数 |

新增通用语义优化且对多个后端都成立时，优先修改 PyTorch；只为 NPU 放宽/限制能力时，不应直接污染通用 pattern，应在 torch_npu 侧做 capability gate 或后端 lowering。

### torch_npu 承接层

| 层级 | 入口 | 主要职责 |
|---|---|---|
| backend loader | `src/torch_npu/torch_npu/_inductor/__init__.py` | 按 `default/ascendc/mlir/dvm/triton_experimental` 加载 patch、lowering、pass 和 codegen |
| FX pass | `src/torch_npu/torch_npu/_inductor/fx_passes/` | NPU 自定义 pass、上游 pattern gate、attention 和并行调度 |
| lowering | `src/torch_npu/torch_npu/_inductor/lowering.py` 及相关列表 | 把 Aten/Inductor 语义映射到 NPU 可承接节点或 fallback |
| GEMM choice | `src/torch_npu/torch_npu/_inductor/kernel/mm.py` | `mm/addmm` 的 ATen、CATLASS、fallback 和 autotune choice |
| scheduler/codegen | `src/torch_npu/torch_npu/_inductor/scheduler.py`、`codegen/` | fusion、tiling、wrapper 和最终内核生成 |
| experimental Triton | `src/torch_npu/torch_npu/_inductor/triton_experimental/` | NPU Triton 独立 backend、override、lowering 和 codegen |
| DVM/MLIR | `src/torch_npu/torch_npu/_inductor/dvm/`、`ascend_npu_ir/` | 其他后端的图融合、lowering 与代码生成 |

### Triton Ascend

`src/triton-ascend/` 是 Triton 编译器/后端源码。只有遇到 Triton 语言或 Ascend backend 本身缺陷时才修改这里。普通 pass 缺少 kernel 时，优先把 kernel 与 torch_npu 的 lowering/注册放在 torch_npu 或独立实验目录，不应先改 Triton 编译器。

当前 Triton Ascend 工作树已有三处非本任务修改，本任务必须保留并隔离。

## 首批三个候选如何修改

### `mm_plus_mm`

调用链：

```text
src/pytorch/torch/_inductor/fx_passes/post_grad.py:979
  is_valid_mm_plus_mm / mm_plus_mm
    -> src/pytorch/torch/_inductor/kernel/mm_plus_mm.py:128
       tuned_mm_plus_mm
```

NPU 仍在 `src/torch_npu/torch_npu/_inductor/fx_passes/post_grad.py:24` 保持共享 pattern gate；T-023 没有全局解除它，而是在同文件增加 default-off、NPU-only、different-K duplicate handler，并在 `torch_npu/_inductor/kernel/mm_plus_mm.py` 追加独立 K1/K2 template choice。开发步骤1–5已完成：

1. 先构造两个矩阵乘结果相加的触发图；
2. 证明上游 pattern 在 NPU 被 gate 拦截；
3. 比较“两次 mm + add”、vendor/CATLASS/AscendC 融合方案；
4. 已证明首批两 shape p50 改善 15.29%/18.04%，但 Triton workspace使 peak allocated多270,336 B；
5. 因 strict memory gate失败，template保持默认关闭、131072 output cap和extern fallback，verdict为条件性有益。

### `pad_mm`

上游实现位于 `src/pytorch/torch/_inductor/fx_passes/pad_mm.py`。NPU experimental backend 在 `src/torch_npu/torch_npu/_inductor/triton_experimental/overrides.py:74` 通过关闭 `shape_padding` 禁用整个 mm/addmm/bmm padding 家族；default路径还会被上游CUDA/XPU-only `check_device()`拦截。T-025测试侧绕过后已证明pad→GEMM→slice功能可承接，原“不能承接”假设不成立。

T-026在首批M/N/K非对齐static fp16 contiguous shape上完成三轮测量：mm/bmm/addmm p50分别回退72.65%/65.31%/120.63%，额外task和padding buffer是主因。当前正确结论是保持产品gate并拒绝独立padding替身；只有未来大shape/特殊layout先显示收益，才重新登记融合masked-load GEMM候选。

### `addmm` fusion

上游 add+mm -> addmm pattern 位于 `src/pytorch/torch/_inductor/fx_passes/post_grad.py:1853` 附近。NPU 已在 `src/torch_npu/torch_npu/_inductor/kernel/mm.py:126` 注册 addmm lowering，并可以选择 ATen/CATLASS/fallback；experimental backend 仍可在 `triton_experimental/fx_passes.py:534` 禁用 fusion。

因此这里首先要查 capability 条件和性能，而不是从零写 addmm。测试要覆盖 bias broadcast、alpha/beta、连续/转置输入、静态/动态 shape，以及 fusion 前后的 kernel 数。

### B2 alias-sensitive custom pass

入口位于 `src/torch_npu/torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py`。
T-028 证明只看数值会漏掉 storage alias 错误；T-029/T-031 因而修改：

- `cat_to_view_pass` 的 full-cover identity 分支不再直返 parent，而是 contiguous clone；
- `fold_reduce` 最终保留原 sum，不再执行 size-one reduction 折叠；
- `utils/get_binary_fold_result.py::_get_fold_result` 保留 alias-safe clone 实现，避免未来
  复用时重新引入直返输入错误；
- `test/_inductor/test_dynamic_shape_fx_passes.py` 用结构断言固定最终策略。

性能决定了两条路线不能合并处理：cat 的 task/显存受益，fold_reduce clone 则 p99 回退
6.72%。这里不需要再写 Triton copy，因为候选已经是单 Triton pointwise，且原 sum 更快。

T-032/T-033 又验证了同文件中的 `fold_cat`：同维、单用户 nested cat 在目标 pass 内由
2 个 cat 展平为 1 个，三轮 paired p50/p99 改善 10.14%/10.32%，task 2→1，allocated
peak 减少 2,097,664 B，因此现有 pass 已是 `supported-beneficial`，不需要替身实现。
`fold_cast`、`fold_clone`、`fold_detach` 的代表 positive 则在目标 pass 前被前序 pipeline
消除或旁路，当前只记 reachability-neutral，不能把前序收益归因给这些 pass。

T-034/T-035 继续验证 `fold_sink_view`、`fold_squeeze`、`fold_to_copy`、`fold_where` 和
`fold_redundant_ops`。前四条实际到达的 positive/negative 均通过数值、stride、alias 与
对象身份合同；`fold_to_copy` 在目标 pass 前被消除或转为 prims cast。`fold_where` 的
where→clone 虽让 device kernel 时间下降约 26.46%，但端到端 p50/p99 只改善 1.16%/3.12%、
task和显存不变，最终为 `supported-neutral`。这类纯图规约继续修改 FX pass 即可，不应为了
单独复制写一条新的 Triton kernel。

T-036 再验证 PRE layout pass `cat_slice_cat_fold_pass` 与 `pad_slice_fold`。修复前两条
alias case 的所有逐元素误差都是 0，但前者把两个独立 cat 输出合并成同一 storage/对象，
后者让 pad-slice 结果 alias 原输入且改变 stride。源码现以 cat1 完整用户集合和 pad-slice
消费者物化 allowlist 缩窄 rewrite；图输出、view、未知和原地路径保持原图。T-036 wheel 下
60/60 FX 测试与 6/6 NPU worker 通过。这里不应手写 Triton：问题在图变换的 alias 合同，
不是设备缺少 kernel；T-036 功能关闭时把 paired 性能留给下一任务。

T-037 已关闭上述性能：cat-slice-cat 三轮中位 p50/p99 改善 24.00%/22.87%、task
2→1；pad-slice 改善 31.35%/30.34%、task 3→1、allocated peak 减少 10,485,248 B。
两条均为 `supported-beneficial`，现有 FX pass 已是正确优化位置，无需 Triton 替身。

## 测试和性能证据

所有测试从 `/home/z50063656/tmp` 启动，不能在 torch_npu 源码目录中导入 torch。每个 backend 使用 fresh process，避免全局 patch 和 cache 串扰。

每个可适用 pass 至少记录：

- PyTorch、torch_npu、Triton Ascend、CANN、driver、SoC；
- dtype、shape、stride、动态/静态、前向/反向；
- pass observer/counter 和变换前后 FX 图；
- generated code、kernel 数、graph break 与 fallback；
- warmup、runs、mean、stdev、p50、p99、首轮编译时间、峰值内存；
- baseline 与 candidate 必须同机、同版本、同输入配对测试。

没有这些证据时只能标记 `not-run`，不能宣称 NPU 可用或更快。

## 当前进度与剩余工作

- 已生成旧 `/Dynamo` 静态快照 194 条。
- `Pass/src` 与旧扫描同口径时为 189 条；少的 5 条全部来自当前未初始化的 torchair 子模块，不是主干 pass 回退。
- 在同口径基础上补入 8 个 DVM/MLIR 图变换、53 个函数式/生成式 pattern 和 1 个 pad-mm 控制 gate，当前概念级清单为 251 条。
- 当前矩阵中 direct case 164 条、observer 41 条、registry container 25 条、人工审查 21 条；生成变体不重复计数。
- 环境已确认稳定并完成 P0、T-011 至 T-036；pad family和B2前18条已完成结构/NPU
  分流，fold_cat 已关闭有益性能、fold_where 已关闭中性性能结论，T-036 两条 layout pass
  已修复 alias 并在 T-037 关闭 beneficial 性能。当前进入 B2 其余9条，随后进入B3/B4；T-023只
  保留匹配 headers 环境的无 shim复验。
