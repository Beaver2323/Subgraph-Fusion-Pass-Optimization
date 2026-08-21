# P1：NPU Custom、DVM/MLIR 与 Attention 批次设计

## 范围

评估矩阵中的 P1 共 66 条：

| 批次 | 数量 | 目标 |
|---|---:|---|
| B2_NPU_CUSTOM | 27 | torch_npu 自定义 pre/post FX pass |
| B3_DVM_MLIR | 8 | DVM/MLIR 图融合、规范化和精度变换 |
| B4_ATTENTION | 31 | 上游 30 个 SDPA pattern family + 1 个 NPU graph attention 包装入口 |

T-027 已为 B2 中7条记录补齐device-independent结构层动态结论，仍没有任何一条因此被判为NPU端到端可用。源码里有测试不等于NPU pass可用；必须继续补齐实际NPU触发、前后图、正确性、generated code/fallback和性能证据。

## B2：27 个 NPU custom pass

注册入口位于 `torch_npu/_inductor/fx_passes/ascend_custom_passes/`。按语义拆成四组：

| family | pass | 数量 | 首要验证 |
|---|---|---:|---|
| 冗余/恒等消除 | `fold_four_op_pass`、`fold_cast`、`fold_cat`、`fold_clone`、`fold_detach`、`fold_expand`、`fold_reduce`、`fold_sink_view`、`fold_slice`、`fold_squeeze`、`fold_to_copy`、`view_fold_pass`、`fold_where`、`fold_redundant_ops` | 14 | alias、stride、symbolic shape 和被删除节点是否真的无语义影响 |
| layout/搬运规约 | `cat_to_view_pass`、`repeat_to_expand_pass`、`cat_slice_cat_fold_pass`、`pad_slice_fold` | 4 | view/expand 是否保持写入语义，是否减少 copy/kernel |
| dtype/index/mask 规约 | `dtype_optimal_pass`、`fold_iota_arithmetic_pass`、`broadcast_const_mask_compress`、`masked_add_compose_pass`、`bool_cast_mul_to_where_pass`、`sign_diff_hamming_fuse_pass` | 6 | int64→int32 边界、广播、布尔语义、溢出和动态 shape |
| 复合算子融合 | `batch_embedding_fusion_pass`、`fused_matmul_relu_pass`、`fusion_attention_v3_pass` | 3 | vendor op 能力、训练/反向、capability gate 和真实 kernel 收益 |

### 运行条件不能忽略

- 多数 custom pass 以 inference 为主要路径。`run_register_pre_custom_passes()` 在 inference 时执行全部 PRE pass；训练时只额外允许 `fusion_attention_v3_pass`。
- `fused_matmul_relu_pass` 默认不注册，只有 `TORCHINDUCTOR_ENABLE_FUSED_MATMUL_RELU=1` 时才有资格测试；不能把默认未执行记为“不支持”。
- `SHUT_DOWN_FX_PASS_LIST` 可以关闭单个 pass 或全部 pass。动态报告必须记录这个环境变量，否则 observer 结果不可比较。
- 静态源码显示：inference 路径先遍历全部 PRE pass，随后又按名称执行一次 `fusion_attention_v3_pass`。该 pass 本身是替换后即幂等，但存在重复扫描/编译开销嫌疑；应在动态 observer 中确认调用次数，再决定是否提出去重修改。

### 已有测试证据与缺口

`test/_inductor/test_dynamic_shape_fx_passes.py` 对以下 7 个 pass 有直接结构断言：

- `fold_expand`
- `view_fold_pass`
- `fold_reduce`
- `fold_slice`
- `repeat_to_expand_pass`
- `fold_four_op_pass`
- `cat_to_view_pass`

这些测试证明部分 static/symbolic FX 变换边界，不证明 NPU 编译、lowering 或性能。其余 20 个 custom pass 没有在当前源码树中找到同名的直接 pass UT；即便存在模型级间接覆盖，也需要单独建立最小触发图。

B2 的建议执行顺序：

1. 先跑 14 个冗余/恒等消除的结构 UT，记录节点计数和负例。
2. 再跑 4 个 layout/搬运规约的 NPU compile，检查 copy/view/expand 生成代码。
3. 对 6 个 dtype/index/mask pass 增加极值、非连续和 dynamic shape 参数。
4. 最后测 3 个复合融合；性能 baseline 必须关闭单个 pass，而不是拿 eager 与整图 compile 比。

## B3：8 个 DVM/MLIR 变换

| pass | 类型 | 最小正例 | 必要负例 |
|---|---|---|---|
| `dvm_graph_fusion` | partition + fused custom op | add→mul→sum | unsupported op 打断 partition、动态图 fallback |
| `annotate_mm_transpose_flags` | metadata | 转置 placeholder 进入 mm/bmm/addmm | 非 placeholder 转置、普通连续输入 |
| `decompose_k1_matmul_to_mul` | graph rewrite | K=1 的 mm/bmm | symbolic K、K≠1、转置输入 |
| `insert_sum_fp32_prepost_cast_prims` | precision rewrite | fp16/bf16 sum | fp32 输入、输出 dtype 已为 fp32 |
| `insert_promote_cast_by_pos_prims` | dtype promotion | 混合 dtype add/mul/compare | 同 dtype、非受支持 op |
| `expand_to_reshape` | layout rewrite | 可等价 reshape 的 expand | 真广播 expand |
| `DvmMlirPostGradPass` | pass 聚合器 | 含上述可变换节点的图 | 无匹配图保持不变 |
| `fold_sum_cast_to_dtype` | MLIR 规约 | sum 后 cast 合并到 dtype | cast 被多用户消费、dtype 不等价 |

现有 `test/_inductor/test_dvm_graph_fusion.py` 已覆盖 DVM graph fusion 的静态/动态图、fp16/fp32/bf16 和一个 matmul 场景，但未逐项证明其余 7 个子变换。`test_dvm_mlir_fusion.py` 可作为端到端候选入口，仍需把每条矩阵记录关联到明确的触发断言。

B3 必须分两层判定：

1. 结构层：变换前后 FX 图符合预期，负例不误改。
2. 后端层：DVM/MLIR 实际承接融合子图，生成代码不走意外 ATen/CPU fallback，数值和性能通过。

只通过结构层时，矩阵最多记录 `trigger confirmed`，不能把 verdict 写成 `supported-beneficial`。

## B4：30 个 SDPA pattern + 1 个 NPU 包装入口

上游 `_sfdp_pattern_1` 至 `_sfdp_pattern_30` 在 `joint_graph` lazy init 中按输入设备生成注册。清单按 30 个概念 pattern 计数，不把 float/half、batch=1、training/inference 等生成变体重复算成新 pass。

| pattern family | 语义 |
|---|---|
| 1–6 | 基础 QKᵀ→scale→mask/dropout→softmax→V |
| 7–12 | Q/K/V permute、fp32 softmax 和 dropout 变体 |
| 13 | 3D bmm attention |
| 14–17 | BERT/DistilBERT mask、permute、dropout 变体 |
| 18–20 | GPT2 causal mask、额外 mask、返回/布局变体 |
| 21–27 | T5 mask、dropout、返回 K/V 的 alias/tuple 变体 |
| 28 | Visformer 非连续 Q/K/V |
| 29–30 | `_safe_softmax` 的 SDPA math decomposition（有/无 mask） |

每个 pattern 至少需要：

- inference 正例；非 inference-only family 再加 training + backward。
- fp16/bf16；fp32 只在 NPU kernel/精度策略允许时执行。
- batch=1 与 batch>1，因为生成图的 clone/reshape 结构不同。
- 无 mask、float mask、bool mask、causal mask；只选择该 family 实际支持的组合。
- 连续与非连续 Q/K/V；tuple 返回 family 必须核对 K/V alias、stride 和梯度。
- `counters["inductor"]["fuse_attention"]`、变换后 SDPA 节点、最终 NPU attention kernel 三项同时出现，才算端到端触发。

PyTorch 的 `test/inductor/test_fused_attention.py` 已有大量上游 pattern 测试，可以复用函数和期望 pattern 名，但其原始设备/后端结论不能直接搬成 NPU 结论。

第 31 条 `npu_fusion_attention_graph` 不是与 30 个 generated pattern 等价的 pattern handler，而是 `torch_npu.npu_fusion_attention_graph` 的 autograd/custom-op 包装入口，当前应保持 `manual-review`。现有 `test/_inductor/test_npu_fusion_attention_graph.py` 主要检查 shape/meta 和直接 op 调用，没有证明 Inductor pattern 触发与生成代码。

静态审查还发现一个必须补精度断言的分支：`npu_fa` 只在 scale 作为第 9 个位置参数时执行倒数转换，关键字 `scale=` 路径没有同样处理；现有 `test_npu_fa_forward_scale_handling` 只检查 shape。因此先建立 positional/keyword 同语义数值对照，再判断是接口约定还是实现缺口，当前不直接修改。

## P1 动态验收顺序

环境解冻后按以下顺序执行：

1. 重新确认运行时源码 commit/导入路径与 `Pass/src` 一致。
2. 对 B2/B3 先跑不依赖性能的结构正负例，确认 harness/observer 有效。
3. default backend 跑 B2 NPU custom；DVM、MLIR 分别 fresh process 跑 B3。
4. B4 先选 pattern 1、5、13、18、21、28、29 做七个代表 family smoke，再扩到全部 30。
5. 只有功能、generated code 和 fallback 三项明确后，才做 pass-on/pass-off paired benchmark。
6. 任一失败先分类为未触发、变换错误、lowering/codegen 缺口、精度、环境或性能，不直接跳到手写 Triton。

## 当前静态结论

- P1 66 条已经完成执行分组和证据缺口定义；T-027现已完成首批7条的结构UT，32/32通过。
- 这7条只标为`structure-trigger-confirmed`，没有任何一条因此被标成NPU端到端可用。
- 两个值得优先验证的静态风险是 PRE attention pass 的重复执行，以及 `npu_fa` scale positional/keyword 路径不一致。
- 共享环境仍冻结，本批次没有运行 NPU，也没有修改 PyTorch、torch_npu 或 Triton Ascend 源码。
