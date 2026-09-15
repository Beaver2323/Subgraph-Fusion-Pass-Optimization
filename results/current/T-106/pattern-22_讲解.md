# T-106 pattern 22：T5加法mask与K/V返回，修复后再评估性能

> 更新时间：2026-09-15 11:56 CST（UTC+08:00）。原例、独立OFF/ON及六臂验签完成；功能修复PASS，性能PERF_REGRESSED。

## 1. Pattern及意图

```python
# 冻结PyTorch8e86e0a，torch/_inductor/fx_passes/fuse_attention.py:654
def _sfdp_pattern_22(query, key, value, attn_mask):
    query = query.permute([0, 2, 1, 3])
    key = key.permute([0, 2, 1, 3])
    value = value.permute([0, 2, 1, 3])
    score = torch.matmul(query, key.permute(0, 1, 3, 2))
    masked_score = score + attn_mask
    score = masked_score.type_as(query)
    return score.float().softmax(dim=-1).type_as(query).matmul(value), key, value
```

它识别T5样式attention计算链，既优化attention，又保留转置后的K/V返回给调用者。
没有额外scale，因此replacement的SDPA使用`scale=1.0`；浮点mask先转query.dtype。
注册与改写属于joint FX阶段，在lowering、scheduler与最终kernel选择之前。
**命中SDPA图改写不保证最终执行融合FA内核。**

## 2. 功能测例从哪里来

原社区方法 `test/inductor/test_fused_attention.py:1387 _test_sdpa_rewriter_22`：
FP32 Q/K/V `[4,2,16,32]` 与共享mask `[1,1,2,2]`，再覆盖batch=1与mask转置。
四组输入 × 返回attention/K/V三个Tensor，共12次比较；不测试反向传播。
NPU保留原数值断言与四组循环，仅做已审查的设备和CUDA代码符号适配。

| 对比项 | GPU reference | NPU triton_experimental |
| --- | --- | --- |
| 原社区运行 | 1方法、0跳过、本编号4次精确改写 | 安装态完整方法1/1、0跳过、4次精确改写 |
| 原数值合同 | 原CUDA测试通过 | 修复前首组严重超差；修复后12次Tensor比较通过 |
| 下游实现 | 原CUDA代码断言适用，GPU原件见reference审查 | 数学展开：NPU bmm、safe_softmax及bmm，非fusion_attention |
| 对齐结论 | 冻结源码的reference | PARTIAL_ALIGNED：数值合同通过，kernel实现不同 |

[GPU逐编号审查](gpu_reference_review.json)绑定用户上传的原运行、哈希及实际目标边界FX；
本报告没有重新执行GPU或宣称GPU性能数据。

## 3. 遇到的真实缺陷及修复

首组attention输出3772/4096元素超差，并非未命中：共享mask的load实际外维为1，
生成器却用输出tile外维发射`extract_slice`。三种子逐kernel定位到行最大值首先错误。

```python
# torch_npu/_inductor/triton_experimental/codegen/triton.py
# _maybe_rewrite_select_lane_load发射示意；对应完整实际代码见修复报告
# 之前：把共享mask的singleton外维错误扩大
extract_slice(full, [0, 0, 0], [real_block_x2, real_block_x1, 1], [1, 1, 1])
# 之后：每个load各自的真实外维；后续算子再做合法广播
extract_slice(full, [0, 0, 0], [full.shape[0], full.shape[1], 1], [1, 1, 1])
```

同时为可静态证明范围的拓宽load添加地址上下界，未知布局保留原load。
仅改两个NPU方法；未改PyTorch、容差或产品禁用项。安装态原例与3邻接共21次Tensor/9次改写、
22场景/23执行边界全部通过；[根因及调用栈](../../../issues/REF-sfdp-pattern-22-native/根因分析.md)、
[前后FX/IR/output_code与部署备份](../../../issues/REF-sfdp-pattern-22-native/部署与边界回归报告.md)。

## 4. 性能测例与实际OFF/ON

没有找到针对22号FX改写的社区原生benchmark；社区SDPA benchmark直接从已融合SDPA接口起步，
不能用于判断这个FX rewrite收益。因此复用冻结 `_get_sfdp_patterns` 同编号half-inference
注册样例的search函数、shape、stride和dtype，替换未初始化empty数据为固定种子正态输入（std=0.25）。
这属于**社区注册图派生微图**，不是社区原生性能测例，也不是模型端到端。

| 输入 | 形状 | dtype |
| --- | --- | --- |
| Q/K/V | `[2,4,8,16]` | FP16 |
| 加性mask | `[2,1,1,4]` | FP32 |

每臂比较返回tuple全部Tensor与同卡eager，派生域`rtol=0.2 / atol=2e-3`，
不回写原社区容差；dropout=0，仅inference-forward，无梯度。
OFF仅删除22号注册，整轮joint与邻接保持；实际OFF总attention及22精确计数均0，ON均1。
两臂无graph break或CPU转移线索，先确认数值与精确目标，再签门禁。

```python
# 实际功能输出：issues/REF-sfdp-pattern-22-native/evidence/
# functional-20260915T112026+0800/{off,on}/debug/torch_compile_debug/.../output_code.py
# 以下保留实际调用结构，省略缓存分配和具体参数：
# OFF：两个FP16 BMM + 中间cast/softmax Triton核
extern_kernels.bmm(q_half, kt_half, out=scores_half)
# add_mask -> FP16 -> FP32 -> softmax -> FP16
extern_kernels.bmm(prob_half, v_half, out=output_half)
# ON：FP32提升、两个FP32 BMM、安全softmax、多输出K/V仍保留
extern_kernels.bmm(q_float, kt_float, out=scores_float)
torch.ops.aten.any.dim(non_inf, -1, True)
# safe_softmax的amax、exp-sum、全遮挡处理等实际Triton核
extern_kernels.bmm(prob_float, v_float, out=output_float)
# cast回FP16；返回output、原K/V的permute视图
```

因此性能是在**修复后的同一安装态**比较22号开/关，不能写成“修复补丁加速比”。
实际是否变快由六臂样本判断，不凭“发生融合”推定收益。

## 5. 性能与当前处置

物理NPU5，OFF1/ON1/ON2/OFF2/OFF3/ON3六个新进程独占串行；预热10次、
host同步墙钟与NPU Event各100次，编译时间/峰值内存单列。六臂全部通过数值与精确目标检查。

| 臂 | host p50 / p99（ms） | Event p50 / p99（ms） |
| --- | --- | --- |
| OFF1 | 0.713345 / 0.762149 | 0.597610 / 0.627605 |
| ON1 | 0.997625 / 1.439624 | 0.859660 / 1.103087 |
| ON2 | 0.899575 / 0.982403 | 0.783970 / 0.831388 |
| OFF2 | 0.751750 / 0.792392 | 0.632840 / 0.661563 |
| OFF3 | 0.742460 / 0.801660 | 0.617070 / 0.656540 |
| ON3 | 0.912760 / 0.997290 | 0.797220 / 0.840294 |

按三个同模式臂分位数取中位数：host p50/p99时延增加 **22.94%/25.86%**，
Event增加 **29.19%/27.99%**，正式判定`PERF_REGRESSED`。四组p50轮间最大/最小比为1.054～1.109，
没有触发既定1.2高波动门禁；ON1的尾延迟较高也原样保留，不挑选最快一轮。
OFF编译加首次执行约90.43～92.91秒，ON约128.71～131.51秒，不混入稳态时延。
记录的峰值allocated为OFF55,296/ON59,904字节，reserved均316,669,952字节；不等于整个设备占用。

实际生成代码支持的解释是：ON走数学SDPA而非融合FA，两个BMM提升为FP32，
safe-softmax由4个相关核增至6个，并额外执行NPU `any`和转换核。
这说明路径更重，与实测回退一致；没有做逐算子profiling，因此不把回退百分比精确归因给某一个kernel。

原件入口：[八臂与同卡原例验签包](attention_bundles/pattern-22.json)、
[六臂样本/FX/IR/output_code](../../../issues/REF-sfdp-pattern-22-native/evidence/benchmark-20260915T114030+0800/)、
[性能汇总](performance_summary.json)。离线复核命令如下，不运行NPU：

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/review_attention_completion.py \
  --check-current \
  --pattern 22
```

本轮不改默认开关；若回退也不撤掉数值正确性修复。21号的邻接接替问题单独保留，不能计入本号收益。
本号正式闭环；本报告11:56时T-106合计3/4（22/23/24）。17:21用户另接受21号归因受限处置后，
批次为4/4已处置（仅3项实测）；[21号单列讲解](pattern-21_讲解.md)。23/24性能为各自历史源码指纹下的结果，
本轮只新增其安装态邻接功能复验，没有重测或迁移旧性能到新安装态。
