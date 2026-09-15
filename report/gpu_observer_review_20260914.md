# T-091、T-102～T-107 GPU 精确归因复核与 NPU 下一步

> 更新时间：2026-09-14 20:04 CST（UTC+08:00）。收件提交：`e0aff07`。
> 后端边界：GPU `inductor-default`；NPU 仅 `triton_experimental`。

## 1. 本轮结论

7 批 **28/28 原生 case 通过、零 skip**，完整性校验通过，可恢复 1370 份文本。
114 条具名 entry 观察及其前后 FX 已按正文哈希核验，**25 个 case 有本编号目标证据，3 个命中其他编号**。
`valid-reference-suite` 仅是 runner 的原生成功/工件齐备判定，不能替代逐编号控制节点复核。
本轮没有执行上传的 Python 工件，没有凭观察器额外声明数值/梯度通过，也没有自动扩大冻结分母。

| 任务 | 原生 case | 本编号观察通过 | 待处理 |
| --- | ---: | ---: | --- |
| T-091 | 1/1 | 1 | NPU 原社区功能已通过；性能门禁待做 |
| T-102 | 5/5 | 5 | 逐例 NPU 功能 |
| T-103 | 5/5 | 5 | 逐例 NPU 功能 |
| T-104 | 5/5 | 5 | 逐例 NPU 功能 |
| T-105 | 5/5 | 3 | 16、17 的 reference 映射不符；18～20 可推进 |
| T-106 | 4/4 | 4 | 逐例 NPU 功能 |
| T-107 | 3/3 | 2 | 29 的 reference 映射不符；28、30 可推进 |

逐 case 的真实 target 名、训练/推理后缀、观察数量与 FX 哈希位于
`results/current/T-xxx/gpu_reference_review.json`。任务级 mixed 状态不应阻塞同批已通过的其他单元。

## 2. 收件选择与历史保护

T-091/104/105/106 使用新 `text-handoff.json`；T-102/103/107 使用新 `manifest.json`。
T-104～106 的旧分片仍在目录里，不能优先找到 manifest 就误读上轮。

`review_uploaded_gpu_tasks.py` 现在要求多入口时显式选择，`--check-current` 按复核记录绑定输入重验；
新增观察器审查要求精确编号、handler 成功返回、真实前后正文、未改测试/设备/断言/产品 gate。
历史复核已保存到 `results/history/T-xxx/<旧run>/gpu_reference_review.json`；原收件在 Git `d6889bb` 可恢复，
归档前已验证该 revision 的旧入口字节 SHA256 与旧复核一致，没有覆盖旧实验事实。

```bash
# 显式选择示例；只读复核，不修改产品、不自动冻结分母
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/review_uploaded_gpu_tasks.py \
  --task 104 \
  --input /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/results/incoming/T-104/text-handoff.json \
  --require-observer
```

## 3. 三个不同编号不是 NPU 缺陷

| 计划编号 | 原生观察到的成功 replacement | 判断 |
| --- | --- | --- |
| 16 | `_sfdp_pattern_14_inference`、`_sfdp_pattern_5_inference` | 本轮没有 16 的成功观察；不能沿用旧“16 已命中、只保留数学链”的预期表述 |
| 17 | `_sfdp_pattern_15_inference` | 本轮没有 17 的成功观察，不能以 15 的通过替代 |
| 29 | `_sfdp_pattern_30_{half_}{bs1_}{training/inference}`（实际后缀见 JSON） | 本轮没有 29 的成功观察；与 30 重叠但暂不自动合并单元 |

这些是 GPU reference 的精确映射问题，不是 NPU 不支持，更不是用户漏传文件。
继续重复当前命令预计不会增加本编号证据；先审核合同、注册条件及不同输入是否需要登记 derived 邻接，
再提供有明确意图的补跑命令。不得通过关闭竞争 pattern 或修改 guard 强制制造原生命中。

源码能解释候选方向，但下面是**源码推导**，不冒充观察器已经采集了所有前序简化步骤：

```python
# test/inductor/test_fused_attention.py:988、1095
# 16 的 CUDA 入口禁用训练检查；17 也只运行 inference，dropout(training=False) 会消去。
self._check_common(dot_prod_attention, check_train=False, has_dropout=True)

# test/inductor/test_fused_attention.py:1663，_test_sdpa_rewriter_29
attn_mask = torch.zeros(1, 1, query.size(1), key.size(1), device=query.device, dtype=query.dtype)
attn_weight = (q @ k_t) + attn_mask
attn_weight = torch.ops.aten._safe_softmax(attn_weight, -1)
# 零 mask 简化后可能与 30 的无 mask 图同构；本轮实际成功名称全部为 30。
```

已通过的 dropout 合同也不泛化：3/4/6/8/10/12 的 inference 可命中无 dropout 的相邻编号，
但 training 有各自精确编号，按实际后缀列示。6/28 的随机输出/梯度不逐值比较范围沿用原社区断言；
22 的特殊 stride 不要求最终始终保留 CUDA SDPA kernel。观察器不提供额外的数值 oracle。

## 4. T-091：为何命中但不能称为新计算优化

```python
# test/inductor/test_split_cat_fx_passes.py:1555
def fn(x, y):
    return torch.stack([x, y], axis=1)

# 本轮 GPU 及 NPU 的 normalize_stack_default 目标边界
# before：axis 已由更早处理规范化；after 主要是 handler 重建节点。
stack = torch.stack([x, y], dim=1)
stack_1 = torch.stack([x, y], dim=1)
```

源码 `torch/_inductor/fx_passes/split_cat.py:383` 的 handler 确实调用、返回并替换节点。
`graph_changed=true` 是文本/节点层变化，不能扩大为数学表达或运行 kernel 改变。
原社区数值断言成立；性能必须另用合法 OFF/ON 测量，不根据节点名变化认定收益。

NPU 原生入口因 main `HAS_GPU` 为 0 tests。最小 adapter 保留原方法与配置，只改 GPU 入口到真实 NPU。
物理 NPU 5 / 910B2 / Pass：1 test、0 skip，双方输出均为 NPU FP32 `[4,2,4]`，原 assertEqual 通过。
实际生成代码：

```python
# issues/REF-stack-axis-normalization-native/evidence/t091-native-y2vojco3/adapter/
# debug/torch_compile_debug/.../output_code.py:61
with torch.npu.utils.device(0):
    buf0 = torch.ops.aten.cat.default([arg0_1, arg1_1], 1)
return (reinterpret_tensor(buf0, (4, 2, 4), (8, 4, 1), 0),)
```

这是 NPU 图内 extern cat 与视图，没有 CPU 运算路径；不能写成 Triton 融合核。
后端由 adapter 注册器实际确认，启动器的通用 `backend_verified=false` 原字段保留，不事后伪填。
详细命令、环境、适配代码、源码重建调用链与全部 FX/IR/output_code 见
[复现报告](../issues/REF-stack-axis-normalization-native/复现报告.md)、
[适配报告](../issues/REF-stack-axis-normalization-native/适配报告.md)。

## 5. 下一步

1. 不要求用户重跑已有效的 25 个 case；attention 的 24 个单元逐例执行 NPU 原生入口，再按真实阻断最小适配。
2. T-091 补独立 OFF/ON 功能图审查及性能门禁，尚不记正式比较/性能完成。
3. 16/17/29 先重审 reference 映射，不以不同编号证据签 NPU/性能门禁。
4. T-098/T-100 首次收件、T-078 FP16 value=1 正确邻接、T-076/T-077 严格历史再认证仍独立待办。

本轮没有修改或部署 torch_npu 产品修复，没有创建 PR，也没有执行新的 git push。

## 6. 工具验证

统一 `validate_all.py` 检查通过，213 个标准库回归通过；其中新增观察器反例检查和同批逐编号状态检查。
原测试中“全部28个仍待归因”的历史固定断言，已更新为24个 attention 待 NPU、3个归因不符、1个 T-091 功能阶段通过，
同时继续断言正式 comparison、冻结 reference 和性能处置各40，防止阶段进度被误计为闭环。
严格历史再认证仍为 `pending=41、exempt=3`，不将工具通过当作历史补证完成。
