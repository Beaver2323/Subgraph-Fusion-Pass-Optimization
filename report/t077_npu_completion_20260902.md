# T-077 NPU/comparison 闭环与修复验证报告

> 更新时间：2026-09-02 22:54 CST（UTC+08:00）
> PyTorch：`release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> torch_npu 基线：`master@83cc452480c3546fd5cccf853bfe3a360ce9dbfc`
> NPU：Ascend910B2，backend=`triton_experimental`

## 结论

T-077 已完成 5/5 acceptance units 的 GPU/reference → NPU → comparison 闭环。GPU 分母由
11/11 direct cases、17/17 variants 冻结；NPU 侧所有单元均有动态证据。最终得到 1 个完整行为一致、
3 个以设备门禁为首个分歧的预期产品差异，以及 1 个真实 lowering correctness 回归。

| acceptance unit | GPU reference | NPU 结果 | 单元结论 | 修复 |
| --- | --- | --- | --- | --- |
| `AU-apply-gumbel-max-trick` | 分布正例命中 1 次 | 同样命中 1 次，统计合同通过 | `BEHAVIOR_UNCHANGED` | 不需要 |
| `AU-b2b-gemm` | 4 正例命中、2 负例不命中 | 正例由 CUDA/XPU guard 拒绝，6/6 数值正确 | `EXPECTED_PRODUCT_DIVERGENCE` | 不需要 |
| `AU-decompose-mem-bound-mm-decompose-bmm` | 正例分解、2 负例保持 bmm | 干净卡 3/3 正确；正例 guard 拒绝 | `EXPECTED_PRODUCT_DIVERGENCE` | 不需要 |
| `AU-decompose-mem-bound-mm-decompose-mm` | 2 正例分解、4 阈值负例保持 mm | 目标 pass 均按产品边界处理；tiny-mm 负例暴露错误左梯度 | `NPU_REGRESSION` | 本地候选已验证 |
| `AU-decompose-mem-bound-mm-decompose-addmm` | 原始超大动态 M 命中 1 次 | 同尺寸保持 addmm，counter=0 且正确 | `EXPECTED_PRODUCT_DIVERGENCE` | 不需要 |

## MM 回归定位

失败输入是 fp32 `2048x2 @ 2x2`，属于 `decompose_mm` 的 M 阈值负例。目标 pass 在 GPU/NPU 都不应
命中，NPU transformed FX 也确实保留 `aten.mm`。原安装态的观测为：

- forward 最大绝对误差：`4.76837158203125e-7`；
- right gradient 最大绝对误差：`0`；
- left gradient 最大绝对误差：`29.026336669921875`；
- left gradient 不一致元素：`4085/4096`。

生成代码显示 AOTAutograd backward 的 `mm_2` 被 upstream `_use_small_mm_pointwise` 选成两个 NPU
Triton pointwise kernel。因此首个分歧在 lowering，不在 FX pattern、decomposition、scheduler 或
运行时。详细源码与 GPU/NPU 行为解释见
[`t077_pattern_gpu_npu_guide_20260902.md`](t077_pattern_gpu_npu_guide_20260902.md)。

## 修复候选

独立 worktree 中的候选提交：

```text
dfbcc25b76743ea6c1c5cd61b6b30f0a910148a6
fix(inductor): guard NPU small-mm pointwise lowering
```

它在 `torch_npu/_inductor/triton_experimental/overrides.py` 中对 upstream
`_use_small_mm_pointwise` 增加 NPU-only、幂等 guard；NPU 返回 `False` 后使用可靠 extern mm，其他设备
继续调用原函数。验证结果：

- 新增 NPU-only 行为与幂等性单测：2/2 通过；
- fp32 与 bf16-autocast 六变体：6/6 通过；
- 每个变体的 forward、left-grad、right-grad 误差均为 0；
- `git diff --check` 与 Python AST 解析通过。

候选当前只存在于本地 detached worktree，未推送、未合入，不能登记为正式 fixed issue。可审阅补丁为
[`backend_fix_dfbcc25.patch`](../issues/REF-decompose-mm-native/backend_fix_dfbcc25.patch)。

## 证据入口

- 结构化单元结果：`results/current/AU-*/{npu_result,comparison_result}.json`；
- GPU reference：[`t077_gpu_reference_20260902.md`](t077_gpu_reference_20260902.md)；
- 逐 pattern 学习导读：[`t077_pattern_gpu_npu_guide_20260902.md`](t077_pattern_gpu_npu_guide_20260902.md)；
- 原始失败：`/home/z50063656/tmp/t077-npu-results/decompose-mm-isolate0/`；
- 修复后六变体：`/home/z50063656/tmp/t077-npu-results/decompose-mm-fixed-suite0/`；
- 干净卡 BMM：`/home/z50063656/tmp/t077-npu-results/decompose-bmm-clean-final3/`；
- 原始大 M addmm：`/home/z50063656/tmp/t077-npu-results/decompose-addmm-clean0/`。

## 后续边界

T-077 验证任务本身已完成。剩余动作是产品代码变更评审：获得授权后推送/合入候选，重新构建或安装
正式 torch_npu 产物，并用同一六变体合同回归；只有正式合入后，才将该记录从
`regressions/known_issues.yaml` 移入 `regressions/fixed_issues.yaml`。
