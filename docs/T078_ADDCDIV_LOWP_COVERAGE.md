# T-078 addcdiv FP16/BF16 覆盖修订与验证步骤

> 更新时间：2026-09-08 22:44:08 CST（UTC+08:00）
> 当前状态：FP32/BF16 已完成功能与性能闭环；FP16 GPU reference 有效，NPU 已通过显式低精度
> 舍入 lowering 修复 `value!=1` 原 FMA 图。`value=1` 预期不命中，已移除补充 pattern。
> 修复前 OFF 不正确，FP16 暂无合法
> pass 收益 denominator。

## 1. 为什么需要补测

PyTorch 的 addcdiv 重融合 guard 并没有屏蔽 FP16/BF16：

```python
# PyTorch: torch/_inductor/fx_passes/post_grad.py:2040-2061
if not (isinstance(inp_val, torch.Tensor) and inp_val.dtype.is_floating_point):
    return False
if not (
    isinstance(out_val, torch.Tensor)
    and out_val.device.type in ("cuda", "xpu")
):
    return False
for key in ("t1", "t2"):
    ...
    if not (isinstance(val, torch.Tensor) and val.dtype.is_floating_point):
        return False
```

lowering 同样按 floating dtype 接受该路径：

```python
# PyTorch: torch/_inductor/lowering.py:8582-8610
use_fma = (
    dtype.is_floating_point
    and device is not None
    and device.type in ["cuda", "xpu"]
)
...
if use_fma:
    return ops.fma(value_expr, ops.div_rn(t1_val, t2_val), self_val)
```

CUDA FP16 eager 的 `acc_type` 是 FP32，所以 `value!=1` 与 Inductor 共享如下舍入链：

```text
q32   = RN_FP32(float(t1) / float(t2))
r32   = RN_FP32_FMA(float(value), q32, float(self))
out16 = RN_FP16(r32)
```

两条路径都不对 quotient/product 做 FP16 中间舍入。`div_rn` 固定除法的 FP32 舍入，
FMA 避免普通 mul/add 之间多一次 FP32 舍入，最终 store 才落回 FP16。因此 CUDA 的
位级一致来自 eager/compiled 使用相同的 FP32 运算顺序和舍入边界，不是忽略了低精度。

但两条社区测例使用未指定 dtype 的 `torch.randn(64, 64)`，实际只覆盖 FP32。原统计矩阵把
3 个 FP32 variants 写成整个 acceptance unit 已覆盖，遗漏了源码允许而社区未执行的低精度域。

## 2. 本轮最小扩展

社区原生 12 条 case 和已有 FP32 结论不变。新增两条 `tracking_mode=derived` 测例：

- `REF-addcdiv-fma-fp16-derived`
- `REF-addcdiv-fma-bfloat16-derived`

派生入口位于 `runners/t078_addcdiv_dtype_reference.py`。它只把 dtype 改为 FP16/BF16，保持社区
`64x64`、`value=2`、eager/compiled 对照、counter=1、`tl.fma` 和 `div_rn` 判据。它不是社区
原生测例，结果中会写成 `derived-valid-after-community-contract`。

关键断言如下：

```python
# Tracker: runners/t078_addcdiv_dtype_reference.py
self.assertTrue(torch.equal(actual, expected))
self.assertEqual(counters["inductor"].get("addcdiv_fma_fused", 0), 1)
self.assertIn("tl.fma", code)
self.assertIn("triton.language.div_rn", code)
```

## 3. GPU 一键执行

在 GPU 服务器更新仓库后，从数据盘临时目录运行：

```bash
export TRACKER_ROOT=/data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization

cd /data/z50063656/tmp

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-078 \
  --gpu 2
```

该命令会执行 12 条原生 case 加 2 条 dtype 派生 case，并自动生成适合文本复制的 handoff。
若只补低精度：

```bash
cd /data/z50063656/tmp

bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-078 \
  --gpu 2 \
  --case REF-addcdiv-fma-fp16-derived \
  --case REF-addcdiv-fma-bfloat16-derived
```

成功条件不是“脚本退出 0”这么宽泛，而是两条 case 都满足：测试数为 1、无 skip、FX before/after
存在、位级一致、counter=1 且生成代码包含 FMA/div_rn。失败必须原样回传，不能改为容差通过。

## 4. NPU 三臂归因

GPU 通过后，NPU 使用 `triton_experimental`，并对每个 dtype 在三个 fresh process 中使用同一输入：

1. `OFF`：禁用目标重融合，观察常规编译路径；
2. `decomposed`：确认 `div → mul → add` 的低精度误差；
3. `re-fused`：启用 `addcdiv → FMA/div_rn` 重融合。

必要调用链与证据：

```text
torch.compile
  -> post_grad_passes
  -> _is_addcdiv_fma_eligible
  -> _fuse_addcdiv_to_fma
  -> aten.addcdiv lowering
  -> Triton Ascend codegen
```

每臂保存 `fx_graph_readable.py`、`fx_graph_transformed.py`、`ir_pre_fusion.txt`、
`ir_post_fusion.txt` 与 `output_code.py`。正式运行共六个 fresh process：

- FP16 三条 compiled 输出哈希相同，均相对 eager 最大误差 `0.0625`、1187/4096 元素不同；
  重融合未增加误差，但当前 NPU Triton 编译路径不满足 eager 位级合同。
- BF16 的 OFF、分解、重融合和 eager 输出哈希完全相同，重融合 counter=1 且生成 FMA/div_rn。

完整数据和调用栈见
[低精度三臂与 BF16 修复报告](../issues/REF-addcdiv-fma-codegen-native/低精度三臂与BF16修复报告.md)。

## 5. 当前结论边界

- FP32：功能修复已验证，`triton_experimental` 性能为 `PERF_NEUTRAL`。
- BF16：产品 guard 已最小放宽，正式源码位级/codegen 验证通过；三轮 OFF/ON 为
  `PERF_NEUTRAL`，保持启用。
- FP16：GPU reference 的正式扩展合同为 `value=2`。NPU 对 `value!=1` 原 FMA 图显式保存除法、
  乘法后的 FP16 舍入并保持 counter=1；`value=1` 只属于 bitwise 邻接，预期 counter=0。
- 当前矩阵显示 5 个覆盖 variants：5 个已验证、0 个 pending。

## 6. 2026-09-08 GPU 回传复核

本轮只选择两条低精度派生 case，suite 因而正确标记为 partial；所选范围为 2/2 passed、2/2
`reference_valid=true`。两条 case 均在 PyTorch `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`、
CUDA 12.6、A100 上执行：

- FP16：FX 从 `div → mul(value=2) → add` 变为 `aten.addcdiv(value=2)`；位级一致、counter=1、
  `tl.fma` 与 `div_rn` 断言均通过。
- BF16：同一结构改写与同一组断言通过。

初次上传的 1.3 review 包只携带合并 FX，原始 FX、前后 IR、`output_code.py` 和日志仅有哈希。
这不推翻 GPU pass，但不满足独立代码审计。导出规则已修正；从现有 run 补导出即可，不需要重跑 GPU。

## 7. NPU 修复与性能结果

产品源码的当前范围为 `(FP32, FP16, BF16)`。FP32/BF16 使用 FMA/div_rn；FP16 使用后端专属
显式舍入 lowering。正式计数范围是 FP16 `value=0.3/2/7.7`：均 bitwise true、最大误差 0、
counter=1。旧证据中的 `value=1 counter=1` 来自已移除的 `add(div)` 补充 pattern，只保留为
被纠正历史；它不参与 FMA acceptance unit 或性能统计。
BF16 性能沿用社区功能 case 的 `64×64/value=2`，
三轮 OFF/ON 的 NPU Event p50 回退 `0.78%`、p99 改善 `9.86%`，没有显存或 dispatch 变化，按
阈值判定 `PERF_NEUTRAL`。

机器可读结果位于：

- `results/current/AU-post-grad-fuse-addcdiv-to-fma/lowp_three_arm_result_20260908.json`；
- 同目录的 `bfloat16_source_fix_result_20260908.json` 和
  `fp16_precision_boundary_result_20260908.json`（修复前历史）、
  `fp16_source_fix_result_20260908.json`（当前）；
- `results/current/T-078/addcdiv_bfloat16_performance_summary_20260908.json`。

FP16 没有报告正式 OFF/ON 收益：修复前 OFF 本身不等价于 eager，不能作为收益 denominator。
详细根因、源码框、调用栈、FX/IR/output_code 对照见
[FP16 精度修复报告](../issues/REF-addcdiv-fma-codegen-native/FP16精度修复报告.md)。

社区 FP16 eager、Inductor lowering 与通用 `emulate_precision_casts` 的边界详见
[社区 FP16 精度处理与 value=1 合同纠正](../issues/REF-addcdiv-fma-codegen-native/社区FP16精度处理与value1合同纠正.md)。
