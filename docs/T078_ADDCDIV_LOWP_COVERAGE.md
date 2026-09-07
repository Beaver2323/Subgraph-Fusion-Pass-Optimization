# T-078 addcdiv FP16/BF16 覆盖修订与验证步骤

> 更新时间：2026-09-08 07:12 CST（UTC+08:00）
> 当前状态：FP32 修复与性能结论有效；FP16/BF16 GPU reference 已通过，等待补导出关键代码/IR 正文并进入 NPU 三臂归因。

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

## 4. NPU 后续三臂归因

GPU 通过后，NPU 必须使用 `triton_experimental`，并在三个 fresh process 中使用同一输入：

1. `OFF`：禁用目标重融合，观察常规编译路径；
2. `decomposed`：确认 `div → mul → add` 的低精度误差；
3. `refused`：启用 `addcdiv → FMA/div_rn` 重融合。

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
`ir_post_fusion.txt` 与 `output_code.py`。已有一次 FP16 ON 观察到最大绝对误差 `0.03125`，但 OFF
路径也观察到同量级差异，因此不能归因于 FMA；三臂同输入完成前既不判定 NPU 缺陷，也不测性能。

## 5. 当前结论边界

- FP32：功能修复已验证，`triton_experimental` 性能为 `PERF_NEUTRAL`，结论继续有效。
- FP16/BF16：不是产品显式 disable；当前是能力与精度待判定。
- 当前矩阵显示 5 个覆盖 variants：3 个已验证、2 个 pending；不会再显示成全 dtype 已闭环。
- 低精度正确性闭环后，若 NPU 合法启用，才按 T-078 现有 fresh-process OFF/ON 合同补性能。

## 6. 2026-09-08 GPU 回传复核

本轮只选择两条低精度派生 case，suite 因而正确标记为 partial；所选范围为 2/2 passed、2/2
`reference_valid=true`。两条 case 均在 PyTorch `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`、
CUDA 12.6、A100 上执行：

- FP16：FX 从 `div → mul(value=2) → add` 变为 `aten.addcdiv(value=2)`；位级一致、counter=1、
  `tl.fma` 与 `div_rn` 断言均通过。
- BF16：同一结构改写与同一组断言通过。

初次上传的 1.3 review 包只携带合并 FX，原始 FX、前后 IR、`output_code.py` 和日志仅有哈希。
这不推翻 GPU pass，但不满足独立代码审计。导出规则已修正；从现有 run 补导出即可，不需要重跑 GPU。
