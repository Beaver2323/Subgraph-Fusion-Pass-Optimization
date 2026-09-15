# T-078 FP16 value=1：GPU 补证与 NPU 修复对照

> 更新时间：2026-09-14 22:10 CST（UTC+08:00）。这是历史修复的新增 GPU 邻接复核，本轮没有修改或重跑 NPU 产品。

## 结论与范围

本次 `reference-20260914T214312+0800-jqft26ak` 确实执行
`REF-addcdiv-fma-fp16-value1-derived`，1/1、零 skip，原 stdout 明确打印
`dtype=float16 value=1.0 bitwise_equal=True max_abs_error=0.0`。仅关闭这个已登记的邻接缺口：
不是完整 T-078 suite 重跑，不增加 acceptance-unit 或性能分母。原错误命名 codegen 包仍保留。

GPU/NPU 都应保持普通 `div + add`，FMA pattern 不命中。两端各自对齐 eager，但并非要求
GPU 与 NPU 的中间舍入、结果逐位一致；总体仍记为 `PARTIAL_ALIGNED`。

## 测例从何而来

社区入口为冻结 PyTorch `8e86e0a` 的
`test/inductor/test_torchinductor.py::GPUTests.test_addcdiv_fma_bitwise_equal_cuda`。
本仓派生入口保留 `64×64`、`value=1`、同设备 eager/compiled 位级判据及 counter=0，只扩展到 FP16。
这是派生测例，不应称作社区原生 FP16 测例。

```python
# 本仓 runners/t078_addcdiv_dtype_reference.py:63（节选）
def fn(s, t1, t2):
    return torch.addcdiv(s, t1, t2, value=args.value)

# 同文件:83 起：位级判据及 value=1 负命中判据
self.assertTrue(torch.equal(actual, expected))
expected_fusion = 0 if args.value == 1.0 else 1
self.assertEqual(counters["inductor"].get("addcdiv_fma_fused", 0), expected_fusion)
```

原 GPU run 没保存执行器源码快照，counter 也没有独立打印。这里依据已绑定 SHA256 的
原始日志、明确位级输出、FX/codegen 与仓库入口交叉复核；不冒充更强的运行时源码溯源认证。

## 为什么都不命中，生成代码却不同

```text
torch.addcdiv(value=1)
  → decomposition 消除乘一
  → FX: aten.div.Tensor → aten.add.Tensor
  → 不满足 div→mul→add pattern，因此 addcdiv_fma_fused=0
  → 后端 lowering/codegen → 真机执行 → 各自与 eager 位级比较
```

GPU 原生成文件归档在本 issue 的
`evidence/gpu-20260914T214312/cases/REF-addcdiv-fma-fp16-value1-derived/debug/torch_compile_debug/run_2026_09_14_21_43_23_534976-pid_2123823/torchinductor/model__0_inference_0.0/output_code.py`：

```python
# 上述 GPU output_code.py:77（输入已提升到 tl.float32）
tmp3 = (tmp1 / tmp2)
tmp4 = tmp0 + tmp3
tl.store(out_ptr0 + (x0), tmp4, None)
```

NPU 修复后的原生成文件：

```python
# issues/REF-addcdiv-fma-codegen-native/evidence/FP16精度修复/
# 20260909-fixed/value-one/output_code.py:100
tmp3 = (tmp1 / tmp2)
tmp4 = tmp3.to(tl.float16)  # 保留 NPU eager 的 quotient 舍入边界
tmp5 = tmp4.to(tl.float32)
tmp6 = tmp0 + tmp5
tl.store(out_ptr0 + (x0), tmp6, x0mask)
```

| 项目 | GPU 新补证 | NPU 2026-09-09 修复原件 |
|---|---|---|
| 输入合同 | FP16、64×64、value=1 | FP16、64×64、value=1 |
| FX | div + add，无 mul/addcdiv 重融合 | div + add，无 mul/addcdiv 重融合 |
| codegen | 普通 FP32 除加，最后存回 FP16 | quotient 先舍入 FP16，再提升做加法 |
| 与本设备 eager | 位级一致、max error=0 | 位级一致、mismatch=0、max error=0 |
| 目标 FMA | 不命中 | 不命中，counter=0 |
| 性能 | 未测；正确性负命中邻接 | 未测；不能制造 FMA ON 路径 |

因此修复发生在 **NPU FP16 div lowering 的舍入边界**，不是“让 value=1 命中 FMA”。
修复前/后和近邻回归详见
[FP16 精度修复报告](../REF-addcdiv-fma-codegen-native/FP16精度修复报告.md)。

## 如何复核

```bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/review_t078_value_one.py \
  --check-current
```

该命令只校验原件、整包/逐文件哈希、日志和已审核合同，不导入 torch，不执行还原的 Python。
机器结论在 [增量复核记录](../../results/current/T-078/fp16_value_one_review_20260914.json)。
旧 comparison 已保存在 [历史快照](../../results/history/T-078/comparison-before-value1-reference-20260914.json)，
原 NPU `arm_result.json` 和修复前后 FX/IR/output_code 均未改写。
