# 2026-09-09 FP16 当前源码真机再验证

> 记录时间：2026-09-09 00:43:01 CST（UTC+08:00）  
> 设备：Ascend 910B2（物理卡 2，进程内 `npu:0`）  
> 后端：`triton_experimental`  
> PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`

## 结论

- `value=0.3/2/7.7`：当前 NPU 专属 addcdiv lowering 均与 NPU eager 位级一致，
  `addcdiv_fma_fused=1`，生成代码有两次显式 FP16 舍入且没有 FMA/div_rn。
- integer-self 与 tensor-valued value 两个 guard 均保持 `counter=0`。
- `value=1`：结构合同正确，仍是 `div -> add` 且 `counter=0`；但普通编译路径没有恢复
  NPU eager 的除法后 FP16 舍入，导致 1176/4096 个元素不同，最大绝对误差 `0.015625`。
- 因此不能恢复 `add(div) -> addcdiv` 来掩盖误差；该分支必须作为普通低精度
  `div -> add` correctness regression 独立诊断和修复。

## 实际执行入口

所有进程均从 `/home/z50063656/tmp` 启动，并在导入 `torch` 前选择后端：

```bash
source /home/z50063656/Pass/activate_pass.sh

export ASCEND_RT_VISIBLE_DEVICES=2
export SET_NPU_DEVICE=0

python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-addcdiv-fma-codegen-native/run_fp16_source_fix.py \
  --device 2
```

`value=1` 邻接分支另用一个 fresh process 执行：

```bash
python \
  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-addcdiv-fma-codegen-native/verify_lowp_three_arm.py \
  --dtype float16 \
  --arm re-fused \
  --value 1 \
  --source-fix
```

## 首次分歧代码

当前 `value=1` 的 transformed FX 保持预期结构：

```python
# value-one-failure/debug/fx_graph_transformed.py
div = torch.ops.aten.div.Tensor(arg1_1, arg2_1)
add = torch.ops.aten.add.Tensor(arg0_1, div)
```

最终 Triton 代码把 FP16 输入提升到 FP32 后连续执行除法与加法，直到最终 store 才落回 FP16：

```python
# value-one-failure/debug/output_code.py
tmp0 = tl.load(in_ptr0 + (x0), x0mask).to(tl.float32)
tmp1 = tl.load(in_ptr1 + (x0), x0mask).to(tl.float32)
tmp2 = tl.load(in_ptr2 + (x0), x0mask).to(tl.float32)
tmp3 = (tmp1 / tmp2)
tmp4 = tmp0 + tmp3
tl.store(out_ptr0 + (x0), tmp4, x0mask)
```

这里缺少 `tmp3.to(tl.float16).to(tl.float32)`，是当前 compiled 与 NPU eager 首次可见的
舍入边界差异。

## 文件导航

- 三个有效 `value!=1` 结果：`value-0p3/`、`value-2p0/`、`value-7p7/`；
- 两个 guard：`guards/`；
- 新回归与完整 FX/IR/codegen：`value-one-failure/`；
- 仓库级机器可读摘要：
  `results/current/AU-post-grad-fuse-addcdiv-to-fma/fp16_source_revalidation_20260909.json` 与
  `value_one_device_failure_20260909.json`。
