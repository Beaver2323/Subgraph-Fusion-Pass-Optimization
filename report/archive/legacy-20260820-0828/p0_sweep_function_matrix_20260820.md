# P0 代表覆盖功能矩阵（2026-08-20）

## 结论

在物理 NPU 2、fresh process、Inductor cache 关闭的条件下，default/addmm fusion 与 triton_experimental/mm_plus_mm 分别完成 8 个代表配置，共 16/16 `compile-correct`。每个配置都生成一份 debug artifact；addmm 的 8/8 `output_code.py` 出现 `torch.ops.aten.addmm.default`，mm_plus_mm 的 8/8 出现 `extern_kernels._mm_plus_mm`，没有 graph-break counter。

| cohort | 配置 | addmm/default | mm_plus_mm/experimental |
|---|---|---|---|
| dtype | fp16、bf16、fp32；shape-A/contiguous/static | 3/3 正确且目标触发 | 3/3 正确且目标触发 |
| shape | fp16；small、unaligned、large；contiguous/static | 3/3 正确且目标触发 | 3/3 正确且目标触发 |
| layout | fp16/shape-A/transposed/static | 1/1 正确且目标触发 | 1/1 正确且目标触发 |
| dynamic | fp16/shape-A/contiguous/dynamic + 第二组 replay | 1/1 两 shape 正确且目标触发 | 1/1 两 shape 正确且目标触发 |

shape profile 的 M/K/N 为：small `(32,64,48)`、shape-A `(192,256,320)`、unaligned `(191,255,319)`、large `(512,768,640)`。dynamic replay 将对应 M/K/N 各增加 8。

## 正确性与资源边界

- addmm 最大绝对误差：fp16 最大 0.0625；fp32 shape-A 为 `1.1444091796875e-05`；bf16 shape-A 为 0.5。bf16 差值来自 fused addmm 与 eager mm+add 的数值路径差异，逐元素 `rtol=atol=3e-2` 通过；仍需在最终精度覆盖中保留该证据，不能只报告“通过”。
- mm_plus_mm 的 8 个首 shape 与 dynamic replay 最大绝对误差均为 0。
- addmm 首次编译+执行范围为 12.928-14.250 s，编译期峰值 allocator 为 38,912-7,017,472 bytes。
- mm_plus_mm 首次编译+执行范围为 13.231-14.553 s，编译期峰值 allocator 为 49,664-8,786,432 bytes。
- 峰值在编译前 reset，包含编译、首跑和 dynamic replay，不解释成纯稳态 runtime 内存。

## 证据位置

- addmm：`results/p0_sweep_function_addmm_{dtype,shape,layout,dynamic}_20260820/`
- mm_plus_mm：`results/p0_sweep_function_mmplus_{dtype,shape,layout,dynamic}_20260820/`
- 每个目录的 `p0_gate_probe.json` 保存输入 shape/dtype/stride、容差、counter、observer、首次编译和峰值内存。
- 每个目录的 `debug/` 保存 transformed FX graph、Inductor IR 和 `output_code.py`。

## 结论边界与下一步

本轮只执行 current 模式，没有 pass-off 性能 baseline，因此仍不能把 16 个配置写成性能有益。下一步使用相同 cohort 运行 current/disabled、warmup 10、runs 100、3 轮交错 paired benchmark；性能只能在同 dtype/shape/layout/dynamic 配置内部比较。
