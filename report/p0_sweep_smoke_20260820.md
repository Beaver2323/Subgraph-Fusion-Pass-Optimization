# P0 覆盖扩展哨兵（2026-08-20）

## 结论

扩展后的 `run_p0_gate_probe.py` 已在物理 NPU 2 上完成旧行为回归和两个高信息量新配置。旧 shape-A 回归 4/4 `compile-correct`，目标图与首轮报告一致；两个 dtype/layout/dynamic 新哨兵 2/2 `compile-correct`，第二组 shape replay 也通过正确性，并保留目标 pass。

| case / backend | 新配置 | 首次输入 → replay | 目标图 | 结果 |
|---|---|---|---|---|
| addmm fusion / default | bf16、small、transposed、dynamic | `(32,64)@(64,48)` → `(40,72)@(72,56)` | 符号 shape 的 `torch.ops.aten.addmm.default` | 两次正确；replay 最大绝对误差 0.125，在 bf16 `rtol=atol=3e-2` 下通过 |
| mm_plus_mm / triton_experimental | fp32、unaligned、transposed、dynamic | `(191,255)@(255,319)` → `(199,263)@(263,327)` 两组 | 符号 shape 的 `extern_kernels._mm_plus_mm` | 两次正确；replay 最大绝对误差 0 |

这里的“transposed”不是标签模拟：输入由反向 storage shape 创建后执行 `transpose(-2,-1)`；addmm 两个矩阵 stride 分别为 `(1,32)`、`(1,64)`，mm_plus_mm 首组 stride 为 `(1,191)`、`(1,255)`。dynamic 模式生成的 `output_code.py` 使用符号维度，并真正执行第二组不同 M/K/N 输入。

## 旧行为回归

fp16/shape-A/contiguous/static 在 default 和 triton_experimental 上共 4 个正例全部通过：

- default/addmm：单个 `torch.ops.aten.addmm.default`。
- experimental/addmm：保持 `mm + add`。
- default/mm_plus_mm：`extern_kernels.mm + aten.addmm`，没有目标 fusion。
- experimental/mm_plus_mm：单个 `extern_kernels._mm_plus_mm`。

这证明 schema 2、配置路径和 debug 目录扩展没有改变旧默认 gate 行为。

## 证据位置

- 旧行为 JSON：`results/p0_sweep_smoke_20260820/p0_gate_probe.json`
- addmm 新配置：`results/p0_sweep_new_smoke_addmm_20260820/p0_gate_probe.json`
- mm_plus_mm 新配置：`results/p0_sweep_new_smoke_mmplus_20260820/p0_gate_probe.json`
- generated code：以上目录的 `debug/<backend>/<case>/<config>/round_1/current/torch_compile_debug/`

## 边界与下一步

本轮没有 benchmark，因此只证明扩展 harness、正确性、动态 replay 和目标图有效。下一步按非笛卡尔 cohort 执行 dtype、shape、layout、dynamic 的 current 功能矩阵；全部健康后，才对相同配置运行 current/disabled 三轮 paired benchmark。
