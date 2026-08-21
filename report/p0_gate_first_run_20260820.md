# P0 NPU Gate 首轮动态结果（2026-08-20）

## 结论

在 Conda `Pass`、Ascend 910B2、CANN 9.0.1 上，10 个正/负 case 分别使用 `default` 和 `triton_experimental` fresh process 运行，共 20 个组合，结果均为 `compile-correct`。所有组合都使用 `fullgraph=True`，eager/compiled 数值对齐，没有把 skip 或环境不可见计为成功。

功能可用不等于目标 pass 已触发。本轮目标 pass 结论为：

| family | `default` 正例 | `triton_experimental` 正例 | 负例 | 当前判断 |
|---|---|---|---|---|
| `addmm_fusion` | 触发，`mm + tensor bias -> aten.addmm` | 未触发，保留 `mm + add` | 两 backend 的 scalar 负例均未误触发 | backend gate 已确认 |
| `mm_plus_mm` | 未进入目标 fusion，变为 `mm + addmm`，2 个 extern call | 触发 `extern_kernels._mm_plus_mm`，1 个 extern call | 广播负例均未进入目标 fusion | backend 行为分化，experimental 已可用 |
| `pad_mm` | `shape_padding=True`，但未出现 pad/slice | `shape_padding=False`，未出现 pad/slice | 对齐负例未 pad | default 需进一步区分启发式未选中与结构不支持；experimental gate 已确认 |
| `pad_bmm` | `shape_padding=True`，但未出现 pad/slice | `shape_padding=False`，未出现 pad/slice | 对齐负例未 pad | 同上 |
| `pad_addmm` | `shape_padding=True`，但未出现 pad/slice | `shape_padding=False`，未出现 pad/slice | 对齐负例未 pad | 同上 |

因此，本轮不能把三个 pad family 记为“pass 可用”：它们只证明原始 mm/bmm/addmm 能正确编译。补充的 `force_shape_pad=True` 诊断也未触发，并已定位为上游 device gate 明确排除 NPU，见下方补充诊断。

## 环境与执行合同

- Python 3.11.15。
- PyTorch `2.14.0a0+git8e86e0a`，source head `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`。
- torch_npu `2.14.0a0+git83cc452`，source/runtime git version `83cc452480c3546fd5cccf853bfe3a360ce9dbfc`。
- Triton Ascend metadata `3.2.2`，source head `8bd9f380d2786002b84b5248f00838c26f900515`。
- 固定 NPU 0；执行前 NPU 0-6 无外部运行进程，避开有外部 Python 进程的 NPU 7。
- 每个 case/backend 使用独立 worker；设置 `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`，避免 cache hit 绕过 pass observer。
- 开启 `TORCH_COMPILE_DEBUG`，保留 transformed FX graph、Inductor IR、`output_code.py` 和 provenance mapping。
- 本轮没有使用 `--benchmark`；首次编译与运行为 13.674-20.358 秒，仅用于诊断，不用于 backend 性能排名。

## 正确性与资源摘要

- `compile-correct`：20/20。
- 最大绝对误差：0.0625；断言阈值为 fp16 的 `rtol=1e-2, atol=1e-2`，全部通过。
- 峰值 NPU allocator 记录范围：1,249,792-201,960,448 bytes。
- 所有用例 `fullgraph=True` 编译成功，counter 中未出现 graph break。
- 每个结果的 stderr、有效 config、counter delta、observer 汇总和 debug root 均写入对应 JSON。

## 关键证据

### addmm fusion

`default/addmm_fusion_positive` 的 transformed graph 只有 `torch.ops.aten.addmm.default`，pattern matcher 记录 1 次匹配、2 个节点；`triton_experimental` 对相同正例保留 `aten.mm.default + aten.add.Tensor`，目标 fusion match 为 0。scalar 负例在两个 backend 都保持 `mm + scalar add`。

这证明 `_disable_addmm_fusion_pass` 当前控制有效。现阶段没有理由直接手写 Triton addmm；应先对 `default` 的 fused vendor/ATen 路径与 experimental 的 mm+pointwise 路径做 paired benchmark。

### mm_plus_mm

`triton_experimental/mm_plus_mm_positive` 的 transformed graph 出现 `torch__inductor_fx_passes_post_grad_mm_plus_mm`，IR 和 `output_code.py` 最终调用 `extern_kernels._mm_plus_mm`，extern call 数为 1。`default` 正例则保留一次独立 `mm`，第二组矩阵乘与加法改写为 `aten.addmm`，extern call 数为 2。

广播负例没有进入 `_mm_plus_mm`。`default` 对广播 bias 仍可合法改写为 addmm，这属于另一个 pattern，不能算作目标 pass 误触发。

### pad families

三个 family 在 `default` 上的有效配置均为 `shape_padding=True, force_shape_pad=False`，在 experimental 上均为 `shape_padding=False`。六个正例/backend transformed graph 没有 `constant_pad_nd`、pad 或 slice，仍是原始 `aten.mm`、`aten.bmm`、`aten.addmm`。

`pad_bmm` 四个组合都有一次通用 pattern match，但对齐负例也同样出现，且 transformed graph 没有 padding，因此没有把该计数归因给 pad pass。

### pad 强制结构诊断补充

随后仅在 `default` backend 对三个 pad 正例显式传入 `force_shape_pad=True`。3/3 case 仍为 `compile-correct`，但 transformed graph 仍是原始 mm/bmm/addmm。observer 的 pattern active-config snapshot 证明 pass 执行期间确实为 `shape_padding=True, force_shape_pad=True`；编译结束后 config 恢复为 False 是 `torch.compile(options=...)` 临时 patch 的正常生命周期。

根因位于上游 `torch/_inductor/fx_passes/pad_mm.py`：`check_device()` 只接受 CUDA 或 XPU，NPU 在 `can_pad()` 中直接返回 False，尚未进入 `should_pad()` 的强制分支和 replacement。因此三个 pad family 当前在 NPU 上属于 `unsupported`，而不是“启发式认为不划算”。强制诊断原始 JSON 位于 `results/p0_gate_force_pad_20260820/p0_gate_probe.json`。

## 证据位置

- 哨兵：`results/p0_gate_smoke_20260820/p0_gate_probe.json`
- 首轮分 family JSON：`results/p0_gate_first_run_20260820/*/p0_gate_probe.json`
- debug artifacts：各 family 目录下的 `debug/<backend>/<case>/torch_compile_debug/`
- 对应矩阵：`report/pass_src_20260820/pass_evaluation_matrix.csv`

## 后续动作

1. 为 pad family 设计后端可注册的 device/capability gate，或 NPU 专用 pass；方案获批前不修改功能源码。
2. 先验证 NPU pad/slice + vendor GEMM 的正确性、额外内存和收益，再决定是否值得开放；手写 Triton 不是默认方案。
3. 对 `addmm_fusion` 和 `mm_plus_mm` 建立单 pass A/B，而不是仅比较 eager 与 compiled；统一 warmup 10、runs 100，并记录 kernel 数和峰值内存。
4. P0 结构结论稳定后，进入 P1 NPU custom/DVM/MLIR 批次。
