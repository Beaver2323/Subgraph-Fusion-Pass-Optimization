# T-032：B2 第二批冗余规约 pass 结构与 NPU 编译

## 结论

本批覆盖 `fold_cast`、`fold_cat`、`fold_clone`、`fold_detach`。新增 8 个
device-independent 测试后，`test_dynamic_shape_fx_passes.py` 完整结果为 41/41；随后
8 个 default-backend NPU fresh worker 均功能正确。只有 `fold_cat` 的 positive 真正到达
目标 pass 并完成 2→1 个 cat 的改写，可进入单 pass paired 性能阶段；另外三条当前为
reachability-neutral/partial，不把前序 pipeline 的消除归功于目标 pass。

## 环境与方法

- PyTorch `2.14.0a0+git8e86e0a`，editable source 位于 Benchmark 工作区。
- torch_npu `2.14.0a0+git83cc452`，当前安装为 T-031 source-built wheel。
- CANN 9.0.1，物理设备 Ascend910B2；本批使用物理 NPU 1，运行前后均无外部进程。
- 每个 case 使用独立进程、debug/cache 目录、`torch.no_grad()`、dynamic compile 和
  default NPU backend；测试均从 `/home/z50063656/tmp` 启动。
- fresh Triton host launcher 仍使用已登记的 audit-only C++20/header shim，因此本批证明
  NPU 图与 kernel 运行能力，不提升正式无 shim 环境结论。
- 语义合同同时比较数值、shape、dtype、stride、相对每个输入的 storage alias、Python
  对象身份和 `requires_grad`。

## 结果

| pass | positive | negative | 语义 | 当前结论 |
|---|---|---|---|---|
| `fold_cast` | 同 dtype cast 在 pass 前已消失 | fp16→fp32 cast `1→1` | 两侧完整合同通过 | partial reachability；无可归因性能 |
| `fold_cat` | nested cat `2→1` | multi-user inner cat `2→2` | tensor/tuple 完整合同通过 | 功能通过；进入 T-033 性能 |
| `fold_clone` | clone→relu 在 pass 前已消失 | 图输出 clone `1→1` | 新 storage 语义保持 | partial reachability；无可归因性能 |
| `fold_detach` | detach→relu 在 pass 前已消失 | 直接 detach 输出整图旁路 | alias、对象身份、requires_grad 均匹配 | reachability-neutral |

全部 tensor 输出的 max/mean absolute error 均为 0。这里的“0”只作为数值证据；最终功能
判断还依赖完整语义合同。例如直接 detach 输出必须与输入共享 storage、不是同一 Tensor
对象且 `requires_grad=False`，这些条件本批均满足。

原始结果位于 `results/t032_b2_redundancy_compile_20260824/`，逐条状态已写入
`pass_evaluation_matrix.csv`。T-032 不包含性能数据，不把首次 compile+run 用作稳态结论。
