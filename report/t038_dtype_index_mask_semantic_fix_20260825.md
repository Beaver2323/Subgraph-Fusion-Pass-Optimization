# T-038 dtype/index/mask pass 语义修复与 NPU 功能报告

日期：2026-08-25  
范围：`dtype_optimal_pass`、`fold_iota_arithmetic_pass`、
`broadcast_const_mask_compress`  
环境：PyTorch `2.14.0a0+git8e86e0a`、torch_npu
`2.14.0a0+git83cc452`、CANN 9.0.1、Ascend910B2

## 结论

三条 pass 原实现都存在“普通样本数值看似相等，但可观察语义已经改变”的边界。本轮已用
保守 capability gate 修复：危险场景保持原图，安全正例继续触发。源码态与安装态 FX 测试
均为 67/67；9 个 NPU fresh worker 的目标 pass、图门禁、设备执行和完整输出合同全部通过。

NPU worker 使用 T-022 已登记的 audit-only C++20/CANN header launcher 垫片，因此本报告能
证明 pass rewrite 和生成 device kernel 的开发态功能可用，但不能证明当前组合的无垫片
fresh launcher 环境已兼容。性能尚未 paired 测量，三条 pass 不能因本报告直接标记
`supported-beneficial`。

## 修复内容

| pass | 修复前 blocker | 当前保护 | 安全正例 |
|---|---|---|---|
| `dtype_optimal_pass` | int64 arange 直出变 int32；float32→int64 可能截断并变 dtype | 仅比较闭包允许降级；float32 不再进入 `.to(int64)→int32` | arange→comparison、int32→int64→comparison |
| `fold_iota_arithmetic_pass` | `cmp(a-b,0)→cmp(a,b)` 在 Inf/NaN、定宽整数溢出时不等价 | 停用无 range/dtype 证明的 cmp-sub 子改写 | 保留 int32 范围内 iota downcast 与常量 CSE |
| `broadcast_const_mask_compress` | 小 mask 经 where 广播后的大输出被改回小 shape | 只有 mask shape 与 where output shape 静态完全相等才压缩 | equal-shape 0/1 mask |

产品修改限定在
`torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py` 与
`test/_inductor/test_dynamic_shape_fx_passes.py`，没有修改 PyTorch、Triton、C++ 或新增 kernel。
手写 Triton不能修复 dtype、overflow 或 shape 合同，因此本轮不做替身算子。

## 构建与测试

- T-036 旧 wheel 已归档为
  `artifacts/torch_npu_t036_before_t038_dtype_mask_fix.whl`，SHA256 为
  `d745cf3afd6a2859a68d6c31dd02a46498264e82dedff34d726c2be2609c6b9d`。
- T-038 wheel 从当前 torch_npu 源码构建，SHA256 为
  `dffad49056538fc4250b444b2c40a619db3b0897b00f8906f53757a857b167d8`，并按
  `pip install --no-deps --force-reinstall` 安装。
- 源码 namespace 隔离测试 67/67；安装态测试 67/67。两个修改文件此前的
  `py_compile` 与 lintrunner 均通过。

## NPU 结果

原始 JSON 位于
`results/t038_dtype_index_mask_compile_audit_shim_20260825/`。所有输出均严格逐元素相等，
mismatch count 为 0；shape、dtype、stride、`requires_grad` 和输入 alias 合同一致。

| profile | pass 前→后 | 输出合同 |
|---|---|---|
| arange 直出 | dtype int64→int64 | int64 `(4096)`、stride `(1)`，严格相等 |
| arange comparison | dtype int64→int32 | bool `(4096)`，严格相等 |
| float32→int64 直出 | dtype int64→int64 | `±3,000,000,000` 保留，严格相等 |
| int32→int64 comparison | dtype int64→int32 | bool `(3)`，严格相等 |
| safe iota comparison | iota int64→int32 | bool `(4096)`，严格相等 |
| Inf cmp-sub | sub `1→1`、ge `1→1` | bool `(4096)`，严格相等 |
| int32 overflow cmp-sub | sub `1→1`、ge `1→1` | bool `(4096)`，严格相等 |
| equal-shape mask | where `1→0`、full `2→0` | float32 `(32,4096)`、stride `(4096,1)` |
| broadcast mismatch | where `1→1`、full `2→2` | float32 `(32,4096)`、stride `(4096,1)` |

首次 compile+run 为约 17.19–30.42 秒，只记录环境与 fresh compile 成本，不用于 pass 性能
判定。

## 失败和中性尝试

1. 直接把 torch_npu 源码根加入 `PYTHONPATH` 时，backend 自动加载因源码树没有已编译
   `_C` 失败；禁用自动加载后，测试 config stub 又依次缺 `is_ascend950` 和
   `enable_fused_matmul_relu`。补齐两个仅测试用默认值后 67/67 通过。这些都发生在测试收集前。
2. 首批 4 个 NPU worker 已观察到正确图门禁，但 Triton launcher 从 editable PyTorch 的空
   include view 查找 `ATen/ATen.h` 失败，未执行 kernel；第五项在确认重复根因后中断。
3. 前置 wheel headers 后，Triton仍固定用 C++17 编译 PyTorch 2.14 headers，heuristic 对每个
   config 重试 GCH。force-disable cache 与正常 fresh cache 两轮都被主动中断；后者证明根因
   是 C++ 标准合同，不是单纯缓存问题。
4. 最终只复用 T-022 的 audit-only wrapper：C++17→C++20、前置两个旧 CANN 缺失类型，并
   `TRITON_DISABLE_PRECOMPILE=1`。9/9 完成，但正式环境缺口保持开放。
5. broadcast mismatch 生成代码出现 Triton 关于 int8 condition 的未来弃用 warning；当前
   kernel 正确执行且严格匹配。它属于 lowering/runtime 后续兼容观察项，不是本 pass guard
   的当前功能失败。

## 后续结果

T-039 已按上述计划完成 18/18 fresh paired worker。`dtype_optimal_pass` 与
`fold_iota_arithmetic_pass` 的三轮中位 p50 分别改善 52.06%/55.78%，为
`supported-beneficial-development-audit-shim`；mask pass 只改善 0.30%、task/显存不变，
为 `supported-neutral-development-audit-shim`。详细方法、profiler 与 replacement 决策见
`t039_dtype_index_mask_performance_20260825.md`。性能结果没有恢复本报告已证伪的不安全改写。
