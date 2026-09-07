# T-057 GELU `approximate` 正确性审计（2026-08-26）

## 结论

installed P-013 wheel 中，`triton_experimental` GELU decomposition 对
`approximate="none"` 和 `"tanh"` 生成完全相同的 sigmoid 近似图；从 PyTorch API/CPU
reference 看，`none` 的 forward/backward 合同失败。P-017 已在 current source 中完成最小修复，
源码覆盖验证恢复了 `none` 的 erf/CDF/PDF 语义，并保留 `tanh` 原路径。因此要区分三层 verdict：

- installed P-013 相对当前 eager NPU：两模式 forward/backward 均在严格容差内，
  `eager-compatible`；
- installed P-013 相对 PyTorch `approximate` 合同：`none` 被错误折叠成 tanh approximation，
  `upstream-contract-failed`；
- P-017 current source：FP32/FP16/BF16 的 `none/tanh` 均被 experimental Triton 承接，
  `none` 对 CPU reference 通过，非法参数合同通过，状态为
  `source-verified-wheel-pending-shared-diff`。

这不是缺少 GELU kernel，也不是 fallback。目标 op 在 decomposition 阶段被展开成 pointwise，最终
进入一个 experimental Triton kernel。修复应让 Python decomposition 按参数分支并复用已有
`erf`/pointwise codegen，不需要手写 Triton GELU；现有 codegen 能生成
`libdevice.erf/exp`。

## 运行边界

- 环境：`/home/z50063656/Benchmark/env.sh`，installed P-013 wheel；
- backend：两个 fresh process，显式 `triton_experimental`；
- 设备：运行前确认空闲的物理 NPU 1；cwd `/home/z50063656/tmp`；
- 输入：FP32、4097 点 dense `[-6, 6]`，grad 为 4097 点 `[-1.5, 2]`；
- worker：`t057_gelu_approximate_probe.py`，SHA256 `753b21e1...6bc4b`；
- P-016 隔离：两个 worker 都显式 `elide_int_float_int=False`；
- 调试：独立 cache、`TORCH_COMPILE_DEBUG=1`、audit launcher shim。

有效 v2 原始结果：

| mode | result JSON SHA256 | generated code SHA256 |
|---|---|---|
| none | `fd3df5e3...ef0e` | `f03bba4e...3cea` |
| tanh | `4a462b95...8af1` | `ef0b8def...34b` |

初版结果没有 CPU reference，已原样保留但不作为最终判定；v2 增加 CPU/upstream 第二参考层和
精确 kernel-call marker。

## 动态结果

### 当前 eager NPU 参考

NPU eager 的 `none` 与 `tanh` 在本轮 forward/backward 都是逐位相同。compiled 两模式相对
requested NPU eager 的误差也相同：

| 输出 | 最大绝对误差 | `rtol=1e-5, atol=1e-6` |
|---|---:|---:|
| forward | `4.76837158203125e-7` | 通过 |
| backward | `3.4570693969726562e-6` | 通过 |

因此只使用 eager NPU 作为 oracle 会得到“两模式都可用”的结论，并看不到参数被忽略。

### CPU/upstream 合同参考

相同 FP32 输入上，CPU 的 two-mode 差异为：

| 输出 | none/tanh 最大绝对差 |
|---|---:|
| forward | `0.0004733968526124954` |
| backward | `0.0007350444793701172` |

compiled `none` 相对 CPU `none` 的最大绝对误差分别为
`0.0004734992980957031`、`0.0007351040840148926`，两者都未通过严格 allclose；数值量级和
CPU none/tanh 差一致。compiled `tanh` 相对 CPU `tanh` 的最大绝对误差仅为
`4.76837158203125e-7`、`3.4570693969726562e-6`。这证明错误方向是把 `none` 固定成 tanh
近似，不是普通随机数值漂移。

## 生成图第一现场

`none` 的 debug 入口为：

`results/t057_gelu_none_fp32_v2_20260826/debug/torch_compile_debug/`
`run_2026_08_26_22_16_05_258789-pid_3579673/torchinductor/`
`model__0_inference_0.0/output_code.py`

`tanh` 对应 run 为 `run_2026_08_26_22_17_22_492211-pid_3608469`。两份 transformed FX 与
kernel body 完全采用同一结构：三次方、`0.044715`、`1.5957691216057308` 和
`tl.sigmoid`；都没有 `erf()` 或 `tl.tanh()` 调用。forward/backward 融合成同一个 Triton
kernel。

`aten.gelu`/`gelu_backward` 在 experimental decomposition table 中被 `_override_gelu_decomp()`
直接替换；这里没有 ACLNN/CPU fallback，也不是 lowering blacklist 阻断。

## eager NPU 为什么也不区分

op-plugin 源码能把 `none/tanh` 映射为不同参数，并在新接口调用 `aclnnGeluV2`/
`aclnnGeluBackwardV2`；但当前兼容分支仍可能调用不带 approximate 参数的旧 `aclnnGelu`/
`aclnnGeluBackward`。本轮动态结果与“旧路径忽略参数”一致。这个 eager/CANN 兼容问题超出
Inductor pass 最小修复范围，应另行交给 API/算子一致性流程，不能作为 experimental 固定忽略
参数的理由。

## P-017 修复边界与实施

在 `_override_gelu_decomp()` 内同步上游语义：

- `tanh` 保持现有 sigmoid 等价近似，避免扩大已通过路径；
- `none` forward 使用 `x*0.5*(1+erf(x/sqrt(2)))`；backward 使用 CDF/PDF exact 公式；
- 非法字符串显式报错；
- 不改 eager/op-plugin、fallback、lowering、codegen 或 Triton 源码。

实际修改只落在
`src/torch_npu/torch_npu/_inductor/decomposition.py:_override_gelu_decomp()`。修改后文件 SHA256 为
`059ec0db...02d7d5b`；P-014 的 erfc cleanup 仍保留。当前源码覆盖 runner SHA256 为
`38bfc907...0ec416`。

### 源码契约探针

`t057_p017_source_contract_probe.py` 直接从 current source AST 提取 override，不导入共享源码树中的
`torch_npu`，并在 CPU 进程临时替换 decomposition table 后恢复原条目。探针 SHA256
`7627b8bf...6318f`，结果为：

- `none` FP64 forward/backward 相对 PyTorch reference 最大绝对误差 `0/2.220446e-16`；
- `tanh` 最大绝对误差 `8.881784e-16/1.243450e-14`；
- forward/backward 的非法 `approximate="invalid"` 均抛出预期 RuntimeError；
- 总判定 `passed=true`。

### NPU source-overlay 结果

六个 dtype/mode fresh process 均使用 current-source override、显式关闭 P-016 目标 pass，并进入
experimental Triton kernel：

| dtype | mode | 对 CPU forward 最大绝对误差 | 对 CPU backward 最大绝对误差 | kernel 特征 |
|---|---|---:|---:|---|
| FP32 | none | `1.430511e-6` | `4.768372e-7` | `libdevice.erf`，无 sigmoid |
| FP32 | tanh | `4.768372e-7` | `3.457069e-6` | `tl.sigmoid`，无 erf |
| FP16 | none | `6.103516e-5` | `1.525879e-5` | `libdevice.erf`，无 sigmoid |
| FP16 | tanh | `3.051758e-5` | `3.814697e-6` | `tl.sigmoid`，无 erf |
| BF16 | none | `1.525879e-5` | `2.384186e-7` | `libdevice.erf`，无 sigmoid |
| BF16 | tanh | `1.525879e-5` | `3.814697e-6` | `tl.sigmoid`，无 erf |

FP32 `none` 以 `rtol=1e-5, atol=1e-6` 对 CPU forward/backward 均通过；FP16/BF16 分别以
`2e-3/2e-3`、`2e-2/2e-2` 通过。`tanh` 路径未被修改，对当前 eager NPU 回归通过；其 FP32
backward 相对 CPU 的最大绝对误差虽只有 `3.457e-6`，但因接近零位置的相对误差，原严格
`1e-5/1e-6` allclose 标志为 false，不能表述为逐点严格 CPU allclose。

源码结果文件哈希：

| dtype/mode | result JSON SHA256 | generated code SHA256 |
|---|---|---|
| FP32 none | `44393615...acc6` | `a1de6411...b632` |
| FP32 tanh | `ecef883c...010d` | `b0ded19b...c267` |
| FP16 none | `c6c01c9b...7824` | `24f1972a...d470` |
| FP16 tanh | `182d8874...c285` | `43515eb7...428f` |
| BF16 none | `88908f42...2531` | `56570256...a39` |
| BF16 tanh | `9d8e7258...b131` | `fe2935d0...cfa0` |

首次 FP32 tanh source-overlay 重试前的一轮在把输入搬到 NPU 前因设备子进程启动超时
`507033/E39007` 结束，目录中没有 `result.json`，仅作为环境失败保留；确认 NPU1 健康空闲后的
retry1 成功，不能把前一轮记成 P-017 功能失败。

### 独立 wheel 收口（2026-08-27）

从相同基线创建 detached worktree，保留已验证的 P-014 erfc cleanup，并只叠加 P-017 15 行
GELU 修复。完整构建的 wheel SHA256 为
`5b928d5a4f219c5f3c744feb90e85685c67b676de154c12b6d27229572c27b13`；source、wheel 内和
独立 venv 安装态的 `decomposition.py` SHA256 均为
`059ec0db0a6d71dfcb5cbe1da606d4aa5f6ead4c130abfdae8d35bf0b5027d5b`。

无 source overlay 的 FP32/FP16/BF16 × none/tanh 六组 NPU 验证 6/6；非法 approximate 的
forward/backward compile 2/2 抛出预期消息；P-014 近邻 1/1。阶段状态升级为
`verified-installed-wheel`。共享 Benchmark 安装态仍为 P-013，未被覆盖。错误的历史 installed
`none` 不做性能测试；修复以正确性为目标，现有 Triton codegen 已能承接，不进入手写 kernel 路线。
