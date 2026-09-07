# T-057 `int→float→int` 边界正确性审计（2026-08-26）

## 结论

`triton_experimental` 默认开启的 `_elide_int_float_int_roundtrip_pass` 不满足通用
`torch.compile` 正确性合同，当前 verdict 为
`correctness-failed-default-on`：

- Float32 的 pass ON 在 `±(2**24+1)` 返回原整数，而 eager 经 Float32 舍入后相差 1；
  pass OFF 与 eager 完全一致，完成了单 pass 归因。
- FP16/BF16 的 pass ON 同样删掉整条转换并直接返回输入，在各自精确整数边界外与 eager
  相差 1。
- 三个 dtype 的 pass ON 都把 eager 的新 tensor 改成输入 alias。即使数值恰好相等，直接
  输出场景仍有可观察语义变化。
- FP16/BF16 的 pass OFF 另有独立现象：FX/IR 保留低精度转换，但生成 Triton 代码把中间
  转换提升成 `tl.float32`，所以没有复现 eager 的 FP16/BF16 舍入。它不能用来反证 pass
  正确，也不能与 Float32 的单点归因混为一谈。

正确的第一步是默认关闭该 pass；不应手写另一个 Triton kernel 来复制“删除 cast”。未来若
重新开放，只能接受带中间 dtype 精确域证明和输出 alias 保护的保守 cohort。

## 运行边界

- 环境：`/home/z50063656/Benchmark/env.sh`，当前 P-013 installed wheel；
- 设备：每次运行前确认无进程的物理 NPU 1；
- cwd：`/home/z50063656/tmp`；
- backend：每个 dtype/mode 都是 fresh process，显式
  `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`；
- pass A/B：`elide_int_float_int=True/False`；
- 审计 launcher：既有 `t022_launcher_cc_wrapper.sh`、site-packages Torch headers、
  `TRITON_DISABLE_PRECOMPILE=1`；因此本报告只关闭 pass 正确性归因，不关闭正式无 shim
  launcher 环境缺口；
- worker：`t057_int_float_int_boundary.py`，SHA256
  `9d6e46b3f3f190f123bd64b36ad2ef11e6c61fc937e6f28641557327af2d494f`。

所有输入保持在 Int32 可表示范围内，排除了 experimental 的 Int64 boundary downcast 这一
并行变量。

## 六组结果

| 中间 dtype | pass | eager/compiled 精确相等 | mismatch index | 最大整数差 | compiled alias input | generated code |
|---|---:|---:|---|---:|---:|---|
| Float16 | ON | 否 | 0, 12 | 1 | 是 | 直接 `return (arg0_1,)` |
| Float16 | OFF | 否 | 0, 12 | 1 | 否 | 单 Triton kernel，但中间为 `tl.float32` |
| BFloat16 | ON | 否 | 0, 2, 3, 9, 10, 12 | 1 | 是 | 直接 `return (arg0_1,)` |
| BFloat16 | OFF | 否 | 0, 2, 3, 9, 10, 12 | 1 | 否 | 单 Triton kernel，但中间为 `tl.float32` |
| Float32 | ON | 否 | 0, 8 | 1 | 是 | 直接 `return (arg0_1,)` |
| Float32 | OFF | 是 | 无 | 0 | 否 | 单 Triton kernel，`i32→f32→i32` |

eager 的三个输出都不是输入 alias；所有 pass ON 输出都是输入 alias。Float32 OFF 同时满足
数值和 alias 合同，因此 Float32 是最干净的 ON/OFF 归因证据。

原始 JSON SHA256：

- Float16 ON：`613b1c49...dc42`；OFF：`838e1adb...251`；
- BFloat16 ON：`344bd3b4...e9d`；OFF：`d1e1e137...2fa`；
- Float32 ON：`759efabc...9772`；OFF：`36e06301...97c2`。

## 生成图第一现场

按图模式失败排查要求，已检查六次运行的 `output_code.py`、
`fx_graph_transformed.py` 和 `ir_pre_fusion.txt`。

### pass ON

Float16 ON 的第一现场为：

`results/t057_int_float_int_float16_on_20260826/debug/torch_compile_debug/`
`run_2026_08_26_21_49_33_440023-pid_3051104/torchinductor/`
`model__0_inference_0.0/output_code.py`

Float32/BFloat16 ON 结构相同：transformed FX 只剩输入，wrapper 直接
`return (arg0_1,)`，没有 Triton kernel，且 `experimental_marker=false`。这不是“目标算子
fallback”，而是 post-grad pass 已在 lowering/codegen 前消除了两个 convert。

### pass OFF

Float16 OFF 的第一现场为：

`results/t057_int_float_int_float16_off_20260826/debug/torch_compile_debug/`
`run_2026_08_26_21_50_08_454634-pid_3064083/torchinductor/`
`model__0_inference_0.0/output_code.py`

BFloat16 OFF 的对应 run 为
`run_2026_08_26_21_52_24_577686-pid_3122762`。两者 transformed FX 和 IR 都保留
`prims.convert_element_type(int64→lowp→int64)`，但最终代码均为：

```python
tmp0 = tl.load(...)
tmp1 = tmp0.to(tl.float32)
tmp2 = tmp1.to(tl.int32)
tl.store(..., tmp2, ...)
```

`prims.convert_element_type` 位于 torch_npu `LOWERING_OVERRIDE_OP`，不在 NPU fallback
列表；因此它已进入 NPU Triton lowering，并非 ACLNN/CPU fallback。IR 还记录目标为
Float16/BFloat16，语义变化发生在 Triton compute-type codegen：默认
`use_compute_types=True` 把低精度 store type 提升成 Float32。Float32 OFF 的同类代码恰好能在
`2**24` 边界产生正确舍入，所以数值与 eager 完全相等。

## 根因分层

### 根因 A：目标 pass 无值域证明

`fx_passes.py` 的 tracer 只记录布尔量 `saw_float`，接受 FP16/BF16/FP32/FP64 全部中间
dtype；源码注释承认 Int64→Float32 只在有限范围精确，但默认 gate 仍为 ON。对任意 runtime
输入，静态 dtype 相同不代表 round-trip 值相同。

### 根因 B：目标 pass 破坏输出 alias

pass 用原始整数 source/view 替换外层整数 cast。原 cast 会分配新 storage；替换节点如果进入
graph output，就把新 tensor 改成输入 alias。本轮三个 pass-ON 结果全部直接复现该问题。

### 独立现象 C：低精度 convert 融合上浮

pass OFF 时 NPU lowering 保留了 conversion IR，但默认 compute-type codegen 不强制中间
FP16/BF16 materialization。该行为需要独立 issue/capability 评估；不能靠重新打开不安全 pass
掩盖，也不能在 P-016 的最小安全修复里顺带改通用 lowering。

## P-016 修复边界

第一阶段只把 `elide_int_float_int` 默认值改为 False，保持显式 opt-in 能力，阻止默认静默错误。
它是 Python config 修复，不需要手写 Triton，也不需要重编 PyTorch；最终安装验证仍应由同一
torch_npu source 构建 wheel 后 `--no-deps` 安装。

未来重新开放默认值前，至少需要：

1. tracer 记录每个实际中间 float dtype，而不是 `saw_float`；
2. 没有 runtime range proof 时，只允许整个整数 dtype 域都可精确表示的组合，例如
   Int8/UInt8→任意普通 float、Int16→Float32/64、Int32→Float64；Int64 不得仅凭 dtype 放行；
3. outer cast 或其 view closure 可到达 graph output 时保持原图，或者显式物化新 storage；
4. 覆盖正负边界、slice/select/scatter/view 路径、输出 alias、非连续 layout 与 dynamic shape；
5. 正确性关闭后才在 DIFM 真实 cohort 做 fresh-process paired 性能，并把假设的 index 值域写成
   可验证合同。

当前不做性能测试：错误结果更快没有工程意义。

## P-016 第一阶段实施结果

已按登记边界完成 source-only 最小修复：

- `triton_experimental/config.py` 只把默认值改为 False，并注明这是安全证明完成前的 opt-in；
  修改后 SHA256 `f33deebb...4c73`；
- `triton_experimental/fx_passes.py` 只修正“默认 ON”和“embedding indices always safe”的误导
  注释，不改变 tracer/rewrite 实现；修改后 SHA256 `bf4f2b89...7ec2`；
- `py_compile` 与 `git diff --check` 通过；
- source gate probe `t057_p016_source_gate_probe.py` 在关闭 backend autoload 的 CPU 进程中通过：
  默认值 False、默认 installer 保持原 post-pass identity、显式 opt-in 能注册、测试后恢复原
  post-pass。probe SHA256 `e6bca70a...2d4`。

T-056 的可复现 generator 和三张 CSV 已同步：config inventory 现在记录默认 False，TE-FX-001
记录 `correctness-failed-source-default-disabled-wheel-pending`。当前 installed wheel 仍是 P-013，
没有被 source overlay 污染；由于共享 torch_npu tree 同时有其他未安装 diff，P-016 不直接构建
或安装 wheel。
