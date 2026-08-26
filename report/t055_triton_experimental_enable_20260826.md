# T-055 triton_experimental 入口与隔离基线（2026-08-26）

## 结论

当前源码构建并以 `--no-deps` 安装的 torch_npu wheel 中，
`triton_experimental` 的三种选择入口都能在 Ascend 910B2 上完成 pointwise 编译、执行和
wrapper marker 校验；原 T-055 的 12 个 experimental 参数化用例全部通过。一个同进程
default/experimental 隔离用例失败，根因为从 experimental 回切 default 时重复注册
`aten.erfc.default` decomposition。T-055 因此记为
`entry-supported-isolation-blocked`，不能把 12/13 写成完整可用。

该失败与 pass 数值、NPU lowering 或 Triton device kernel 无关，当前也不需要手写 Triton
替身。P-014 将先修复 backend loader 的 decomposition 重入合同，再重跑同一 13 项基线。

## 运行合同

| 项目 | 值 |
|---|---|
| 环境 | `/home/z50063656/Benchmark/env.sh`，Conda `benchmark-py311` |
| PyTorch / torch_npu | `2.14.0a0+git8e86e0a` / `2.14.0a0+git83cc452` |
| 当前安装 wheel | P-013 source-built wheel，SHA256 `3909fd649d777b8dfd393342da0ff2b88c5cce2ef219f0d103d063af4c2d4989` |
| 设备 | 运行前后进程表为空的物理 NPU 1，Ascend 910B2 |
| cwd | `/home/z50063656/tmp` |
| cache | 每次重试独立的 Inductor/Triton/debug 目录，`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| launcher | 已登记的 audit-only T-022 C++20/header wrapper；不能代表正式无 shim launcher 已修复 |
| test snapshot | `test_triton_experimental_enable.py` SHA256 `6ec3665587e35771935fc5d2d3718feb892368bf2332023d2732d8437c4b583f` |
| runner snapshot | `t055_triton_experimental_enable_runner.py` SHA256 `7968f8638b6ba68620e0480e811d46af08b96733a635b50980df3815f167b3bc` |

测试文件在原始运行结束后被共享工作区中的另一项工作扩展了两个 provenance 用例，同时
`triton_experimental/codegen/triton.py` 也出现未安装进当前 wheel 的对应修改。本轮不覆盖、
不归因这些修改；有效复测只点名原 T-055 的 `TestTritonExperimentalEnable` 和
`TestTritonExperimentalIsolation`，没有运行新增 provenance 类。

## 结果分层

### 原始 installed/no-shim 运行

原始 13 项运行在进入有效 backend 判定前暴露两个启动合同：

1. 四个 global-config 用例直接访问尚未惰性注入的
   `torch._inductor.config.npu_backend`；
2. 其余编译在 fresh launcher 因 editable PyTorch include view 缺少 `ATen/ATen.h` 失败。

这两个结果是环境/bootstrap 证据，不是 experimental pass verdict。原始日志保留在
`results/t055_triton_experimental_enable_20260826/test.log`。

只读探针确认：`import torch_npu` 后 config key 仍不存在；首次调用
`torch._inductor.config.get_config_copy()` 后 `npu_backend=default` 被注入。因此审计 runner
只做这一次 prime，并使用既有 T-022 launcher wrapper；没有修改产品源码或环境包。

### 中性重试

- `...shim_retry_20260826/test.log`：runner 初版没有把测试文件目录加入 `sys.path`，在收集前
  因 `ModuleNotFoundError: testutils` 退出；未运行用例。
- `...shim_retry2_20260826/test.log`：修正 runner 后在受限沙箱内枚举 13 项，但设备计算节点
  不可访问，13 项都在创建 NPU tensor 时以 `aclInit 507008` 退出；未进入编译。

二者均为启动/执行层中性尝试，不计入产品成功率。

### 有效 NPU 复测

沙箱外、空闲物理 NPU 1 上运行 13 项，耗时 251.311 秒：

- experimental options 入口：4/4 通过；
- experimental global config 入口：4/4 通过；
- experimental environment 入口：4/4 通过；
- 同进程 default→experimental 隔离：0/1，default 首段在 loader 中失败；
- 合计：12/13，通过项同时完成 eager/compiled equality 与 experimental wrapper marker 校验。

有效日志和 debug 根目录为
`results/t055_triton_experimental_enable_shim_retry3_20260826/`。

## 图模式失败诊断

按图模式诊断合同，本轮运行前已打开 `TORCH_COMPILE_DEBUG=1`。debug run 中只有成功完成的
`model__0_inference_0.0` 至 `model__11_inference_11.11`，共 12 份 `output_code.py`；失败的
隔离用例没有 `model__12` 或 `output_code.py`。这是因为失败发生在
`_NpuBackendScope` 触发 default loader 后、FX lowering/codegen 之前：

1. `_InductorNpuRegistry` 发现已加载 backend 从 `triton_experimental` 变为 `default`；
2. `_load_backend()` 调用 `restore_inductor_baseline()`，再进入 `_load_triton_backend()`；
3. `_register_triton_decompositions()` 再次执行
   `@register_decomposition([aten.erfc])`；
4. 第一次 default 激活遗留的 `aten.erfc.default` 仍在 Inductor decomposition 表中，PyTorch
   拒绝重复 entry，报 `duplicate registrations for aten.erfc.default`。

`restore_inductor_baseline()` 当前只恢复 lowering/scheduler 全局属性，不恢复 decomposition
表。`aten.erfc` 虽出现在 `lowering_fallback_list.py` 中，但异常发生在 fallback 注册和图生成
之前，所以不能归类为 fallback 黑名单或算子未入图。

## P-014 修复提案

修改前回滚边界：

- `torch_npu/_inductor/decomposition.py` 源码与安装态逐字节相同，SHA256
  `3fd37092abdbf006253dc8124e26aa66fd990a2bf20c3209a0efab61d7d9af98`；
- 该文件修改前没有 tracked diff；
- 当前 wheel 回滚 artifact 仍由 P-013 检查点管理。

拟只在 `_register_triton_decompositions()` 的既有 cleanup 集合中加入 `aten.erfc`，让每次
default 激活先删除旧 erfc overload entry，再注册当前 default decomposition。不能简单给整个
函数加 `@run_once`：experimental 会覆盖 gelu/dropout 等 decomposition，default 回切仍需要
重新应用自己的表状态。不得修改 Triton kernel、lowering、fallback 列表或 backend 选择优先级。

验证顺序：

1. 静态/纯 Python 重入测试：连续两次 default registrar 不再报重复 entry；
2. source/installed 目标隔离用例，要求 default wrapper marker 与 experimental marker 各自正确；
3. installed 原 13 项必须 13/13；
4. 增加 experimental→default→experimental 往返哨兵，避免只修单向切换；
5. 重建并以 `--no-deps --force-reinstall` 安装 wheel 后重复 2–4；
6. launcher shim 结果只用于产品 Python 修复归因，正式无 shim launcher 缺口继续开放。

在 P-014 关闭前不进入 T-056 pass inventory 的动态性能阶段，避免把 backend 隔离缺陷污染
pass-on/pass-off 对照。

## P-014 源码实施进展

已按提案只在 `DECOMPOSITION_OVERLOAD_OP` 增加一项 `aten.erfc`；修改后
`decomposition.py` SHA256 为
`bd6ac8acea23b8347ff01fdf9095e1ea592b6508fd99a267d4ee028da07ac373`，`py_compile` 和
`git diff --check` 通过。

- 进程内加载当前 source registrar，连续调用两次：`p014_reentry PASS True`；
- 进程内只 overlay 当前 source registrar，预加载 experimental，再执行现有 isolation test：
  `experimental→default→experimental` 1/1 通过，耗时 42.978 秒；
- 证据：`results/t055_triton_experimental_enable_shim_retry3_20260826/p014_source_reentry.log`
  与 `results/t055_p014_source_switch_20260826/test.log`。

当前安装 wheel 未改变，故 installed 13/13 仍待验证。共享源码另有非本轮产生且尚未安装的
`triton_experimental/codegen/triton.py` 修改；直接从当前树构建会把该修改一并带入共享环境，
破坏 P-014 单文件归因。因此在获得隔离构建快照或该共享修改完成前，不构建、不安装 wheel。
T-056 可以继续静态 inventory，但不得启动受该隔离缺陷影响的动态 pass 性能对照。
