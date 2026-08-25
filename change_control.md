# Pass NPU 项目变更控制记录

## 当前冻结状态

- 阶段：用户于 2026-08-21 明确要求继续，暂停已解除；当前恢复 T-014 的
  unaligned/128³ standalone Triton 正确性闸门，尚未进入 profile 或性能验证。
- 单独环境：使用 Conda `Pass`（`/home/z50063656/envs/Pass`）。
- NPU 运行环境：用户已确认稳定；每轮只选择采样前空闲的物理卡，最近 T-011 至
  T-014 使用 NPU 6，不固定占用某张卡，也不终止其他任务。
- 环境稳定性：E-002 冻结已解除；E-003 的 CPU/NPU eager 与 Inductor smoke 全部通过。
- PyTorch/Triton 功能源码修改：仍禁止，除非提案另行批准。torch_npu 仅保留 T-011
  已批准并验证的 `torch_npu/_inductor/lowering.py` tracked diff。
- 当前允许修改：本审计目录的脚本、矩阵、结果和说明文档；任何修改必须先记录在本文档。
- 已完成构建：torch_npu 当前 commit 的 wheel 已从源码构建并以 `--no-deps` 安装；不得让后续 pip 操作替换 PyTorch/Triton 依赖。
- 恢复约束：按 `PAUSED_CHECKPOINT_20260821.md` 的 correctness → profiler → paired
  benchmark 顺序推进；功能源码接入和环境安装仍禁止，恢复记录见 E-030。

PyTorch 工作树保持干净；torch_npu 的唯一 tracked 功能 diff 是 T-011；Triton Ascend
仍只有三处进入本任务前已存在的兼容修改。torch_npu 源码树中的 wheel/codegen 构建生成物
继续原样保留，不执行破坏性清理。

## 修改前置流程

任何代码修改开始前，必须先在本文档新增一条记录并完成审核。每条记录至少包含：

1. 目标 Pass、触发图和业务收益假设。
2. 计划修改的仓库、文件和函数。
3. 当前源码证据，以及问题属于未触发、lowering 失败、fallback、精度还是性能。
4. 候选方案对比：通用 Inductor、NPU lowering、vendor op、CATLASS、AscendC、Triton。
5. 正确性范围：dtype、shape、stride、dynamic shape、forward/backward、空 Tensor 和边界值。
6. 性能方案：paired baseline/candidate、CANN/SoC/版本、warmup、runs、mean/stdev/p50/p99、峰值内存和编译时间。
7. 回退与开关：默认关闭方式、capability gate、失败时的原图 fallback。
8. 审核状态：`draft`、`approved`、`implemented`、`verified` 或 `rejected`。

## 当前环境建立记录

| 项目 | 计划值 | 状态 |
|---|---|---|
| 环境路径/名称 | `/home/z50063656/envs/benchmark-py311` (`benchmark-py311`)，由 `/home/z50063656/Benchmark/env.sh` 激活 | verified |
| Python | 3.11.15 | verified |
| PyTorch commit/wheel | `release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`；editable Python source 为 `/home/z50063656/Benchmark/pytorch-upstream`，native 库/headers来自环境 site-packages | verified |
| torch_npu commit/wheel | `master@83cc452480c3546fd5cccf853bfe3a360ce9dbfc`；T-023 修复版 `dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，SHA256 `d0ee10794f8cb63d528c86f27294a2a52a4b8b5f484eb6be53323d22b2157718`，`--no-deps` 安装 | verified |
| Triton Ascend commit/wheel | runtime module `3.2.0`；任务 source 参考 `release/3.2.2@8bd9f380d2786002b84b5248f00838c26f900515`；fresh launcher需审计 shim 的合同已单列 | verified-with-launcher-caveat |
| CANN | `/usr/local/Ascend/cann9.0.1/cann-9.0.1`，9.0.1 | verified |
| NPU SoC | 8 x `Ascend910B2`，设备 0-7，Health OK | verified |
| 测试服务器/容器 | 当前机器，未进入额外容器；所有测试从 `/home/z50063656/tmp` 启动 | verified |

## 环境核验记录

### E-001：核验 Benchmark 候选运行环境

- 状态：`verified-candidate`，尚未作为性能基线。
- 启动脚本：`/home/z50063656/Benchmark/env.sh`。
- Python 环境：`benchmark-py311`，路径 `/home/z50063656/envs/benchmark-py311`。
- CANN：`/usr/local/Ascend/cann9.0.1/cann-9.0.1`，版本 9.0.1。
- 约束：仍未创建本任务专属环境；本任务继续保持源码冻结。用户反馈有其他进程正在修改环境，当前动态结果全部降级为诊断记录，不得作为性能基线。torchair 导入因 protobuf 元数据缺失失败。
- 设备检查：沙箱外只读 `npu-smi info` 已成功；`npu-smi` 版本 26.0.rc1，共 8 张 Ascend 910B2，设备 0-7 均为 `Health=OK`，单卡 HBM 65536 MiB。采样时 NPU 0 有两个 Python 进程，其余设备空闲。
- Python 核验：从 `/home/z50063656/tmp` source `Benchmark/env.sh` 后，解释器为 `/home/z50063656/envs/benchmark-py311/bin/python`；PyTorch `2.14.0a0+git8e86e0a`，torch_npu `2.14.0`，`torch.npu.is_available()=True`，设备数为 8，设备 0 名称为 `Ascend910B2`。
- 权限差异：沙箱内出现 netlink `Operation not permitted` 并误报设备数为 0；同一命令在已授权 NPU 只读执行中通过，后续 NPU 测试必须沿用已授权执行方式。
- 依赖核验：未加载 torch_npu 的 clean process 可导入 Triton，运行时 `triton.__version__=3.2.0`；`importlib.metadata` 同时报告 `triton=3.5.0`、`triton-ascend=3.2.1`，说明环境存在包内容与元数据不一致风险。`protobuf` 分发元数据未找到。
- 运行规则：后续测试仍从 `/home/z50063656/tmp` 发起，并通过 `source /home/z50063656/Benchmark/env.sh` 激活环境。

### V-001：Benchmark 环境最小 NPU smoke

- 状态：`partial-blocked`；环境不稳定期间停止后续动态测试。
- 范围：只验证一个小型 eager 张量加法和一个最小 `torch.compile` 入口，不修改源码、不安装依赖、不宣称 pass 性能结论。
- 已完成：source `Benchmark/env.sh` 后，eager `torch.randn + torch.randn` 在 `npu:0`、`float16`、`128x128` 上通过，结果有限，首次样本约 6.78 ms（只用于链路诊断，不是性能基线）。
- 未完成：最小 `torch.compile` 在导入 `torch._dynamo` 时失败，错误为 NumPy ABI 不匹配：`numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject`；尚未进入 Inductor pass 或 NPU backend。
- 记录项：设备、dtype/shape、是否成功、异常文本、编译首轮与稳态耗时；在环境锁定后必须从零重跑，不能复用本次耗时。

### E-002：外部环境变更冻结

- 状态：`released-20260820`。用户已明确确认共享环境稳定，可以恢复本任务的受控构建与动态核验。
- 触发：用户反馈有其他进程正在修改 Benchmark 环境，可能改变 Python 包、源码路径、动态库或 Inductor cache。
- 处理：停止所有 NPU 编译/性能探针；不安装依赖、不切换包、不构建 PyTorch/torch_npu、不清理缓存。恢复条件是用户确认修改完成，并重新采集 E-001 的完整版本与路径快照。

### E-003：Pass 环境 torch_npu wheel 重建与最小栈核验

- 状态：`verified-20260820`；wheel 已构建、安装并通过 CPU/NPU eager 与 Inductor 最小验证。
- 目标环境：Conda `Pass`，Python 3.11.15，路径 `/home/z50063656/envs/Pass`。
- 源码合同：PyTorch `release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`；torch_npu `master@83cc452480c3546fd5cccf853bfe3a360ce9dbfc`；Triton Ascend `release/3.2.2@8bd9f380d2786002b84b5248f00838c26f900515`。
- 构建前证据：PyTorch 从 `Pass/src/pytorch` 解析；原安装 torch_npu 2.14.0 的 `direct_url.json` 指向旧 Benchmark wheel，虽然抽查四个 Python 文件与当前源码哈希相同，但不能证明整个 native wheel 来自当前 commit；构建前 `Pass/src/torch_npu/dist` 不存在。
- 构建范围：已初始化 torch_npu 当前 commit 所需 submodule；为遵守测试/导入必须从 `/home/z50063656/tmp` 发起的项目规则，从该目录调用源码 `setup.py build bdist_wheel --dist-dir <torch_npu>/dist`，等价执行仓库构建入口而不从源码目录启动 Python。当前任务只验证 Inductor，设置 `DISABLE_INSTALL_TORCHAIR=TRUE`、`DISABLE_RPC_FRAMEWORK=TRUE`，不构建无关 torchair/RPC；记录 wheel 文件名、大小和 SHA256；在 `Pass` 环境使用 `python -m pip install --no-deps` 安装。不得让 pip 替换 PyTorch、Triton 或其他依赖。
- Triton 保护：保留 `triton-ascend` 的三处既有兼容修改；源码和 site-packages 当前哈希一致。安装后运行 `zyf-env-2-14/scripts/patch_compat.py` 做幂等校验，不覆盖未知差异。
- 验证顺序：所有 Python smoke 从 `/home/z50063656/tmp` 启动；先版本/导入路径，再 CPU eager/Inductor、NPU eager/Inductor。完整通过前不运行 P0 pass 性能探针。
- 性能边界：本条只做环境功能验证，不采集或发布 pass 性能结论。
- 构建诊断：前两次从 `/home/z50063656/tmp` 调用绝对路径 `setup.py` 均在进入 CMake 前失败，错误为 `ModuleNotFoundError: tools.setup_helpers.version`，未生成 `build/` 或 `dist/`。原因是环境中 `/home/z50063656/Benchmark/pytorch-upstream/tools` 普通包遮蔽了 torch_npu 源码的 namespace `tools`；仅追加 `PYTHONPATH` 不能改变该解析结果。后续使用进程内 namespace shim 将 `tools.__path__` 精确指向当前 torch_npu 的 `tools/`，不修改源码文件。
- 构建诊断续：namespace/torchgen packaged bootstrap 生效后，代码生成成功并进入 CMake；CMake 首次配置因 editable PyTorch 把 Python 包位置报告为源码 `Pass/src/pytorch/torch` 而找不到 `TorchConfig.cmake`。实际 native wheel 的配置位于 `Pass` 环境 `site-packages/torch/share/cmake/Torch`，后续只通过 `Torch_DIR`/`CMAKE_PREFIX_PATH` 指向该标准位置，不修改源码。
- 构建诊断续二：补充 CMake 搜索路径后已完成配置，但 Ninja 仍从 `setup.py:get_pytorch_dir()` 得到源码 Python 包路径，因而把 `libtorch_python.so` 错误定位为 `Pass/src/pytorch/torch/lib/libtorch_python.so`；实际文件位于 `Pass` 环境 `site-packages/torch/lib/`。本轮仅在临时 build bootstrap 中把已加载模块的 `torch.__file__` 作为构建路径提示重定向到 native wheel 包，Python 实现仍来自既有 editable 源码，不向 PyTorch 或 torch_npu 功能源码写入兼容补丁。
- 构建诊断续三：上述修正后 CMake/Ninja 已完成 1224/1225 个构建单元并链接 `libtorch_npu.so`，最后由 setuptools 编译 `torch_npu._C` 时失败。原因是扩展源写成相对路径 `torch_npu/csrc/InitNpuBindings.cpp`，而受项目规则约束构建从 `/home/z50063656/tmp` 启动，setuptools 因此在临时目录下查找该文件。保留全部增量构建产物，在 `/home/z50063656/tmp` 增加一个仅用于相对路径解析的临时 `torch_npu` 软链接后重试；该链接不属于源码树或最终 wheel。
- 构建产物：`dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，约 33 MiB，SHA256 `479a481a36f9edad9754e4b076f591d4abd9bca89d019c308cd14681eb6ba52b`。METADATA 版本为 `2.14.0a0+git83cc452`，wheel 内含 `_C.cpython-311-aarch64-linux-gnu.so`、`libtorch_npu.so` 和 `libop_plugin_atb.so`。
- 安装证据：在 Conda `Pass` 中执行精确 wheel 路径的 `python -m pip install --no-deps --force-reinstall`；卸载旧 `torch_npu 2.14.0` 后安装 `2.14.0a0+git83cc452`。运行时 `torch_npu.__file__` 和 `_C.__file__` 均位于 `Pass` 的 site-packages，`torch_npu.version.git_version` 为完整 commit `83cc452480c3546fd5cccf853bfe3a360ce9dbfc`，`direct_url.json` 指向本条 `Pass/src/torch_npu/dist` wheel。
- Triton 证据：兼容脚本检查源码与 site-packages 的 `backend_register.py`、`utils.py`、`runtime/__init__.py` 六个目标，全部返回 `already patched`，没有新增 Triton 修改。
- 验证证据：受限沙箱内版本和 CPU Inductor 通过、NPU 因设备权限不可见；随后在真实机器权限下从 `/home/z50063656/tmp` 运行固定环境验证，CPU Inductor 与 NPU Inductor 均得到 `tensor([1.3818, 1.3818])`，NPU eager 也通过，最终 `status=pass`。完整验证没有使用 `--allow-no-npu`。
- 构建现场：构建脚本主动删除的三个 ACL submodule 已恢复到父仓固定 commit；保留 `build/` 和 `dist/`。torch_npu 父仓当前仅显示两个 submodule 有构建产物，以及 514 个由 codegen/build 产生的未跟踪文件；没有修改父仓已跟踪功能源码。暂不执行破坏性的清理，避免误删后续增量构建资产。

### E-004：P0 pass 动态 gate 首轮

- 状态：`verified-20260820`；20/20 组合 `compile-correct`，目标触发结论已写入矩阵和首轮报告。
- 环境前置：E-003 已完整通过；启动前 `npu-smi info` 显示 NPU 0-6 无运行进程，NPU 7 有两个外部 Python 进程。本轮固定使用 NPU 0，不在 NPU 7 上采样。
- 首个哨兵：从 `/home/z50063656/tmp` 启动 `run_p0_gate_probe.py`，先运行 `addmm_fusion_positive/default`，开启 debug、关闭 benchmark，仅验证 worker 隔离、编译正确性、observer/counter 和 debug artifact 能否形成证据。
- 扩展条件：哨兵通过且证据字段完整后，再扩展到 10 个正/负 case 和 `default,triton_experimental` 两个 backend；首轮仍不加 `--benchmark`，因此结果只能用于可用性和 pass 触发判断，不能用于性能排名。
- 结果目录：`results/p0_gate_smoke_20260820`；后续全量首轮使用独立目录，避免覆盖哨兵证据。
- 首轮结果：`default,triton_experimental` × 10 个正/负 case 全部正确；default addmm fusion 正例触发、experimental 被 gate；experimental mm_plus_mm 正例触发 `_mm_plus_mm`、default 未触发目标；三个 pad family 在两个 backend 的正例都未观察到 pad/slice，其中 experimental 明确把 `shape_padding` 置 False。详见 `report/p0_gate_first_run_20260820.md`。

### E-005：pad family 强制结构诊断

- 状态：`verified-unsupported-20260820`；强制配置在 pass 执行时生效，但 NPU 被上游 device gate 排除。
- 目的：首轮 default 虽有 `shape_padding=True`，三个非对齐正例仍未触发。使用公开 Inductor config `force_shape_pad=True` 区分“replacement/lowering/codegen 不可用”与“正常启发式未选择”。
- 修改范围：仅扩展审计脚本 `run_p0_gate_probe.py`，新增显式 `--force-shape-pad` 诊断参数，并将该标志传入 fresh worker 的 `torch.compile(..., options=...)`；不修改 PyTorch、torch_npu、Triton 源码，不绕过 experimental backend 的产品 gate。
- 执行范围：仅 `default` backend 的 `pad_mm_positive,pad_bmm_positive,pad_addmm_positive`；开启 debug、关闭 benchmark、固定 NPU 0。强制结果只能用于结构可用性判断，不计为真实性能收益。
- 回退：默认参数行为保持不变；若强制路径失败，保留原始 gate/启发式，不做功能源码修改。
- 执行结果：3/3 case 均 `compile-correct`，但 transformed graph 仍只有原始 mm/bmm/addmm，没有 pad/slice。pattern 事件的 active config snapshot 明确为 `shape_padding=True, force_shape_pad=True`；worker 完成后读取到 False 是 `torch.compile(options=...)` 临时 config context 正常恢复，不能解释为 backend 覆盖。
- 根因证据：PyTorch `torch/_inductor/fx_passes/pad_mm.py:check_device()` 只接受 `(a.is_cuda and b.is_cuda)` 或 `(a.is_xpu and b.is_xpu)`。NPU FakeTensor 两者均为 False，`can_pad()` 在性能启发式和 replacement 之前直接返回 False；因此即使强制也无法触发。experimental 还额外通过 `_disable_pad_mm_pass()` 设置 `shape_padding=False`。
- 结论：当前 PyTorch pad-mm family 对 NPU 为明确不可用，不是本轮 shape 选择不佳。三个矩阵条目升级为 `unsupported`；在功能源码提案获批前保持原图 fallback，不做临时 monkeypatch。

### E-006：P0 单 pass 性能 A/B

- 状态：`verified-first-shape-20260820`；两个 pass 均完成 3 轮 × 100 次同 backend A/B，并表现稳定正收益。
- 目标：在同一 backend 隔离评估两个已确认可触发的 pass：`default/addmm_fusion_positive` 与 `triton_experimental/mm_plus_mm_positive`。
- 测试侧修改：仅扩展 `run_p0_gate_probe.py`，新增 `--disable-target-pass`。worker 在 fresh process 内、backend 完成加载后，把当前 family 的已注册 pattern `extra_check` 临时置 False，作为无目标 pass baseline；进程退出即恢复，不写 PyTorch/torch_npu/Triton 源码。
- A/B 合同：同 backend、同 dtype/shape、同一空闲 NPU；current 与 disabled 各自 fresh process；关闭 Inductor cache；warmup 10、runs 100；记录 compiled mean/stdev/p50/p99、首次编译、峰值内存、目标图和 extern call 数。
- 边界：不同进程的单轮结果先标记 diagnostic；若差异接近噪声或 p99 波动明显，需要交错重复至少 3 轮后才能写最终 performance verdict。
- 回退：默认不传该参数，现有探针行为不变；该开关禁止用于宣称产品 gate 已解除。
- 设备变更：采样前复查发现 NPU 0 被外部 `pta_helper.py` 和另一个 Python 任务占用，NPU 1 也被同一外部任务占用；未终止外部进程。性能 A/B 改用当时无运行进程的 NPU 2，并在采样结束后再次检查占用。
- 设备复核：采样结束后 `npu-smi info` 显示物理 NPU 2 无运行进程；采样期间没有观察到外部进程进入该卡。
- `addmm/default`：current 为单个 `aten.addmm`，disabled 为 `mm + add`；三轮 current p50 为 0.219265/0.217130/0.229520 ms，disabled 为 0.284825/0.269925/0.264035 ms。按三轮 p50 中位数计算，0.219265 对 0.269925 ms，延迟下降约 18.8%；p99 中位数 0.237830 对 0.285640 ms，下降约 16.7%。
- `mm_plus_mm/experimental`：current 为单个 `_mm_plus_mm` extern，disabled 为两次 mm 加 add；三轮 current p50 为 0.254355/0.236265/0.238375 ms，disabled 为 0.290240/0.291095/0.295635 ms。p50 中位数 0.238375 对 0.291095 ms，延迟下降约 18.1%；p99 中位数 0.259290 对 0.317660 ms，下降约 18.4%。
- 编译稳定性：addmm disabled 第三轮首次尝试出现一次 `NoTritonConfigsError`，内层为 Triton worker `OSError: could not get source code`；唯一一次受控重试通过。失败 JSON 保留但不计入性能统计，作为工具链稳定性风险。
- 结论边界：两个 pass 在当前 fp16/shape-A 上均为 `beneficial-first-shape`，暂不把整个 pass 的最终 verdict 升级为 `supported-beneficial`；还需补 dtype、shape、layout 和动态 shape 覆盖。

### E-007：P0 覆盖扩展实机哨兵

- 状态：`verified`；执行设备为采样前空闲的物理 NPU 2，未终止 NPU 0/7 上的外部 Python 任务。
- 旧行为：schema 2 的 fp16/shape-A/contiguous/static 回归共 4 个正例，default/experimental 两 backend 全部 `compile-correct`，四条 generated graph 与 E-004 一致。
- addmm 新配置：default、bf16、small、真实 transposed stride、dynamic；首 shape `(32,64)@(64,48)` 和 replay `(40,72)@(72,56)` 均正确，目标图保持符号化 `aten.addmm`。
- mm_plus_mm 新配置：experimental、fp32、unaligned、真实 transposed stride、dynamic；首 shape `(191,255)@(255,319)` 和 replay `(199,263)@(263,327)` 均正确，目标图保持符号化 `_mm_plus_mm` extern。
- 证据：`report/p0_sweep_smoke_20260820.md` 和 `results/p0_sweep_*_20260820/`；本轮未采性能，不升级 final verdict。

### E-008：P0 代表覆盖功能矩阵

- 状态：`verified`；在物理 NPU 2 上按 T-009 的非笛卡尔 cohort 执行 current 模式。
- 规模：addmm/default 8 个配置，mm_plus_mm/experimental 8 个配置，共 16/16 `compile-correct`；8+8 份 output code 全部出现各自目标实现，未出现 graph-break counter。
- 覆盖：fp16/bf16/fp32；small/shape-A/unaligned/large；contiguous/transposed；static/dynamic + 第二组 shape replay。
- 精度：addmm fp16 最大绝对误差 0.0625，fp32 为 `1.1444091796875e-05`，bf16 为 0.5 但逐元素 `rtol=atol=3e-2` 通过；mm_plus_mm 最大绝对误差为 0。bf16 原始差值保留为后续精度边界。
- 证据：`report/p0_sweep_function_matrix_20260820.md` 与 `results/p0_sweep_function_{addmm,mmplus}_*_20260820/`；本轮没有 disabled baseline，不写性能 verdict。

### E-009：P0 扩展性能矩阵

- 状态：`verified`；两个 pass 的代表性能网格和 mm_plus_mm dynamic 高样本复核均完成，物理 NPU 2 在采样前后无运行进程。
- 规模：每个 pass 的 fp16/bf16/fp32 各 current/disabled 三轮，warmup 10、runs 100，共 36/36 `compile-correct`；第 2 轮反转执行顺序。
- p50 中位数：fp16 0.245615 对 0.292720 ms（下降 16.1%）；bf16 0.256425 对 0.295280 ms（下降 13.2%）；fp32 0.250550 对 0.291505 ms（下降 14.0%）。
- p99：三种 dtype 的三轮中位数下降 12.2%-12.5%；fp16 current 第 3 轮单轮 p99 为 0.790580 ms，尾部抖动保留，不删除该轮。
- 编译与内存：current 首次编译约 12.865-13.916 s；disabled 约 19.480-20.474 s。disabled 编译期峰值约 202 MB，包含 Triton 编译/autotune，不能解释为纯 runtime 内存。
- mm_plus_mm dtype：fp16 p50 0.272185 对 0.311030 ms（下降 12.5%）；bf16 0.261255 对 0.312805 ms（下降 16.5%）；fp32 0.279350 对 0.313920 ms（下降 11.0%）。fp16/fp32 p99 改善约 9.9%/9.6%，没有回退但低于 10%。
- addmm shape：small/unaligned/large 的 p50 分别下降 16.1%/16.4%/16.2%；small disabled 首次编译三轮约 35 s，显著高于其他配置，保留为编译成本异常；unaligned disabled 第 1 轮 p99 1.222570 ms，三轮中位数仍改善 10.5%。
- mm_plus_mm shape：small/unaligned/large 的 p50 分别下降 11.0%/13.1%/11.0%；unaligned 的 p99 中位数仅改善 3.5%，无回退但尾延迟收益弱，需保留风险。
- transposed layout：addmm p50/p99 中位数下降 15.4%/18.1%；mm_plus_mm 只下降 6.4%/1.6%。后者目标图与正确性均通过，但低于 10% 性能主门槛，记为该 layout 的 `supported-neutral` 证据。
- dynamic replay：addmm 的 p50/p99 中位数下降 13.1%/16.5%；mm_plus_mm 的 p50 下降 9.51%、p99 仅 0.42%，mean 中位数因尾部波动回退 2.89%。后者接近 10% 门槛，按 T-009 追加 runs 300 独立复核后再定性。
- dynamic 复核：mm_plus_mm 使用 warmup 20、runs 300 再跑 current/disabled 三轮，p50/p99/mean 中位数分别改善 8.74%/5.80%/10.32%；按 p50 主门槛确认为 dynamic 配置的 `supported-neutral` 证据。
- 网格结论：addmm 8/8 配置 p50 超过 10%；mm_plus_mm 6/8 超过 10%，transposed/dynamic 为 neutral，0 个 p50 回退。功能 16/16、主性能矩阵 96/96、复核 6/6 均正确。
- 设备闭环：全部采样结束后再次检查，物理 NPU 2 无运行进程；外部任务未被终止或混入。
- 证据：`report/p0_sweep_performance_20260820.md`，以及 `results/p0_sweep_perf_{addmm,mmplus}_{dtype,shape,layout}_20260820/`、`results/p0_sweep_perf_{addmm,mmplus}_dynamic_20260821/` 和 `results/p0_sweep_perf_mmplus_dynamic_retest300_20260821/`。

### E-010：P0 语义覆盖哨兵与矩阵

- 状态：`verified-with-blocker`；6 个 inference 语义 case、3 个 forward/backward case 和 generated code 已完成，发现一个 torch_npu reduction 接口阻断。
- 探针兼容性：可选 case 从 10 个扩展为 18 个，但无参数默认集合仍为原 10 个；三个 backward case 明确拒绝 dynamic 和 `--benchmark`。`py_compile`、`--list`、`--help`、默认数量与非法参数检查通过，脚本无超过 120 列。
- 设备与环境：物理 NPU 6 在哨兵前无运行进程。首次普通沙箱运行因 Ascend HAL 不可见记为 `environment-blocked` 并保留；受控设备访问重跑后进入真实 NPU 编译，不能把首次阻断记为 pass 失败。
- addmm full bias：default、fp16、small、`(M,N)` bias 为 `compile-correct`，最大绝对误差 0，pattern counter 为 1；generated code 为单个 `torch.ops.aten.addmm.default`，符合正例预期。
- mm_plus_mm different K：experimental、fp16、small 为 `compile-correct`，最大绝对误差 0，post-grad 图出现 `mm_plus_mm` marker；但 generated code 是两个 `extern_kernels.mm` 加一个 Triton pointwise add，不是 `_mm_plus_mm` extern。
- 不同 K 根因：`post_grad.py:is_valid_mm_plus_mm()` 只分别校验两条 matmul 的内部 K，并允许两条 K 不同；`kernel/mm_plus_mm.py:tuned_mm_plus_mm()` 又要求两对输入 size 分别完全相同，否则主动调用两个 mm 和 add lowering。当前证据应记为安全 fallback/融合实现未承接，不得误报为 pattern 未触发。
- addmm vector-bias backward：AOTAutograd 已生成 forward 和 backward debug 图，但 backward 在 bias 梯度的 `aten.sum.dim_IntList` lowering 阶段报 `make_reduction() got an unexpected keyword argument 'strict_sum'`，未进入梯度比较。这不是 addmm 数值失败；PyTorch 2.14 的 `lowering.make_reduction()` 新增 `strict_sum`，而 torch_npu 当前覆盖函数仍只有两个位置参数，并全局替换上游函数，属于明确的 torch_npu/PyTorch lowering 接口不一致。
- inference 矩阵：addmm full/row bias 正例均生成单个 `aten.addmm`，mixed dtype 负例保留 mm + add；mm_plus_mm 不同 K 出现 marker 后安全 unfuse，M/N broadcast 两个负例均无 marker。6/6 正确。
- backward 隔离：mm_plus_mm same-K 输出与 4 个输入梯度最大误差均为 0，forward 生成 `_mm_plus_mm`；addmm full-bias 输出与 3 个输入梯度最大误差均为 0，forward 生成 `aten.addmm`。这证明 pass 本身可进入 AOTAutograd，vector-bias 失败由通用 reduction 阻断。
- 设备闭环：开始前物理 NPU 6 无运行进程；结束复查时出现非本轮 worker 的 PID 3579493。未终止该进程，也未在出现后继续执行 NPU 测试；本轮不采性能，不使用绝对耗时下性能结论。
- 证据：`report/p0_semantic_matrix_20260821.md`、`results/p0_semantic_{addmm,mmplus}_matrix_20260821/` 和三个 backward 结果目录；受限环境记录位于 `results/p0_semantic_smoke_addmm_20260821/`。

### E-011：语义矩阵结束后的源码工作树复核

- 状态：`recorded`；本轮只读复核，不清理、不覆盖其他流程产物。
- PyTorch：`git status --short` 为空，仍是干净工作树。
- torch_npu：commit 仍为 `83cc452480c3546fd5cccf853bfe3a360ce9dbfc`；当前有 2 个 dirty submodule worktree 标记和 514 个未跟踪文件，主体是 `torch_npu/csrc/aten/` 下的代码生成 C++/header 及构建依赖。代表文件 mtime 为 2026-08-20 18:02-18:03，早于本轮语义测试；`torch_npu/_inductor/lowering.py` 无 tracked diff，mtime 为 2026-08-17。本任务没有运行 torch_npu build，也没有编辑这些文件。
- Triton Ascend：仍只有已登记的 `backend_register.py`、`runtime/__init__.py`、`utils.py` 三个 tracked 修改，没有纳入本任务。
- 处理原则：上述 torch_npu 生成物很可能来自此前源码 wheel 构建，属于用户/其他流程状态；不得擅自删除。后续若实施 P-004，应新建受控开发分支或明确构建清理策略，不能把现有生成物混入功能 diff。

### E-012：T-011 首次 wheel 构建环境阻断

- 状态：`diagnosed`；首次 `ci/build.sh --python=3.11` 在编译前退出，旧 wheel 未覆盖，SHA256 仍为 `479a481a36f9edad9754e4b076f591d4abd9bca89d019c308cd14681eb6ba52b`。
- 表象：`setup.py` 报 `ModuleNotFoundError: tools.setup_helpers.version`，但该文件实际存在于当前 torch_npu 源码。
- 根因：Pass 环境的 `_editable_skbc_torch.pth` 额外把旧 `/home/z50063656/Benchmark/pytorch-upstream` 加入 `sys.path`；其中带 `__init__.py` 的 `tools` 包遮蔽了 torch_npu 本地 namespace `tools`。此外，从 torch_npu 源码根运行 setup 时必须设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，否则 PyTorch 自动加载到缺少本地 `_C` 的源码 `torch_npu` 包。
- 受控修正：不编辑 `.pth`、Benchmark 或 torch_npu build helper。只在构建进程内移除这一条旧 Benchmark sys.path，并设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；预检查确认 `tools.setup_helpers.version` 解析到 Pass torch_npu source，`torch` 仍解析到 Pass PyTorch source `2.14.0a0+git8e86e0a`。

### E-013：T-011 第二次 wheel 构建 torchgen packaged 阻断

- 状态：`diagnosed-and-restored`；第二次构建进入 torch_npu codegen 后，因 `Pass/src/pytorch/torchgen/packaged/ATen/native/tags.yaml` 不存在而在 C++ 编译前退出。旧 wheel SHA 未变。
- 根因：Pass 的 PyTorch 是 editable 安装；torchgen Python 代码和 `__file__` 来自 `Pass/src/pytorch/torchgen`，但 `torchgen/packaged` 资源只存在于同一 Conda 环境的 site-packages。torch_npu `get_torchgen_dir()` 按 `torchgen.__file__` 查找 YAML，无法看到 editable 安装拆分出去的 packaged resource。
- 构建副作用：`generate_code.sh` 在复制 ACL headers 后会删除 `third_party/acl_src`。失败后 `ge`、`graph-autofusion`、`runtime` 三个原本干净的子模块一度显示 deleted；已按构建前记录的精确 commit 用 `git submodule update --init --checkout` 恢复。Tensorpipe/DVM 的既有 dirty 状态未触碰。
- 受控修正：仅在构建期间创建 `/home/z50063656/Pass/src/pytorch/torchgen/packaged` 临时 symlink，目标为 Pass 环境 `/home/z50063656/envs/Pass/lib/python3.11/site-packages/torchgen/packaged`；构建后无论成功或失败都核对并移除该 symlink，再恢复三个 ACL 子模块。不得复制、编辑或伪造 YAML。

### E-014：T-011 第三次 wheel 构建 Torch CMake package 阻断

- 状态：`diagnosed-and-restored`；第三次构建完成 torch_npu codegen 和 Python package staging，在 CMake configure 阶段因找不到 `TorchConfig.cmake` 退出。临时 torchgen symlink 已移除，三个 ACL 子模块已恢复；除目标 lowering.py 外没有新增 tracked diff，旧 wheel SHA 未变。
- 根因：editable PyTorch 的 `torch.utils.cmake_prefix_path` 指向 `/home/z50063656/Pass/src/pytorch/torch/share/cmake`，该目录不存在；实际匹配的 `TorchConfig.cmake` 位于 Pass 环境 `/home/z50063656/envs/Pass/lib/python3.11/site-packages/torch/share/cmake/Torch/TorchConfig.cmake`。
- 受控修正：下一次构建仅为该进程设置 `CMAKE_PREFIX_PATH=/home/z50063656/envs/Pass/lib/python3.11/site-packages/torch/share/cmake`，继续使用 E-012/E-013 的 import 与临时 resource 修正。不编辑 CMakeLists、PyTorch editable metadata 或环境持久配置。

### E-015：T-011 第四次 wheel 构建 PyTorch 原生库路径阻断

- 状态：`diagnosed-and-restored`；第四次构建完成 codegen 与 CMake configure，进入 Ninja 后因 `/home/z50063656/Pass/src/pytorch/torch/lib/libtorch_python.so` 不存在而退出。构建创建的临时 torchgen symlink 已移除，三个 ACL 子模块已恢复；PyTorch 源码树干净，torch_npu 当前唯一可见的 tracked diff 是目标 `lowering.py`。
- 根因：`setup.py:get_pytorch_dir()` 直接从 editable `torch.__file__` 取目录，并以显式 `-DPYTORCH_INSTALL_DIR=...` 传给 CMake。当前 Python 源码来自 `/home/z50063656/Pass/src/pytorch/torch`，但匹配该构建的 headers、CMake package 和 `libtorch_python.so` 位于 `/home/z50063656/envs/Pass/lib/python3.11/site-packages/torch`；仅设置 `CMAKE_PREFIX_PATH` 不能覆盖显式参数。
- 构建产物：一次复核误按 `linux_x86_64` 文件名检查，曾误以为 `dist` 原 wheel 被清理；精确枚举确认实际文件名为 `torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，mtime/size/SHA256 均与任务内备份完全一致。两者 SHA256 都是 `479a481a36f9edad9754e4b076f591d4abd9bca89d019c308cd14681eb6ba52b`，回滚能力未丢失。
- 受控修正：下一次构建在一次性 Python wrapper 中预先导入当前 `torch`，只把该进程内的 `torch.__file__` 指向同一 Pass 环境 site-packages 的安装视图，使 `get_pytorch_dir()` 为 CMake 选择匹配的 headers/native libs；子进程 codegen 仍使用 Pass editable Python 源码和 E-013 的临时 packaged resource。不得修改 `setup.py`、CMakeLists、`.pth` 或持久环境。

### E-016：T-011 第五次 wheel 构建 TorchAir 子项目阻断

- 状态：`diagnosed-and-restored`；主 torch_npu CMake/Ninja 已使用正确的 PyTorch install view，并完成约 1287/1332 个目标；失败发生在可选 TorchAir 的独立 `configure`/CMake 子构建。临时 torchgen symlink 与 ACL 子模块已由 wrapper 自动清理，旧 wheel 仍与备份同 SHA。
- 表象：TorchAir 子进程再次从 editable `torch.__file__` 写入自己的 `tools/TORCH_INSTALLED_PATH`，因而寻找 `/home/z50063656/Pass/src/pytorch/torch/lib/libtorch.so`，并缺少 `ATen/ATen.h`。
- 交付一致性：原 wheel 内容清单只有 `torch_npu/dynamo/{__init__.py,_deterministic_guard.py,trace_rule.py}`，不含任何 `torch_npu/dynamo/torchair` 文件；因此原 wheel 本来就是未内置 TorchAir 的构建。下一次设置 `DISABLE_INSTALL_TORCHAIR=TRUE` 不是删减既有交付，而是恢复原 wheel 的组件边界。
- 受控修正：只在 T-011 构建 wrapper 进程内设置 `DISABLE_INSTALL_TORCHAIR=TRUE`，复用已完成的 Ninja 对象继续链接/打包；不修改 TorchAir 源码或其持久 `tools/TORCH_INSTALLED_PATH`。若 wheel 内容边界或 runtime 导入与原包不一致，则不得安装。

### E-017：T-011 第六次 wheel 构建 setuptools 相对源码路径阻断

- 状态：`diagnosed-and-restored`；禁用原 wheel 本就不含的 TorchAir 后，主 Ninja 目标完成到 1239/1240，`libtorch_npu.so`、`libop_plugin_atb.so` 等已链接；随后 setuptools `build_ext` 从 `/home/z50063656/tmp` 查找相对路径 `torch_npu/csrc/InitNpuBindings.cpp` 失败。自动清理再次完成，旧 wheel 未覆盖。
- 原因：全局运行规则要求不得在 torch_npu 源码树 cwd 下 import torch，因此 wrapper 必须从 `/home/z50063656/tmp` 发起；但 `setup.py` 的 `CppExtension` source 使用相对路径，默认假设 cwd 是源码根。主 CMake 使用绝对 `BASE_DIR`，所以此前不受影响，直到最后 Python extension 才暴露。
- 受控修正：wrapper 在 `/home/z50063656/tmp` 下创建一次性 `TemporaryDirectory`，其中只建立 `torch_npu -> <source>/torch_npu` symlink，导入 torch 后进入该临时目录执行 setup，使相对 extension source 可解析；dist/build 输出仍使用既有绝对或可回指路径。finally 删除工作区、torchgen symlink，并恢复 ACL 子模块。不得把 cwd 改回 torch_npu 源码根。

### E-018：T-011 第七次 wheel 安装前内容闸门

- 状态：`build-succeeded-preinstall-rejected`；第七次构建成功生成新 wheel，SHA256 为 `9584d4ecd841d6ad7f5afbd535f2df46f1892620147bef6f048cece52b6ad586`，包内 `lowering.py` 已含目标 `strict_sum` 修复。但安装前文件清单比原 wheel 多 7 条，因此尚未安装。
- 差异：新增 `torch_npu/lib/libtensorpipe.so` 1 条，以及误收进 wheel 的 `torch_npu.egg-info/*` 6 条；新旧包均不含 TorchAir。原 wheel 只带 Tensorpipe RPC headers，不带 `libtensorpipe.so`，说明原构建同时使用 `--disable_rpc` 边界。egg-info 是本轮 setup 在 `build/packages` staging 根生成后被自定义 build_py 整树打包的构建元数据，不应作为运行包内容。
- 受控修正：下一轮 wrapper 同时设置 `DISABLE_RPC_FRAMEWORK=TRUE`，并用 CMake `-U BUILD_TENSORPIPE` 移除旧 cache 开关；精确清理本轮生成的 staging `libtensorpipe.so` 与 `build/packages/torch_npu.egg-info`。wrapper 为 `egg_info` 注入位于一次性 `/tmp` workspace 的 `egg_base`，让 wheel 仍可生成标准 `.dist-info`，但不把 legacy egg-info 目录混入包。清理目标已按路径、mtime 和 archive 差异确认，均为本轮构建产物，不涉及 E-011 保护的源码生成文件。
- 验收：重建后 archive 路径集合必须与原 wheel 一致，TorchAir/Tensorpipe 组件边界一致，包内 lowering 为目标版本，源码 tracked diff 仍只有 `torch_npu/_inductor/lowering.py`；否则继续拒绝安装。

### E-019：T-011 第八次 wheel 构建与安装前验收

- 状态：`verified-install-pending`；使用原 wheel 的 `--disable_torchair --disable_rpc` 边界完成重建。新 wheel 路径仍为 `dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，SHA256 为 `263ffec23ae37c651f3d57199c0cfa8b14f398ea603f8e1434ea14b237792704`。
- 内容验收：新旧 wheel 都是 1317 个 archive entries，排序后的路径集合完全相同；新包不含 TorchAir、`libtensorpipe.so` 或 legacy `torch_npu.egg-info`，平台 tag 仍为 `cp311-cp311-linux_aarch64`。新包只在目标 Python 源码中显示 `strict_sum` 签名和两处 `Reduction.create` 透传。
- 清理验收：临时 torchgen link 和 `/tmp/t011_torch_npu_build_*` 工作区均不存在；PyTorch 工作树干净；torch_npu tracked diff 仍只有 `torch_npu/_inductor/lowering.py`，`git diff --check` 通过。旧 wheel 备份 SHA 保持 `479a481a...a52b`。
- 预安装 runtime：当前已安装包从 `/home/z50063656/envs/Pass/lib/python3.11/site-packages/torch_npu` 加载。普通受限沙箱无法初始化 Ascend HAL，导入 NPU lowering 会在 SOC 探测时报 `aclInit 507008`；这属于已知设备权限边界，安装后动态测试必须在受控 NPU 访问下从 `/home/z50063656/tmp` 运行并开启 `TORCH_COMPILE_DEBUG=1`。

### E-020：T-011 最小 smoke 的图模式 bootstrap 兼容

- 状态：`diagnosed-script-only`；新 wheel 已成功 `--no-deps --force-reinstall`。物理 NPU 6 空闲时首次最小 sum smoke 成功初始化设备，但在进入 reduction lowering 前报 `'function' object has no attribute 'cache_clear'`。
- 根因边界：验证脚本按图模式调试 bootstrap 把 `torch.cuda.get_device_capability` 直接替换为普通 lambda；当前 PyTorch 2.14 的编译/重置链路要求该 capability callable 提供 `cache_clear()`。本次 debug 目录只有 tlparse trace log，没有生成 `output_code.py`，证明失败早于 Inductor codegen，不能归因于 T-011 或 sum 数值。
- 受控修正：把返回值相同的 fake capability 改为 `functools.lru_cache` 包装函数，保留 `(10, 0)` 行为并自然提供 `cache_clear()`；不修改 PyTorch、torch_npu 或 backend 配置。重跑仍使用同一空闲卡和独立 debug/cache 目录，失败则按新产物重新分流。
- 复核：lru-cache capability 重跑仍报同一泛化错误且仍无 `output_code.py`，说明 `transfer_to_npu` 还替换了另一处要求 cache API 的 callable。smoke 先增加完整 traceback 证据再做第三次定位；不得凭同名异常继续猜测或把 bootstrap 失败算成 wheel 回归。

### E-021：T-011 smoke 的 CUDA 迁移层分流

- 状态：`diagnosed-script-only`；带 traceback 的第三次定位确认异常来自 `torch_npu/_inductor/utils.py:patch_is_gpu()` 调用 `get_gpu_type.cache_clear()`。`transfer_to_npu.py` 先把上游由 `functools.cache` 装饰的 `torch._inductor.utils.get_gpu_type` 替换成普通 `_get_npu_type`，导致 cache API 丢失；仍未产生 `output_code.py`。
- 适用性判断：`transfer_to_npu` 的用途是把 CUDA 语义用例迁移到 NPU；T-011 smoke 从创建 tensor 开始就显式使用 `device="npu"`，既不包含 `.cuda()` 也不依赖 CUDA 测试基础设施。现有 P0 探针也走直接 `import torch_npu` 的原生 NPU 路径并已稳定通过。因此该迁移层不属于本 smoke 必需 bootstrap。
- 受控修正：移除 smoke 中 `transfer_to_npu` 导入，保留 `import torch_npu`、`_dynamo.use_jit_script=True`、带 cache API 的 capability stub 和 `torch_npu.testing`。不 monkeypatch `get_gpu_type`，避免用测试补丁掩盖 torch_npu 自身初始化行为；若此后失败，才进入真实 Inductor 产物诊断。

### E-022：T-011 最小 reduction 动态验证

- 状态：`verified`；物理 NPU 6 运行前无进程，使用新 wheel、直接 NPU bootstrap、`TORCH_COMPILE_DEBUG=1` 和独立 cache/debug 目录执行 `sum(dim=0, keepdim=True)`。
- 结果：`compile-correct`，输出 shape `(1, 19)`、dtype fp32，compiled 对 eager 最大绝对误差 `4.76837158203125e-07`，低于 `1e-5` 容差；torch_npu 从 Pass site-packages 加载，版本 `2.14.0a0+git83cc452`。原先的 `strict_sum` 编译接口错误未出现。
- 图模式规则：本次用例通过，因此不展开、不读取成功 run 的 `output_code.py`；失败的前三次 bootstrap 定位均确认没有生成 `output_code.py`。证据位于 `results/t011_strict_sum_smoke_20260821/`，其中保留两次泛化错误、一次 traceback 定位和最终 `direct_npu/result.json`，不删除失败记录。
- 下一闸门：继续在采样前空闲设备上重跑原始 blocker `addmm_fusion_backward`；只有输出与 3 个输入梯度都正确，才把 P-004 从接口 smoke 升级为语义闭环。

### E-023：T-011 原始 blocker 与近邻回归闭环

- 状态：`verified`；新 wheel 已按 `--no-deps --force-reinstall` 安装，SHA256 为 `263ffec23ae37c651f3d57199c0cfa8b14f398ea603f8e1434ea14b237792704`。测试前后物理 NPU 6 均无运行进程，没有终止或混入外部任务。
- 原始 blocker：default/fp16/small vector-bias `addmm_fusion_backward` 为 `compile-correct`；forward 最大绝对误差 `0.03125`，满足 fp16 `rtol=atol=0.01`；A、B、bias 三个输入梯度最大绝对误差全部为 0。AOTAutograd `ok=1/total=1`，原 `strict_sum` TypeError 不再出现。
- addmm 近邻：full-bias backward 为 `compile-correct`，forward 与三个梯度最大误差全部为 0；addmm inference 正例为 `compile-correct`，forward 最大误差 `0.03125`，pattern matcher 计数命中且保留既有 default backend 行为。
- 跨 backend 回归：`triton_experimental` 的 same-K `mm_plus_mm_backward` 为 `compile-correct`，forward 与四个输入梯度最大误差全部为 0；证明 default reduction 修复没有破坏既有实验后端训练路径。
- 图产物规则：以上通过用例都以 `TORCH_COMPILE_DEBUG=1` 预置调试目录；按图模式技能的 PASS 分流不展开成功 `output_code.py`。原失败产物和结果 JSON 均保留，可对照接口错误消失前后的状态。
- 结论：P-004/T-011 验收完成，不回滚 wheel。addmm 已覆盖正负 guard、fp16/bf16/fp32、shape/layout/dynamic、full/row/vector bias、inference/backward 和 8/8 代表配置的 >10% p50 收益，矩阵最终 verdict 可升级为 `supported-beneficial`。证据位于 `results/t011_{strict_sum_smoke,addmm_vector_backward,addmm_neighbor_regression,mmplus_backward_regression}_20260821/`。

### E-024：T-012 mm_plus_mm different-K 成对基线

- 状态：`verified-neutral`；功能哨兵 current/disabled 2/2、性能 worker 12/12 均为 `compile-correct`，最大绝对误差全部为 0。current 每轮 pattern match=1；disabled 每轮 patch 1 个目标 entry且 match=0，控制变量有效。
- 性能：shape-A current/disabled 的 p50 三轮中位数为 `0.321445/0.320490 ms`，current 回退 0.30%；unaligned 为 `0.321665/0.330000 ms`，current 改善 2.53%。两者均低于 10% 门槛。shape-A current 第 2 轮 p50/p99 抖动到 `0.511125/1.682840 ms`，原始值保留并用三轮中位数汇总。
- 编译/内存：shape-A current 首次编译+执行中位数慢 8.05%，unaligned 快 0.39%，没有一致方向；同 shape 下 current/disabled 峰值 allocator 完全相同。
- 设备与图产物：功能哨兵前、性能采样前和性能采样后，物理 NPU 6 均无其他进程。哨兵成功，因此按图模式分流不展开成功 `output_code.py`；不同 K 安全 unfuse 采用既有源码和语义报告证据。
- 结论：现有 different-K pattern 匹配没有形成稳定运行时收益，记为 neutral baseline。下一步必须先做 kernel profile 或不接入源码的候选微原型；本条不能直接证明手写 kernel 值得接入。证据见 `report/t012_mmplus_different_k_baseline_20260821.md`。

### E-025：T-013 disabled profiler backend 激活顺序阻断

- 状态：`diagnosed-script-only`；shape-A/current profiler 哨兵成功，产生 30 条 kernel 记录。首个 shape-A/disabled worker 在 profiler 启动前的 Inductor lowering 失败，错误为 torch_npu `patch_algorithm_selector` 覆盖函数不接受 PyTorch 2.14 新增的 `best_config_future` 参数。
- 图模式分流：失败 traceback 位于 `results/t013_mmplus_different_k_profile_20260821/shape_a_disabled/result.json`；torch compile debug 目录已检查，没有生成 `output_code.py`，证明失败早于 codegen，不能记为 different-K 数值或 profiler 回归。
- 根因：P0 orchestrator 会在 worker 导入 torch 前设置 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`。T-013 初版只在 `torch.compile(options=...)` 指定 backend，导致 `import torch_npu` 先按默认 backend 安装 NPU `AlgorithmSelectorCache.__call__` patch；后续 experimental loader 的 baseline restore 只覆盖 lowering/scheduler，没有恢复 algorithm selector。current fallback 没走到该不兼容调用，disabled 的独立 mm lowering 才暴露它。
- 受控修正：T-013 脚本必须在任何 torch/torch_npu 导入前把 `TORCHINDUCTOR_NPU_BACKEND` 精确设置为 `triton_experimental`，并把该值写入结果。不得 monkeypatch selector 签名或修改 torch_npu 功能源码。失败目录原样保留，重试写入新目录。

### E-026：T-013 different-K kernel profile 闭环

- 状态：`verified-headroom`；修正 backend bootstrap 后的 shape-A/unaligned × current/disabled 四组都为 `profile-complete`、最大绝对误差 0。每组 10 个 active step，精确产生 20 个 `aclnnMm_MatMulCommon_MatMulV2` 和 10 个 `triton_unk_fused_add_0`。
- kernel duration：shape-A current/disabled 的 add 占纯 kernel duration 15.49%/15.75%，unaligned 为 14.14%/13.70%；current/disabled 的名称、顺序、数量和 duration 分布一致。
- timeline：shape-A/current 三 kernel 计算合计 `10.912±0.489 μs`，两个步内 gap 合计 `55.153±3.876 μs`，首 mm 到 add 结束 span `66.075±3.631 μs`；unaligned/current 分别为 `13.724±0.416`、`49.591±6.274`、`63.300±6.154 μs`。
- 可行性预算：结合 T-012 current p50，完全消除步内 gap+add 的端到端理论上限为 shape-A 17.68%、unaligned 16.02%。要达到 10% p50，假设其他开销不变，candidate 单 task 必须分别低于约 33.93/31.13 μs。
- 设备：正式扩展前和全部结束后物理 NPU 6 均无其他进程。失败的初版 disabled 和 backend 修正前 current 哨兵目录保留，但不纳入正式四组。
- 结论：profile 数据足以允许审计目录的 standalone 微原型，但不允许直接改功能源码或解除 size guard。详细报告为 `report/t013_mmplus_different_k_profile_20260821.md`。

### E-027：T-014 初版 split-store 正确性失败

- 状态：`diagnosed-candidate-only`；shape-A fallback baseline 正确，最大/平均绝对误差均为 0，首次编译+执行 `20192.29 ms`。两个 standalone Triton candidate 都完成编译/执行，但没有通过正确性，因此未进入性能采样。
- 128³：20,475/61,440 个元素不满足 fp16 容差，最大绝对误差 83.5；最大相对错误的首个位置为 `(64,0)`，最大绝对错误示例位于 `(70,80)`。错误规模接近完整的第 64–127 行区域，符合 128 行 tile 第二个 64 行 split 未可靠写回的特征。
- 64³：输出包含非有限值；结合 128³ 的行边界，优先判断为 split-store 后半块保留 `torch.empty` 未初始化内容，而不是两个独立 K loop 普遍算错。不得放宽容差或继续做性能。
- 分流：Inductor baseline 已成功，不展开成功 `output_code.py`；失败属于直接 Triton candidate，不是 Inductor compile failure。证据位于 `results/t014_mmplus_different_k_triton_20260821/shape_a_functional/result.json`。
- 下一实验：只保留 128³，暂用一次完整 `(BLOCK_M,BLOCK_N)` fp16 masked store 验证 dot 主体；写入 fresh retry 目录。若完整 store 编译失败或仍出现相同错误，再回到 Triton Ascend store/extension 最小复现，不进入性能。

### E-028：T-014 完整 store 后的 accumulation 语义差异

- 状态：`diagnosed-candidate-numerics`；128³ 完整 masked store 后无非有限值，错误从 20,475 个元素降为 4/61,440，最大绝对误差 `0.02978515625`。这确认初版大面积错误来自 split-store，独立 K1/K2 pointer/mask/loop 主体基本成立。
- 剩余根因假设：当前 candidate 用同一个 fp32 accumulator 连续累加 A@B 与 C@D，最后一次性 cast fp16；fallback/eager 则是两个 mm 各自形成 fp16 输出，再执行 fp16 add。两者数学等价但舍入顺序不同，现有 fp16 `rtol=atol=0.01` 下仍有 4 个元素不通过。
- 处理原则：不放宽容差。candidate 改为两个独立 fp32 accumulator，分别 cast fp16 后相加并写回，以贴近 fallback 的 dtype/舍入语义。先用 128³ 重试；若 UB/编译受限，再用完整 store 的 64³ 对照，仍不得进入性能直到 shape-A 通过既定容差。
- 证据：`results/t014_mmplus_different_k_triton_20260821/shape_a_full_store_retry1/result.json`；fallback baseline 仍最大误差 0。

### E-029：T-014 双累加器 shape-A 正确性闭环与暂停

- 状态：`functional-shape-a-128-verified-paused`；用户要求暂存并暂停后，没有启动任何新实验，只等待已经运行的 128³ 重试结束并记录结果。
- candidate：两个 matmul 分别使用 fp32 accumulator，各自转为 fp16 后再做 fp16 add；shape-A `(M,K1,N,K2)=(192,256,320,128)`、row-major contiguous、static、fp16。
- 结果：fallback baseline 与 128³ candidate 均通过 `rtol=atol=0.01`，两者最大/平均绝对误差都为 0；candidate 状态为 `compile-correct`。baseline 首次编译+执行 `19906.98771 ms`，candidate 首次编译+执行 `5107.895 ms`；这些是单次编译诊断值，不是稳态性能数据，禁止据此宣称加速。
- 解释：独立 accumulator 修复了 E-028 的 4 个舍入不一致元素，证明该 standalone kernel 在一个对齐 shape 上功能可行；它尚未验证 unaligned 尾块、单 task profiler、warmup/runs、p50/p99、编译时间对照和峰值内存，因此不能升级 pass verdict，也不能接入 Inductor。
- 证据：`results/t014_mmplus_different_k_triton_20260821/shape_a_two_acc_retry2/result.json`。该用例成功，按图模式分流不读取成功 baseline 的 `output_code.py`；candidate 是直接 Triton 调用，不是 Inductor 失败分支。
- 设备收尾：worker 已正常返回且执行会话结束。随后 `npu-smi info` 仍在 NPU 6 显示 PID 2314011/约 4038 MiB，但 `/proc/2314011` 已不存在；不推断归属、不终止进程。恢复工作前必须重新检查目标设备并只选择进程表明确为空的物理卡。
- 恢复点：先做同一 128³ candidate 的 unaligned correctness；只有通过后才允许按 T-014 原合同做 profiler 和 paired benchmark。暂停期间不执行这些步骤。

### E-030：T-014 从暂停点恢复与 unaligned 正确性计划

- 状态：`verified`；用户于 2026-08-21 明确要求继续，恢复范围不扩展到源码接入或环境安装。
- 恢复核验：PyTorch/torch_npu/Triton commit 分别仍为 `8e86e0a`/`83cc452`/`8bd9f38`；已安装 torch_npu direct URL 和本地 wheel SHA256 都为 `263ffec23ae37c651f3d57199c0cfa8b14f398ea603f8e1434ea14b237792704`。PyTorch 干净，torch_npu tracked diff 仅 T-011 `lowering.py`，Triton 仍只有三处既有兼容修改。
- 设备选择：恢复前 `npu-smi info` 显示物理 NPU 6 被外部 PID 296394 占用，不终止、不混用；选择进程表为空、Health OK 的物理 NPU 1。运行结束后必须复查。
- 首个实验：从 `/home/z50063656/tmp` 激活 `Pass`，固定 experimental backend、关闭 Inductor cache，设置 `TORCH_COMPILE_DEBUG=1`，只运行 unaligned `(191,255,319,K2=127)`、fp16、contiguous、static、`bm128_bn128_bk128` candidate，写入 fresh 目录 `results/t014_mmplus_different_k_triton_20260821/unaligned_128_resumed1`；不传 benchmark 参数。
- 正确性闸门：fallback 与 candidate 都必须通过 `rtol=atol=0.01`、输出有限，并记录 max/mean absolute error。成功时按图模式规则不读取 baseline 的成功 `output_code.py`；若 Inductor baseline 失败则检查本轮 debug 产物，若仅直接 Triton candidate 失败则按 candidate 分支诊断。
- 后续条件：只有 unaligned 正确性通过才登记并实现/执行 candidate profiler；profile 必须记录 CANN/NPU、warmup/active 次数、每次 kernel duration、mean/stdev/p50/p99 和 task 数。达到单 task/预算闸门后才允许 paired benchmark。

### E-031：T-014 unaligned 128³ 正确性闭环

- 状态：`verified`；物理 NPU 1 在测试前后进程表均为空，NPU 6 的外部任务未被终止或混用。
- 结果：unaligned `(M,K1,N,K2)=(191,255,319,127)` 下，fallback baseline 与 `bm128_bn128_bk128` candidate 都为 `compile-correct`，均通过 fp16 `rtol=atol=0.01`；最大/平均绝对误差全部为 0，输出有限。
- 诊断时延：baseline 首次编译+执行 `19090.60845 ms`，candidate `5238.64126 ms`；仍是单次编译诊断值，不用于性能结论。
- 图模式分流：用例成功，因此没有展开本轮 baseline 的 `output_code.py`；debug 产物保留在独立目录。candidate 为直接 Triton 调用，无 Inductor 失败分支。
- 证据：`results/t014_mmplus_different_k_triton_20260821/unaligned_128_resumed1/result.json`。
- 闸门结论：T-014 已完成 shape-A 与 unaligned 两个 fp16/contiguous/static shape 的 standalone 正确性；允许进入 T-015 candidate-only profiler，但仍不允许端到端 benchmark 或源码接入，直到 task 数与单 task duration 达标。

### E-032：T-015 unaligned candidate profiler 哨兵

- 状态：`verified-budget-pass`；物理 NPU 1 在采样前为空，profile 前后 correctness 最大/平均绝对误差均为 0。
- 采样合同：CANN 9.0.1、Ascend910B2、普通 warmup 10、profile warmup 1、active 10、Level0/AiCoreNone，每 step 同步；原始 profiler 目录和 CSV 已保留。
- task 结果：10 个 active step 精确得到 10 个 `different_k_mm_plus_mm_kernel`，`tasks_per_active_step=1.0`，只有一个 kernel group。
- duration：mean/stdev `13.824±0.541 μs`，p50 `13.81 μs`，p99 `14.68 μs`，范围 `13.12–14.68 μs`；p50 低于 unaligned `31.13 μs` 预算。
- profiler 告警：解析器记录 `Failed to get acl to npu flow events`，但同步解析完成、`kernel_details.csv` 存在且 10 条 duration 完整。由于本任务只使用 kernel 计数/duration，不使用缺失的 ACL→NPU flow 关联；step 间 gap 包含显式同步，禁止解释为纯 launch overhead。
- 证据：`results/t015_mmplus_different_k_candidate_profile_20260821/unaligned_resumed1/result.json` 及同目录 `profiler/`。
- 闸门结论：允许在同一采样合同和物理 NPU 1 上扩展 shape-A；shape-A 未通过前仍不得开始 paired benchmark。

### E-033：T-015 shape-A profiler 与 candidate profile 闭环

- 状态：`verified-budget-pass`；shape-A 在物理 NPU 1 完成与 E-032 相同的采样合同，采样前后设备进程表均为空，profile 前后 correctness 最大/平均绝对误差均为 0。
- shape-A task/duration：10 个 active step 精确得到 10 个同名 `different_k_mm_plus_mm_kernel`，每 step 一个 task；mean/stdev `8.714±0.337 μs`，p50 `8.76 μs`，p99 `9.30 μs`，范围 `8.18–9.30 μs`，低于 `33.93 μs` 预算。
- 两 shape 汇总：unaligned 为 `13.824±0.541 μs`、p50 `13.81 μs`、p99 `14.68 μs`；shape-A 为 `8.714±0.337 μs`、p50 `8.76 μs`、p99 `9.30 μs`。两者均为 10/10 单 task 且低于预算。
- profiler 边界：shape-A 同样出现 ACL→NPU flow 关联告警，但 kernel CSV 导出和 10 条 duration 完整；不使用显式同步形成的 step gap 做 launch 结论。
- 证据：`results/t015_mmplus_different_k_candidate_profile_20260821/{shape_a_resumed1,unaligned_resumed1}/result.json` 及各自 `profiler/` 原始目录。
- 闸门结论：T-015 关闭，允许登记 T-016 端到端 paired benchmark。profiler 只证明 device kernel task 数和 duration 预算，不等价于 host 端 p50 收益。

### E-034：T-016 shape-A paired benchmark

- 状态：`verified-p50-gate-pass`；baseline/candidate 在 benchmark 前后 correctness 最大/平均绝对误差均为 0。物理 NPU 1 在采样前为空，NPU 0/6 外部任务未被终止或混入。
- 合同：CANN 9.0.1、Ascend910B2、fp16/contiguous/static shape-A；fresh process，warmup 10、runs 100、3 轮，执行顺序为 baseline→candidate、candidate→baseline、baseline→candidate。
- p50：baseline 三轮 `0.260615/0.265935/0.269100 ms`，candidate `0.225030/0.223730/0.224440 ms`；三轮中位数 `0.265935/0.224440 ms`，candidate 改善 `15.60%`，超过 10% 主门槛。
- p99/mean：p99 三轮中位数 `0.286370/0.243030 ms`，改善 `15.13%`；mean 三轮中位数 `0.270264/0.225999 ms`。每轮 mean/stdev 原值保存在 JSON，不能只引用汇总值。
- 编译：baseline/candidate 首次编译+执行 `20295.57303/2036.69011 ms`；受各自编译缓存/路径影响，只作诊断，不并入稳态收益。
- 内存：每轮 baseline/candidate 的 additional peak 一致为 `246784/1696768 B`；candidate 绝对多 `1449984 B`（约 1.38 MiB），约为 baseline 的 6.88 倍。绝对量较小但属于明确 trade-off，最终方案必须解释或优化，不能写成“内存无回退”。
- 图模式分流：baseline 成功且正确，因此不读取成功 `output_code.py`；T-012/T-013 已证明同一 current path 为两个 mm + Triton add。debug 目录保留。
- 证据：`results/t016_mmplus_different_k_candidate_benchmark_20260821/shape_a/result.json`。
- 闸门结论：shape-A 达到性能门槛，允许继续 unaligned；只有 unaligned 也超过 10% 且 p99/内存可接受，才关闭 T-016。

### E-035：T-016 unaligned 与两 shape 性能闭环

- 状态：`verified-two-shape-p50-pass-with-memory-tradeoff`；unaligned benchmark 前后 baseline/candidate correctness 最大/平均绝对误差均为 0，物理 NPU 1 在采样前后无进程。
- unaligned p50：baseline 三轮 `0.270735/0.278290/0.282165 ms`，candidate `0.229085/0.230640/0.231065 ms`；三轮中位数 `0.278290/0.230640 ms`，candidate 改善 `17.12%`。
- unaligned p99/mean：p99 中位数 `0.305010/0.257210 ms`，改善 `15.67%`；mean 中位数 `0.279959/0.231958 ms`。首次编译+执行 `20047.08402/2007.1312 ms` 只作诊断。
- unaligned 内存：baseline/candidate additional peak 为 `244736/1695744 B`，candidate 多 `1451008 B`（约 1.38 MiB），与 shape-A 趋势一致。
- 两 shape 结论：shape-A/unaligned p50 分别改善 `15.60%/17.12%`，p99 改善 `15.13%/15.67%`；T-015 又证明每调用一个 task且 kernel p50 为 `8.76/13.81 μs`。限定范围内 candidate 为稳定 beneficial。
- 能力边界：当前只能记为 `standalone-fp16-contiguous-static-beneficial`；bf16/fp32、非连续 layout、dynamic、backward 和内存增长规律未验证，default backend gate 未解除，矩阵 final verdict 保持 `not-run`。
- 图模式分流：两个成功 baseline 均不读取 `output_code.py`；失败历史只涉及直接 Triton candidate，已在 E-027/E-028 分流。
- 报告：`report/t014_t016_mmplus_different_k_candidate_20260821.md`；原始证据位于 T-014/T-015/T-016 结果目录。

### E-036：T-017 功能覆盖设备重新路由

- 状态：`recorded`；T-017 首轮开始前复查发现物理 NPU 1 已被外部 PID 829129 占用，NPU 2–6 也出现外部任务；不终止、不等待、不混用。
- 选择：物理 NPU 7 为当时唯一进程表明确为空且 Health OK 的设备，T-017 功能正确性改用 NPU 7。每个 worker 结束后继续复查；若 NPU 7 被占用则停止，不自动迁移到繁忙卡。
- 结论边界：T-017 不采性能，因此与 T-016 的物理 NPU 1 性能基线不做跨卡数值比较；首次编译时延也不用于任何收益结论。

### E-037：T-017 dtype/layout 功能矩阵闭环

- 状态：`verified-4of4`；物理 NPU 7 在每个 worker 边界均无进程，4 个 fresh-process 配置全部 `functional-complete`。所有成功 baseline 按图模式规则不读取 `output_code.py`。
- fp16 回归：shape-A/contiguous 的 baseline、candidate 与 repeat 最大/平均绝对误差均为 0，证明 output dtype 参数化没有破坏 T-014 至 T-016 的既有路径。
- bf16：shape-A/contiguous 保持 `torch.bfloat16`，在 `rtol=atol=0.03` 下 baseline/candidate/repeat 最大/平均绝对误差均为 0。
- fp32：shape-A/contiguous 保持 `torch.float32`，在 `rtol=atol=1e-4` 下 candidate/repeat 最大绝对误差 `2.86102294921875e-05`、平均 `1.4306393723018118e-06`；baseline 对 eager 为 0。
- 真实转置：fp16/unaligned 的四个输入都为 `is_contiguous=False`，stride 为 `(1,191)`、`(1,255)`、`(1,191)`、`(1,127)`；baseline/candidate/repeat 最大/平均绝对误差均为 0，证明 kernel 实际 stride 寻址可用。
- 证据：`results/t017_mmplus_different_k_coverage_20260821/{fp16_shape_a_contiguous,bf16_shape_a_contiguous,fp32_shape_a_contiguous,fp16_unaligned_transposed}/result.json`。
- 闸门结论：dtype/layout standalone coverage 通过，允许登记 T-018 dynamic replay；本轮无 benchmark，不能外推新 dtype/layout 的性能。

### E-038：T-018 shape-A dynamic 语义通过与 candidate specialization 诊断

- 状态：`semantic-pass-specialization-diagnostic-pending`；首轮 shape-A dynamic 使用同一个 `torch.compile(dynamic=True)` callable 和同一 candidate launcher 完成两组 shape。
- baseline：first/replay 最大/平均绝对误差均为 0；replay 诊断时延 `0.9896 ms`，Dynamo counters 为 `unique_graphs=1`、AOTAutograd `ok=1/total=1`，证明 Inductor baseline 没有为第二组 shape 新建 FX graph。
- candidate：first 最大误差 0；replay `(200,264,328,132)` 最大误差 `0.015625`、平均 `4.7637195166316815e-07`，通过 fp16 容差。replay 单次诊断时延 `3664.59358 ms`，接近首次 Triton 编译量级，提示新 divisibility/shape 触发 kernel specialization 编译。
- 解释边界：语义正确不等于无重编译。当前没有同 shape 的立即 repeat，不能区分一次性 specialization 与持续运行慢；不得据此写 dynamic 性能结论。
- 受控修正：只扩展 `t018_mmplus_different_k_dynamic.py`，每个 variant 的 candidate 首次调用后立即重复同一 inputs，记录 repeat correctness/timing。写入 fresh retry 目录，不覆盖首轮证据；shape-A 诊断闭环后才扩展 unaligned。
- 图模式分流：baseline 用例成功，不读取成功 `output_code.py`；specialization 现象来自直接 Triton candidate，不是 Inductor failure。

### E-039：T-018 repeat 诊断设备重新路由

- 状态：`recorded`；shape-A 首轮后复查发现物理 NPU 7 被外部 PID 1160241 占用，不终止、不混用。此前繁忙的物理 NPU 6 已无运行进程且 Health OK，repeat 诊断起改用 NPU 6。
- 边界：T-018 只做 correctness 和秒级 specialization 诊断，不发布跨卡稳态性能结论；每个 worker 仍需在开始/结束复查目标卡。

### E-040：T-018 两 profile dynamic replay 闭环

- 状态：`verified-semantic-with-specialization-boundary`；shape-A 与 unaligned 都用单个 `torch.compile(dynamic=True)` callable 完成 first/replay，Dynamo counters 均为 `unique_graphs=1`、AOTAutograd `ok=1/total=1`。
- shape-A：baseline first/replay 最大误差均为 0；candidate first/repeat 最大误差 0，replay/repeat 最大误差 `0.015625`、平均 `4.7637195166316815e-07`，通过 fp16 容差。fresh specialization 首轮曾为 `3664.59 ms`；缓存后的 replay 首次/立即 repeat 为 `3.562/0.290 ms`。
- unaligned：baseline first/replay 最大误差 0；candidate first/repeat 最大误差 0，replay/repeat 最大误差 `0.015625`、平均 `2.7012933401238115e-07`。candidate first/repeat 为 `1999.008/0.409 ms`，replay/repeat 为 `0.380/0.278 ms`。
- specialization 结论：same-shape repeat 快，且第二次进程能复用磁盘 cache；但新 shape/divisibility 首次可能触发一个 Triton specialization 编译。当前应写为“dynamic 语义可用、存在一次性 specialization compile 边界”，不能写成“任意新 shape 零编译开销”。
- 性能边界：上述时延是单次诊断值，受磁盘 cache 和跨设备运行影响，不是 warmup/runs 性能结论。成功 baseline 不读取 `output_code.py`。
- 证据：`results/t018_mmplus_different_k_dynamic_20260821/{shape_a,shape_a_repeat_diag1,unaligned_repeat_diag1}/result.json`。
- 闸门结论：dynamic semantic 通过，允许进入 backward 承接分析；正式 integration capability gate 需显式接受或限制 Triton specialization 行为。

### E-041：T-019 两 shape backward 语义闭环

- 状态：`verified-two-shape-training-semantics`；shape-A/unaligned 两个 fresh worker 都为 `backward-complete`，物理 NPU 6 在阶段边界无运行进程。
- compiled baseline：两个 shape 的 output 和 A/B/C/D 四个输入梯度最大/平均绝对误差全部为 0；AOTAutograd 均为 `ok=1/total=1`、`unique_graphs=1`，分别生成 forward/backward debug trace。
- candidate wrapper：两个 shape 的 standalone Triton forward output 与四个公式梯度最大/平均绝对误差也全部为 0。wrapper 只作 semantic diagnostic，不是正式 integration。
- 源码结论：AOTAutograd 在 `compile_fx.py:_compile_fx_main()` 中先切分 forward/backward，然后 `fw_compiler`/`bw_compiler` 分别进入包含 `post_grad_passes()` 的编译流程。正式 forward fusion 应依赖独立 backward graph，不新增一个把所有梯度融合进 forward kernel 的伪方案。
- 图模式分流：baseline 全部成功，不读取成功 `output_code.py`；debug forward/backward 路径保留供正式 integration 回归对照。
- 证据：`results/t019_mmplus_different_k_backward_20260821/{shape_a,unaligned}/result.json`；覆盖报告为 `report/t017_t019_mmplus_different_k_coverage_20260821.md`。
- 闸门结论：功能覆盖已包含 dtype/layout/dynamic/backward；下一阶段优先做 bf16/fp32/transposed/large 的 paired 性能与内存，形成 capability gate 后才允许正式源码提案。

### S-001：Pass 项目源码基线发现

- 状态：`recorded`，等待动态环境稳定性确认。
- PyTorch：`/home/z50063656/Pass/src/pytorch`，`release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`，检查时工作树干净。
- torch_npu：`/home/z50063656/Pass/src/torch_npu`，`master@83cc452480c3546fd5cccf853bfe3a360ce9dbfc`，检查时工作树干净。
- Triton Ascend：`/home/z50063656/Pass/src/triton-ascend`，`release/3.2.2@8bd9f380d2786002b84b5248f00838c26f900515`。
- 已有修改保护：Triton Ascend 中已有 `third_party/ascend/backend/backend_register.py`、`runtime/__init__.py`、`utils.py` 三处未提交修改，属于环境/构建适配；本任务不得覆盖、回退或混入这些修改。
- 激活脚本：`/home/z50063656/Pass/activate_pass.sh` 激活 conda 环境 `Pass`，并声明上述三个源码路径。
- 决策：后续静态清单以 `Pass/src` 为项目基线；旧 `/home/z50063656/Dynamo` 清单保留为历史快照。运行时测试必须在确认环境稳定后重新采集导入路径，不能仅凭激活脚本推断实际加载来源。

### T-001：审计工具验证与清单迁移

- 状态：`verified`。
- 范围：只允许修改 `/home/z50063656/Pass/inductor_pass_npu_audit/audit_passes.py`、README 和生成报告；禁止修改三个源码仓库。
- 复核纠正：最初因相邻输出拼接误认为 pattern registry 分支重复传入 `path=path`；带行号复核确认源码只有一次传参，不存在该语法错误。本条保留纠正过程，避免将误判带入后续分析。
- 原因：生成报告仍包含“本机无 NPU”和旧 `/Dynamo` 源码路径，需要迁移到新的项目源码基线。
- 计划：先用 `py_compile` 验证脚本；将报告措辞改为“运行状态由动态探针单独记录”；从 `/home/z50063656/tmp` 对 `Pass/src` 执行静态扫描；结果写到 `report/pass_src_20260820/`，不覆盖旧报告。
- 验证：`py_compile`、新清单生成成功、记录数与阶段/机制统计、旧清单差异摘要。该验证不导入 torch，不访问 NPU。

### T-002：逐 Pass 评估矩阵生成

- 状态：`verified`；当前矩阵 251 条。P0 已写入动态证据，其中 3 条 pad pattern 的 verdict 为 `unsupported`；addmm 的 performance status 为 `beneficial-default-representative-grid-8of8`，T-011 关闭 reduction blocker 后最终 verdict 为 `supported-beneficial`；mm_plus_mm 为 `mixed-beneficial-6-neutral-2-experimental-grid`，但 default backend 仍 gate，最终 verdict 保持 `not-run`。矩阵总计 247 条 `not-run`、3 条 `unsupported`、1 条 `supported-beneficial`。
- 范围：新增项目内矩阵生成工具和 CSV/Markdown 报告；不修改 PyTorch、torch_npu、Triton Ascend 源码。
- 目标：把静态清单转换为后续可逐项填写的评估合同，至少包含适用性、触发、正确性、fallback/codegen、编译开销、稳态性能和替代方案状态。
- 判定枚举：`not-run`、`not-applicable`、`environment-blocked`、`unsupported`、`supported-neutral`、`supported-beneficial`、`supported-regression`。
- 原则：CPU/MKLDNN/CUDA 专用 pass 应判为 `not-applicable` 或设备 gate 正常，不能误记为 NPU 不支持；一个源码注册项只有在实际触发且 generated code 可验证时才能判定可用。
- 输出：`report/pass_src_20260820/pass_evaluation_matrix.csv` 与矩阵说明文档；环境稳定前所有动态字段保持 `not-run`。

### T-003：DVM/MLIR 后端图变换纳入清单

- 状态：`verified`；精确新增 8 条。
- 范围：扩展 `audit_passes.py` 和矩阵机制映射，不修改三个源码仓库。
- 原因：FX 目录扫描没有覆盖通过 `post_grad_custom_post_pass` 或 DVM partition 内部执行的后端图变换。
- 精确纳入：`dvm_graph_fusion`；DVM 的 `annotate_mm_transpose_flags`、`decompose_k1_matmul_to_mul`、`insert_sum_fp32_prepost_cast_prims`、`insert_promote_cast_by_pos_prims`、`expand_to_reshape`；MLIR 的 `DvmMlirPostGradPass` 和 `fold_sum_cast_to_dtype`。
- 排除：`DvmGraphFusionPatch`、`DvmMlirFusionPatch`、`patch_dvm_mlir_post_grad_pass` 等安装器/上下文管理器不作为独立 pass，仅在对应 pass 的触发条件中记录。
- 实现方式：按文件和符号精确白名单扫描，机制标记为 `backend-graph-pass` 或 `backend-subpass`，避免名称启发式误报。
- 预期：当前 189 条清单增加 8 条；重新生成矩阵，动态字段仍为 `not-run`。

### T-004：函数式与生成式 pattern 注册补齐

- 状态：`verified`；新增 53 条概念级 pattern 记录。
- 范围：扩展静态扫描器和矩阵机制映射，不修改三个源码仓库。
- 缺口：当前扫描覆盖装饰器形式，但没有完整覆盖 `register_replacement(...)`、`register_graph_pattern(...)(handler)` 和 `gen_register_replacement(...)` 生成式 family。
- 计数原则：以概念 pattern/handler 为评估单元；同一 pattern 的 dtype、batch-size、training/inference 注册变体记录在测试参数中，不重复计为独立 pass。
- 精确生成式 family：`pad_mm.py` 的 `mm_pattern`、`bmm_pattern`、`addmm_pattern`；`fuse_attention.py` 的 `_sfdp_pattern_1` 至 `_sfdp_pattern_30`。
- 函数式注册：直接提取 `register_replacement` 的 search function，以及 `register_*pattern(...)(handler)` 的 handler；装饰器识别扩展到所有 `register_*pattern` 包装器。
- 验证：新增项必须能回指注册调用或 family 生成器；不得把 replacement helper、installer 或 dtype 变体重复计数。

### T-005：NPU pass 控制开关纳入清单

- 状态：`verified`；新增 `_disable_pad_mm_pass`，当前总数 251。
- 范围：将 `torch_npu/_inductor/triton_experimental/overrides.py` 加入静态扫描，不修改源码仓库。
- 原因：`_disable_addmm_fusion_pass` 已作为 experimental FX 控制项出现，但 `_disable_pad_mm_pass` 位于 overrides 文件；若不纳入，无法从矩阵追踪 pad-mm family 在 NPU 被整族关闭的控制点。
- 预期：新增 `_disable_pad_mm_pass` 一条 `npu-triton-experimental` 记录；与三个 pad family 一起进入 P0 gate 批次。

### T-006：P0 NPU gate 独立探针

- 状态：`verified-static`；已新增审计目录内的探针和说明，未运行 NPU、未修改三个源码仓库。
- 目标：把 P0 的 `mm_plus_mm`、`pad_mm`、`pad_bmm`、`pad_addmm`、`addmm` 五个 family 转成可执行的当前行为用例，并关联两个 `_disable_*` 控制点。
- 隔离：主进程不得导入 torch；每个 case/backend 启动独立 worker，worker 工作目录固定为 `/home/z50063656/tmp`，避免 torch_npu 全局 patch 在 backend 间串扰。
- 当前行为边界：探针只观察现有 gate，不绕过 `_disable_addmm_fusion_pass` 或 `_disable_pad_mm_pass`，也不临时改源码；需要绕过 gate 的候选对照必须另立提案并经审核。
- 证据：记录 Python/PyTorch/torch_npu/Triton 导入路径与版本、CANN/SoC、case shape/dtype、编译首轮、eager 对照、observer/pattern match、异常和可选 debug 目录。
- 性能边界：支持采集 warmup/runs/mean/stdev/p50/p99，但此阶段只用于探针自检；在没有同进程配置隔离或等价源码 A/B 前，不把 eager-vs-compiled 或不同 backend 的差值归因于单个 pass。
- P0 特殊配置：`mm_plus_mm` 必须打开 max-autotune 路径；pad family 使用非对齐静态 shape 且声明 shape-padding；addmm 使用可广播 bias。所有 family 都包含一个不应触发的负例。
- 验证：只做 `py_compile`、`--list`/参数解析和静态审查；动态执行等待 E-002 解冻。
- 验证结果：`run_p0_gate_probe.py` 通过 `py_compile`，`--list` 成功列出 5 个 family 的 10 个正/负用例；运行命令从 `/home/z50063656/tmp` 发起，验证过程未导入 torch。

### T-007：P1 批次与现有测试证据梳理

- 状态：`draft`；只新增调研文档，不运行测试、不修改三个源码仓库。
- 范围：P1 共 66 条，拆为 B2 NPU custom 27 条、B3 DVM/MLIR 8 条、B4 attention 31 条。
- 目标：为每个批次确定 family、现有测试能证明什么、仍缺哪类 NPU 端到端证据，以及后续探针的最小执行顺序。
- 证据边界：源码中出现测试函数或 pass 名只能记为“已有候选测试”；没有 observer、前后图、generated code 和 NPU eager 对齐时，不能将其升级为 NPU pass 可用结论。
- 验证：P1 数量与评估矩阵一致；所有提到的测试路径必须存在；attention 30 个生成式 pattern 不按 dtype/batch/training 注册变体重复计数。

### T-008：Inductor Pass NPU 从头学习指南

- 状态：`verified`；用户明确要求生成从头学习的入门指南，文档和 README 入口已生成。
- 范围：新增 `inductor_pass_npu_beginner_guide.md`，并在审计 README 增加入口；不修改 PyTorch、torch_npu 或 Triton Ascend 功能源码。
- 目标：以本任务为主线讲清 `torch.compile` 到 Inductor pass、lowering、scheduler/codegen 和 NPU backend 的真实调用链，解释评估矩阵、P0 证据与 P1 工作方法，使开发者读完后能够阅读源码、设计最小用例并判断修改位置。
- 源码证据：关键结论必须引用当前 `Pass/src` 的文件、函数和行号；源码没有明确证据时必须直接说明，禁止把测试推断写成实现事实。
- 文档边界：区分当前权威文档与旧 Benchmark/`/Dynamo` 历史快照；不把首个 fp16 shape 的性能数据外推到完整 pass 结论。
- 验证：文档共 744 行，包含规范要求的 6 个二级章节和 3 个 Mermaid 图；全部相对链接存在，当前环境 commit、251 条矩阵统计、20/20 P0 功能结果及 18.8%/18.1% 首形状性能数据与现有报告一致。

### T-009：P0 dtype、shape、layout 与 dynamic 覆盖扩展

- 状态：`verified`；探针、参数说明、旧行为回归、实机哨兵、16 个代表配置功能矩阵、96-worker paired benchmark 和 6-worker 高样本复核均已完成。
- 范围：只扩展 `run_p0_gate_probe.py`、P0 用例说明、结果与审计报告；不修改 PyTorch、torch_npu 或 Triton Ascend 功能源码。
- 目标：把 `addmm_fusion` 与 `mm_plus_mm` 从单一 fp16/shape-A 扩展到代表性 dtype、M/N/K、非连续转置输入和 dynamic compile/replay，并在同 backend 下生成 current/disabled fresh-worker 配对结果。
- CLI 设计：增加 singular worker spec 与 orchestrator sweep 参数；默认参数保持现有 10 case × 2 backend 的行为。配对模式支持 current/disabled，重复轮次交错执行顺序，debug 路径必须包含完整配置与轮次，避免覆盖。
- 正确性：按 dtype 选择容差；dynamic 模式除首个输入外必须用第二组合法 shape replay，不能只设置 `dynamic=True` 就宣称动态可用。
- 性能边界：先做一轮功能/触发哨兵；只有目标图和正确性稳定后才执行 warmup 10、runs 100、3 轮 paired A/B。不同 dtype/shape 的绝对耗时禁止互相归因，结论只比较同配置 current 与 disabled。
- 计划覆盖：dtype sweep（fp16/bf16/fp32）、shape sweep（small/shape-A/unaligned/large）、layout sweep（contiguous/transposed）和 dynamic replay；避免直接做笛卡尔积造成无效长跑。
- 验证：从 `/home/z50063656/tmp` 执行 `py_compile`、`--list`/参数校验、旧默认行为哨兵和空闲 NPU 上的代表矩阵；回填环境、触发图、正确性、p50/p99、编译首轮、峰值内存和异常。
- 静态验证：`py_compile`、`--list`、`--help` 通过；新增 sweep 参数全部出现在帮助中；脚本无超过 120 列；pad family 的非法 shape sweep 在启动 worker 前被明确拒绝。

### T-010：P0 语义覆盖与 forward/backward 闭环

- 状态：`verified-with-blocker`；探针扩展、静态验证、6 个 inference case、3 个 backward case、源码根因和报告均完成；未修改 PyTorch、torch_npu 或 Triton Ascend 功能源码。
- 源码依据：`post_grad.py:is_valid_addmm_fusion()` 要求 bias 可扩展到 `(M,N)` 且三输入 dtype 相同；`is_valid_mm_plus_mm()` 允许两条 matmul 使用不同 K，但要求输出 M/N 相同；`kernel/mm_plus_mm.py:tuned_mm_plus_mm()` 当前又只对两对输入 shape 分别相同的情形保留融合，否则安全退回两个 mm 加 add。语义用例必须同时覆盖 pattern 与 lowering 两层，不能把合法负例或安全 fallback 记为 NPU 编译不支持。
- addmm 用例：增加 `(M,N)` 完整 bias、`(1,N)` broadcast bias、fp32 bias + fp16 matmul 的 dtype 负例，以及 vector-bias forward/backward 用例。若 vector-bias 被已知 reduction lowering 阻断，再增加 full-bias backward 隔离用例，以判断 addmm pass 本身的 forward/backward 是否可用；阻断记录不得删除。
- mm_plus_mm 用例：增加相同 M/N、不同 K 的正例，N 维 broadcast 的合法负例，以及一个 forward/backward 用例；保留已有 M 维 broadcast 负例。
- backward 方法：eager 与 compiled 使用同值独立叶子 tensor，对输出和每个输入梯度分别做 dtype 容差比较；首次编译计时包含 forward 和 backward。训练用例不采性能、不与 inference 延迟混比，暂只做 static shape。
- 目标图：addmm 正例在 default 应出现 `aten.addmm`，dtype 负例应保留 mm + add；mm_plus_mm 不同 K 正例应在 post-grad 图出现 pattern marker，但按当前 lowering 预期安全退回两个 mm 加 add，M/N broadcast 负例连 pattern marker 都不应出现。backward 用例还需检查 AOTAutograd 产生的 forward/backward debug artifacts，不能只看顶层返回正确。
- 验证顺序：先从 `/home/z50063656/tmp` 做 `py_compile`、`--list`、默认 10-case 兼容性与非法训练 benchmark 参数检查；再在采样前空闲的单卡 NPU 上执行当前模式语义矩阵，保存 JSON 和 generated code，最后回填本条、P0 用例设计和独立报告。

### T-011：default backend reduction `strict_sum` 最小兼容修复

- 状态：`verified`；已按 P-004 完成目标 Python 源码修改、静态检查、同组件边界 wheel 重建、`--no-deps` 安装、最小 reduction、原始 vector-bias blocker 和三项近邻回归。功能源码 diff 仅为 `src/torch_npu/torch_npu/_inductor/lowering.py`。
- 精确修改：将 torch_npu `make_reduction()` 签名补齐 keyword-only `strict_sum: bool = False`，并把该值传给两个分支的 `Reduction.create()`。未传参数的既有 NPU reduction 行为保持不变；不通过 monkeypatch、关闭 strict sum 或强制 fallback 绕过。
- 范围边界：本轮先修复已复现的 default backend 全局覆盖路径。`ascend_npu_ir` 内独立 fork 只做静态审查，不在没有对应失败证据时混入同一 patch。
- 构建与回退：记录原 wheel SHA256，并在覆盖 `dist` 同名 wheel 前保存一份任务内备份；优先使用 Conda `Pass` 和项目标准 `ci/build.sh --python=3.11`。若受 E-012 的旧 Benchmark `tools` 遮蔽阻断，则允许通过一次性 Python 启动包装执行相同 `setup.py build bdist_wheel`：只从构建进程的 `sys.path` 移除旧 Benchmark 路径，并设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，不得编辑环境 `.pth` 或源树 build helper。若受 E-013 的 editable torchgen resource 拆分阻断，允许使用已登记的精确临时 symlink，并必须在构建后移除。新 wheel 仍以 `pip install --no-deps --force-reinstall` 安装；失败时可用备份 wheel 同样 `--no-deps` 回退。
- 工作树保护：E-011 的 514 个未跟踪生成文件、2 个 dirty submodule 标记和 Triton 3 个既有修改均不属于本 patch。构建前后分别统计状态；最终源码 diff 只能包含 lowering.py 的目标改动，生成物不加入提交。
- 静态验证：从 `/home/z50063656/tmp` 执行 `py_compile`；核对函数签名、`Reduction.create(strict_sum=...)` 和 tracked diff；检查源文件行宽与基本风格。
- 动态验证：安装新 wheel 后记录版本、导入路径和 wheel SHA；先跑最小 NPU `sum(dim=0, keepdim=True)` compile case，再重跑 addmm vector-bias backward。随后回归 addmm full-bias backward、mm_plus_mm backward 和一个 inference 正例，保存 generated code、输出与梯度证据。
- 验证入口：最小 reduction 使用审计目录 `t011_strict_sum_smoke.py`，按图模式调试约束预置 torch_npu bootstrap，并由运行命令设置 `TORCH_COMPILE_DEBUG=1`；P0 语义回归继续复用 `run_p0_gate_probe.py` 的 fresh-worker 隔离。所有命令从 `/home/z50063656/tmp` 发起。
- 验收：vector-bias backward 必须从 `strict_sum` 编译错误变为输出与 3 个输入梯度正确；无 CPU fallback；既有正例不回退。若出现数值或 backend 回归，立即停止扩展并回退 wheel，不继续做 Triton 或 pad 实验。

### T-012：mm_plus_mm different-K 当前 fallback 成对基线

- 状态：`verified-neutral`；功能哨兵 2/2、性能 worker 12/12 正确，shape-A/unaligned 的 p50 中位数变化分别为 -0.30%/+2.53%，未达到 10% 门槛；只写入测试结果和文档，没有修改 PyTorch、torch_npu 或 Triton Ascend 功能源码。
- 目标：确认 different-K 在 `triton_experimental` 下虽然匹配 `mm_plus_mm` pattern，但当前 lowering 安全 unfuse 后，相对完全禁用目标 pattern 是否具有任何稳态收益；这一步先建立现状基线，不把 pass-on/pass-off 的差异误当成候选专用 kernel 收益。
- 首轮范围：fp16、contiguous、static，覆盖 `shape_a=(192,256,320)` 和 `unaligned=(191,255,319)`；第二条 matmul 的 K 为 K1 的一半。先用 shape-A 单轮 debug 哨兵验证 current/disabled 均正确、current 有 marker、disabled 的目标 pattern 已 patch；哨兵通过后再运行两个 shape、warmup 10、runs 100、3 轮交错顺序的 paired benchmark。
- 隔离与设备：复用 `run_p0_gate_probe.py` 的 fresh worker 和 `--target-pass-modes current,disabled`，所有命令从 `/home/z50063656/tmp` 发起；仅在采样前无其他进程的物理 NPU 6 上运行，并在结束后复查，不终止或混用外部任务。
- 正确性与解释：每轮 compiled 对 eager 必须满足 fp16 `rtol=atol=0.01`；disabled 必须报告目标 entry 已 patch。若 current 与 disabled 的 generated behavior 都是两个 mm 加 add且性能接近，只能证明现有 pattern 对 different-K 没有运行时融合收益；是否值得手写 Triton/AscendC 仍需结合 kernel profile 或候选微原型，不能从该 A/B 单独推出。
- 停止条件：功能、pattern control、设备隔离任一失败即停止性能采样；若 p50 出现大于 10% 的异常差异，先重复并排查 cache/顺序/生成图，不直接升级实现提案。

### T-013：mm_plus_mm different-K fallback kernel profile

- 状态：`verified-headroom`；四组正式 profile 全部正确，证明 current/disabled 都是每迭代两个 aclnnMm + 一个 Triton add，步内 gap+add 的端到端理论上限为 17.68%/16.02%；只新增审计脚本、结果和报告，没有修改三个功能源码仓。
- 目标：把 T-012 的端到端 neutral 结果拆成 NPU kernel 组成和耗时，回答两个 vendor mm、pointwise add 以及相邻 launch/等待中谁占主导；没有 profile 数据前禁止声称手写 kernel 会更快。
- 实现入口：新增 `t013_mmplus_different_k_profile.py`，复用 P0 探针的 case 构造、experimental compile options 和目标 pattern disable 控制；脚本强制从 `/home/z50063656/tmp` 启动，每种 mode 在独立进程运行。
- 首轮范围：fp16、contiguous、static 的 shape-A 与 unaligned，分别 profile current/disabled。每个 worker 先编译并做正确性比较，再 warmup 10；使用 `torch_npu.profiler` 的 NPU activity、Level0/AiCoreNone，profile warmup 1、active 10，并同步每个 step。记录 CANN/NPU/PyTorch/torch_npu 版本、kernel 名称/类型、每次 duration、mean/stdev/p50/p99、总 device duration和相邻 task gap。
- 产物：保留 profiler 原始目录、`kernel_details.csv`、`operator_details.csv` 等导出文件，并额外生成机器可读 JSON 摘要。原始 profiler 文件不做破坏性清理；解析器必须兼容 `Name/Duration(us)` 与 `Op Name/Task Duration(us)` 两种列名。
- 设备与停止条件：仅在物理 NPU 6 采样前后均无其他进程时运行。若 profiler 导出失败、kernel 数不稳定、profile 后正确性失败或 current/disabled kernel 组成不一致，先诊断采样工具，不进入候选微原型。
- 下一闸门：只有 pointwise add + 可消除 launch/gap 在稳态 device 时间中占比足以支撑超过 10% 的理论上限，才登记不接入源码的独立 K1/K2 微原型；否则保持 P-005 不实现并转向 pad/P1。

### T-014：mm_plus_mm different-K standalone Triton 微原型

- 状态：`functional-and-profile-verified-benchmark-approved`；双 fp32 accumulator 的 128³ candidate 在 shape-A 与 unaligned 都达到最大绝对误差 0；T-015 又确认两个 shape 均为每调用一个 NPU task，p50 duration 低于预算。paired performance 尚未执行。本条仍不向 Inductor registry 注册，也不修改三个功能源码仓。
- 目标：用一个 NPU Triton task 计算 `A(M,K1)@B(K1,N) + C(M,K2)@D(K2,N)`，验证独立 K1/K2 loop 是否能在保持正确性的同时低于 T-013 的 shape-A/unaligned `33.93/31.13 μs` 单 task 预算，并在端到端 p50 超过当前 fallback 10%。
- 源码依据：复用 Triton Ascend `third_party/ascend/tutorials/03-matrix-multiplication.py` 的 row-major pointer/mask、fp32 accumulator 和 split-store 写法；修正上游 `torch/_inductor/kernel/mm_plus_mm.py` 只使用 K1 的限制，为 A/B 和 C/D 建立两个独立 loop、尾块 mask 和 stride。
- 首轮能力边界：仅 fp16、2D、row-major contiguous、static shape；shape-A `(192,256,320,K2=128)` 与 unaligned `(191,255,319,K2=127)`。初始 tile 只比较 `128x128x128` 和 `64x64x64`，不启用 autotune，避免把 autotune 选择/缓存混入 kernel 本体判断。
- 验证顺序：先 shape-A 单次编译与 eager 正确性哨兵；通过后运行 unaligned 正确性。两者都通过才进行 current fallback 与各 candidate 的 warmup 10、runs 100、3 轮交错 benchmark，并用 NPU profiler 验证 candidate 每次恰好一个 task。
- 数值：accumulator 为 fp32，输出转回 fp16；compiled/candidate 对 eager 使用 `rtol=atol=0.01`，同时记录 max/mean absolute error。不得通过放宽容差、删除第二个 matmul或在 host 端补 add 获得伪收益。
- 设备与隔离：任何 torch 导入前固定 experimental backend；从 `/home/z50063656/tmp` 启动，只使用采样前后无其他进程的物理 NPU 6。所有 candidate 结果写入 fresh 目录，不覆盖 T-012/T-013。
- 停止条件：任一候选编译失败、精度失败、输出非有限、task 数不为 1，或两个 tile 在任一 shape 的单 task duration 都超过预算，则保留失败并停止源码接入讨论；可以调整 standalone tile，但不得直接改功能源码。
- 接入闸门：只有同一候选在两个 shape 的端到端 p50 都稳定超过 fallback 10%，且 p99、编译时间、峰值内存可接受，才扩展 bf16/fp32、transposed、dynamic 和 backward，并另行登记正式 Inductor template/lowering 方案。

### T-015：mm_plus_mm different-K standalone candidate profiler

- 状态：`verified-budget-pass`；E-032/E-033 的 unaligned/shape-A profiler 均满足一个 task/step 和各自 duration 预算，profile 前后 correctness 保持最大误差 0；不修改 PyTorch、torch_npu 或 Triton Ascend 功能源码。
- 目标：确认 T-014 `bm128_bn128_bk128` candidate 每个 active step 只产生一个 NPU task，并测量 shape-A/unaligned 的单 task duration 是否分别低于 T-013 的 `33.93/31.13 μs` 预算。
- 实现：新增 `t015_mmplus_different_k_candidate_profile.py`，复用 T-014 的输入 shape、candidate launcher 与 correctness，复用 T-013 的 `kernel_details.csv` 解析；在任何 torch 导入前设置 experimental backend。脚本必须从 `/home/z50063656/tmp` 启动并拒绝覆盖已有结果目录。
- 采样合同：每个 shape 先编译/正确性，再普通 warmup 10；NPU profiler 使用 Level0/AiCoreNone、profile warmup 1、active 10，每 step 同步。记录 CANN/NPU/PyTorch/torch_npu/Triton、输入 shape/stride、首次编译、profile 前后正确性、kernel 名称、count、duration mean/stdev/p50/p99 和 `tasks_per_active_step`。
- 执行顺序：先在物理 NPU 1 做 unaligned profiler 哨兵；必须得到 10 个 active kernel、每 step 1 个 task且 correctness 保持通过，才扩展 shape-A。原始 profiler 目录和 CSV 不清理。
- 停止条件：任何非有限/精度失败、task 数不是 1/step、导出/解析失败，或两个 shape 任一单 task duration 超过对应预算，都停止 paired benchmark 并记录 candidate 不达标；不得以 host timing 或单次编译时延代替 profiler。
- 下一闸门：只有两个 shape 均为一个 task/step且 duration 预算通过，才登记 T-016 端到端 current fallback/candidate 三轮 paired benchmark。

### T-016：mm_plus_mm different-K fallback/candidate paired benchmark

- 状态：`verified-two-shape-beneficial-with-memory-tradeoff`；shape-A/unaligned p50 分别改善 15.60%/17.12%，p99 同向改善；candidate additional peak 均比 baseline 多约 1.38 MiB。当前只证明 fp16/contiguous/static forward，功能源码接入仍禁止。
- 目标：在相同 experimental backend、输入和 fresh process 内，比较当前 Inductor different-K 安全 fallback（两个 mm + add）与 T-014 128³ standalone candidate 的端到端稳态延迟、首次编译和峰值 allocated memory。
- 实现：新增 `t016_mmplus_different_k_candidate_benchmark.py`，复用 T-014 shape/candidate/correctness；baseline 使用与 T-012 相同语义的 `torch.compile(fullgraph=True, npu_backend=triton_experimental, max_autotune*)` 当前路径。任何 torch 导入前设置 backend，脚本只允许从 `/home/z50063656/tmp` 启动并拒绝覆盖结果。
- 采样合同：CANN 9.0.1、Ascend910B2、fp16/contiguous/static；shape-A 和 unaligned 各自 fresh process；baseline/candidate 各 warmup 10、runs 100、3 轮，轮次交替执行顺序。每轮记录 mean/stdev/p50/p99、起始 allocated、peak allocated 和 additional peak；记录两者首次编译+执行与测试前后 correctness。
- 图模式约束：运行前设置 `TORCH_COMPILE_DEBUG=1` 和独立 debug 目录。baseline 成功则不读取 `output_code.py`；baseline 编译或数值失败必须先检查本轮生成图。candidate 直接 Triton 失败按 candidate 分支处理。
- 设备：继续使用每轮开始前进程表为空的物理 NPU 1，并在两个 shape 结束后复查；不终止 NPU 0/6 的外部任务。
- 判定：以三轮 p50 的中位数为主，candidate 相对 fallback 两个 shape 都必须改善超过 10%；同时报告三轮 mean/stdev/p99、首次编译和峰值内存。任一 shape ≤10% 或出现 p50 回退，就不进入源码接入；p99、内存或正确性异常也停止。
- 产物：两个 shape 写入 `results/t016_mmplus_different_k_candidate_benchmark_20260821/` 下独立目录；完成后生成独立 Markdown 报告并回填矩阵，不覆盖 T-012/T-015。

### T-017：mm_plus_mm different-K standalone dtype/layout 扩展

- 状态：`verified-4of4`；fp16 回归、bf16、fp32 与真实 transposed stride 全部正确，当前只证明 standalone capability，不接入 Inductor registry/lowering。
- 目标：验证同一独立 K1/K2 kernel 能否保持 fallback 的输出 dtype/舍入语义，并正确处理真实 non-contiguous transpose stride；区分 kernel 能力与 launcher 人为限制。
- 计划修改：仅修改审计目录 `t014_mmplus_different_k_triton.py`，增加 compile-time output dtype，输出 tensor 跟随输入 dtype，按 dtype 使用 fp16/bf16/fp32 容差，并把“必须 contiguous”改为“4 个输入同 dtype、2D、受支持浮点 dtype”；kernel 继续使用已传入的实际 stride。新增 `t017_mmplus_different_k_coverage.py` 作为单配置 fresh-process runner。
- 不变项：两个 matmul 仍各用 fp32 accumulator，再分别 cast 到输出 dtype 后执行该 dtype add；128³ tile、两个独立 K loop、尾块 mask、单 task 结构和 fp16 既有行为不变。不修改 PyTorch、torch_npu、Triton Ascend 功能源码。
- 正确性 cohort：先跑 fp16/shape-A/contiguous 回归；再跑 bf16/shape-A/contiguous、fp32/shape-A/contiguous、fp16/unaligned/transposed。输入通过交换 storage 的最后两维后 transpose 构造，必须记录 stride 并证明 `is_contiguous=False`。
- baseline：每个配置同时运行 eager reference 和当前 experimental Inductor fallback，设置 `TORCH_COMPILE_DEBUG=1`；baseline 成功不读 `output_code.py`，失败则按图模式技能检查本轮产物。candidate 直接 Triton 失败按 candidate 分支诊断。
- 容差：fp16 `rtol=atol=1e-2`、bf16 `3e-2`、fp32 `1e-4`；记录输出 dtype、max/mean absolute error和是否有限，不允许通过放宽容差通过。
- 设备与停止条件：每轮从 `/home/z50063656/tmp` 在开始前进程表为空的设备执行，fresh 输出/debug 目录；E-036 起使用物理 NPU 7。fp16 回归失败立即停止并回滚审计脚本；任一新 dtype/layout 失败则保留 capability gate，不进入 dynamic。
- 下一闸门：4 个配置全部正确后才登记 T-018 dynamic replay；本批次不采性能，不用首次编译时延做结论。

### T-018：mm_plus_mm different-K standalone dynamic replay

- 状态：`verified-semantic-with-specialization-boundary`；两个 profile 的 single compiled baseline 均为一个 FX graph，candidate first/replay/repeat 语义正确；新 shape/divisibility 首次可能触发一次 Triton specialization 编译。不采性能、不接入源码。
- 目标：同一个 `torch.compile(dynamic=True)` baseline 与同一个 standalone candidate launcher 连续处理首 shape 和 M/N/K1/K2 全部变化的第二组 shape，确认尾块、grid 和 runtime K loop 不会错误 specialize。
- 实现：新增 `t018_mmplus_different_k_dynamic.py`，复用 T-014 kernel/correctness；每个 shape profile 在一个 fresh process 内构造 variant 0 和 `M/K/N + 8` 的 variant 1，K2 始终为 K1 的一半。记录两组输入 shape/stride、baseline/candidate correctness、首次/重放时延诊断和 Dynamo counters。
- cohort：fp16/contiguous 的 shape-A `(192,256,320,128) -> (200,264,328,132)` 与 unaligned `(191,255,319,127) -> (199,263,327,131)`，两个 profile 分开运行。
- 图模式约束：设置 `TORCH_COMPILE_DEBUG=1`、独立 debug 目录和 `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`。任一 baseline 失败或数值不一致必须先读本轮 `output_code.py`/变换图；成功则不展开。candidate 直接 Triton 失败单独分流。
- 正确性：两组 baseline/candidate 都必须保持 fp16 输出、有限且通过 `rtol=atol=0.01`；两组 shape 必须不同。不得用两个独立 compiled callable 冒充 replay。
- 设备与停止条件：只在 worker 前后进程表为空的设备上从 `/home/z50063656/tmp` 执行；首轮用 NPU 7，E-039 起改用 NPU 6。任一 profile 失败则保留 static capability gate，不进入 backward。
- 下一闸门：两个 dynamic profile 都通过后，才分析 AOTAutograd/post-grad 阶段下 backward 应由独立梯度图承接还是需要 standalone autograd wrapper，并登记最小训练验证。

### T-019：mm_plus_mm different-K backward 承接验证

- 状态：`verified-two-shape-training-semantics`；compiled fallback 与 standalone wrapper 在 shape-A/unaligned 的 output 和四个输入梯度全部正确，不修改功能源码、不采性能。
- 源码结论：`compile_fx.py:_compile_fx_main()` 先由 AOTAutograd 创建 joint graph并切分 forward/backward，再分别调用 `fw_compiler`/`bw_compiler`；`post_grad_passes()` 明确会在 forward graph 和 backward graph各执行一次。因此正式 different-K forward fusion 在 post-grad 后承接时，不要求直接 Triton kernel 自身具备 eager autograd；梯度应由已切分的 backward graph 独立编译。
- 目标：一方面验证当前 experimental different-K safe fallback 的 compiled forward/backward；另一方面用仅限审计的 `torch.autograd.Function` 包装 standalone forward，并以四个标准 matmul 公式承接 backward，确认 candidate 输出与 A/B/C/D 梯度语义可闭环。
- 实现：新增 `t019_mmplus_different_k_backward.py`，复用 T-014 forward candidate。使用同值独立叶子 tensor和确定性 random `grad_output`，分别运行 eager reference、`torch.compile(fullgraph=True)` baseline和 candidate wrapper；比较 output 与 4 个 input gradients。
- backward 公式：`dA=grad@B.T`、`dB=A.T@grad`、`dC=grad@D.T`、`dD=C.T@grad`。wrapper 仅为 standalone 诊断，不作为正式接入方案；最终 Inductor integration 必须依赖 AOTAutograd 生成的 backward graph并重新做图产物验证。
- cohort：fp16/contiguous/static 的 shape-A 与 unaligned，fresh process；`rtol=atol=0.01`，记录 output/grad 的 dtype、shape、max/mean absolute error。不得只比较 loss 标量。
- 图模式约束：设置 `TORCH_COMPILE_DEBUG=1`。compiled baseline 失败或梯度不一致时必须检查本轮 forward/backward debug 产物；成功时不读取。candidate wrapper 失败按直接 Triton/公式分流。
- 设备与停止条件：只选 worker 前后进程表为空的设备，从 `/home/z50063656/tmp` 启动。任一 shape 的任一梯度失败则保持 training capability 未支持，不进入正式接入。
- 下一闸门：两 shape 训练语义通过后，再决定是先做扩展性能矩阵/内存规模复核，还是登记正式 template/lowering capability gate 提案；两者都必须在功能源码修改前写入本文档。

### T-020：mm_plus_mm different-K 扩展性能与内存矩阵

- 状态：`completed-3-beneficial-1-unstable-hold`；bf16/fp32 shape-A/contiguous 与 fp16/unaligned/transposed 均超过 10% p50 门槛；large 隔离有效重测的 p50 中位数也改善 11.58%，但存在单轮 p50 回退、高方差/长尾和额外峰值内存放大，暂不纳入自动启用范围。T-020 仅完成数据门禁，未修改任何功能源码。
- 目标：判断 T-016 的 fp16/contiguous 两 shape收益能否外推到 bf16、fp32、真实 transposed stride 和更大 shape，并观察 candidate additional peak memory 是常量级 runtime workspace还是随输出规模异常增长。
- 实现：新增 `t020_mmplus_different_k_extended_benchmark.py`，复用 T-014 candidate、T-016 `_sample` 和 T-017 真实转置输入构造。baseline 仍为同配置 experimental Inductor different-K safe fallback；每个配置 fresh process且拒绝覆盖结果。
- cohort：bf16/shape-A/contiguous、fp32/shape-A/contiguous、fp16/unaligned/transposed、fp16/large/contiguous，其中 large 为 `(M,K1,N,K2)=(512,768,640,384)`。避免笛卡尔积，先覆盖最可能改变 kernel 性能或内存的单因素。
- 采样合同：同一物理卡、CANN 9.0.1、Ascend910B2；baseline/candidate 各 warmup 10、runs 100、3 轮交错顺序；记录每轮 mean/stdev/p50/p99、allocated before、peak/additional peak、首次编译和 benchmark 前后 dtype-specific correctness。
- 图模式约束：设置 `TORCH_COMPILE_DEBUG=1`、独立 debug 目录、关闭 Inductor cache。baseline 失败或数值不一致必须检查本轮图产物；成功不读。candidate 失败单独分流。
- 设备：从 `/home/z50063656/tmp` 启动，每个配置前后重新选择进程表为空的同一物理卡；当前首选 NPU 6。若采样期间设备被占用则停止，不能把不同卡的绝对时延拼成同一配置三轮。
- 判定：每个配置独立以三轮 p50 中位数判断。>10% 才进入 candidate capability gate；≤10% 记为 supported-neutral 并保留 fallback；回退则明确排除。p99、mean/stdev 和 additional peak 同时报告，不能用另一个 dtype/shape 的收益替代。
- 停止条件：正确性、设备隔离或三轮完整性失败即停止该配置；一个配置性能不达标不妨碍其他配置完成，但禁止全局打开 candidate。
- 产物：`results/t020_mmplus_different_k_extended_benchmark_20260821/` 和独立 Markdown 报告；完成后更新 P-005 capability 范围，仍不直接修改功能源码。
- E-042（bf16/shape-A/contiguous）：CANN 9.0.1、Ascend910B2、warmup 10、runs 100、3 轮交错采样；baseline/candidate p50 中位数为 `0.271645/0.241225 ms`，改善 `11.20%`，p99 为 `0.299060/0.265340 ms`，改善 `11.28%`，通过 10% 门槛。benchmark 前后两路 max/mean absolute error 均为 0；candidate additional peak 为 `1,696,768 B`，baseline 为 `246,784 B`，多 `1,449,984 B`（约 1.38 MiB）。首次编译时延仅作诊断记录，不用于稳态收益判定；baseline 成功，按图模式规则未读取 `output_code.py`。
- E-043（fp32/shape-A/contiguous）：NPU 7 在 E-042 后出现外部 PID 1494970，未清理；本配置改在空闲 NPU 6 内完成 baseline/candidate 成对采样，不用跨卡绝对时延作 dtype 对比。p50 中位数为 `0.283520/0.240285 ms`，改善 `15.25%`，p99 为 `0.309150/0.260550 ms`，改善 `15.72%`；candidate benchmark 前后 max/mean absolute error 为 `2.8610e-5/1.4306e-6`，通过 `1e-4` 容差。candidate/baseline additional peak 为 `3,392,512/492,544 B`，候选多 `2,899,968 B`（约 2.77 MiB），呈现相对 bf16 近似按 dtype 字节数放大的风险。baseline 第 3 轮 mean/stdev 为 `0.341249±0.368972 ms`，存在单轮长尾；主判据仍是预先登记的三轮 p50 中位数，报告必须保留该稳定性限制。baseline 成功，未读取 `output_code.py`。
- E-044（fp16/unaligned/transposed）：E-043 后 NPU 6 出现外部进程，本配置改在空闲 NPU 3 内成对采样。4 个输入均为真实非连续转置 stride：`(1,191)/(1,255)/(1,191)/(1,127)`。baseline/candidate p50 中位数为 `0.289620/0.252825 ms`，改善 `12.70%`，p99 为 `0.347840/0.277940 ms`，改善 `20.10%`；benchmark 前后 max/mean absolute error 均为 0。candidate/baseline additional peak 为 `1,695,744/244,736 B`，多 `1,451,008 B`（约 1.38 MiB）。baseline 编译期报告 mspti device-time 不可用并回退 event-based autotune，小 NPU kernel 可能选到次优配置；因此该数据代表“当前实际 Inductor 路径”的改善，不声称 baseline 已达硬件理论最优。baseline 成功，未读取 `output_code.py`。
- E-045（fp16/large/contiguous 隔离无效首轮）：采样前 NPU 1 进程表为空，但结束后出现外部 PID 1594529（约 38 GiB），触发本任务预先登记的隔离停止条件。原始结果保留于 `fp16_large_contiguous/result.json`，但不进入 capability verdict：其 p50 表观改善 `20.10%`，p99 却回退 `23.73%`，baseline/candidate 均有多毫秒长尾；candidate max absolute error 为 `0.0625`但通过 fp16 `rtol=atol=0.01`；candidate/baseline additional peak 为 `5,899,264/1,311,744 B`。新鲜目录重测前不对 large 作支持或性能结论。
- E-046（fp16/large/contiguous 隔离有效重测）：在采样前后进程表均为空的 NPU 7 、fresh `fp16_large_contiguous_rerun1` 目录完成。baseline/candidate p50 中位数为 `0.344085/0.304245 ms`，改善 `11.58%`；3 轮 p50 成对为 `0.327660/0.278710`、`0.344085/0.304245`、`0.367740/0.370085 ms`，第 3 轮候选回退约 `0.64%`。p99 中位数为 `1.290280/0.771170 ms`，表观改善 `40.23%`，但 candidate 第 2/3 轮 mean/stdev 为 `0.389957±0.659924`/`0.541238±0.630181 ms`，第 3 轮 p99 `3.611300 ms`，稳定性不足。benchmark 前后 candidate max/mean absolute error 为 `0.0625/8.7023e-7`，通过 fp16 预设容差；candidate/baseline additional peak 为 `5,899,264/1,311,744 B`，候选多 `4,587,520 B`（约 4.38 MiB），证明该开销不是固定常量。结论为 p50 gate 通过但因单轮回退、长尾/方差与内存放大而 hold；需 profiler/更合理 tile 后才能纳入正式 capability gate。baseline 成功，未读取 `output_code.py`。

### T-021：mm_plus_mm different-K 正式接入设计审计

- 状态：`completed-design-t022-prerequisite`；已完成 PyTorch/torch_npu 源码数据流、方案比较、拟修文件、fallback、构建与验证合同；本阶段没有修改功能源码。
- 目标：沿 `post_grad pattern -> tuned_mm_plus_mm -> choice/template -> NPU backend patch/codegen` 找到不需全局解除 size guard 的最小接入面，确定 standalone different-K kernel 应作为通用上游 template 还是 NPU 专用 choice，以及何时必须保留两个 mm + add fallback。
- 必查源码：`torch/_inductor/fx_passes/post_grad.py`、`torch/_inductor/kernel/mm_plus_mm.py`、algorithm selector/template 相关 API；`torch_npu/_inductor/fx_passes/post_grad.py`、experimental backend loader、Triton/CATLASS 模板与 codegen/scheduling 扩展点。必须记录函数和实际数据流，不仅列文件名。
- 设计门禁：原有 same-K 行为不变；different-K candidate 只在 NPU、2D、同 dtype/device、可表示 stride 和已证明 dtype 下成为可选路径；dynamic 新 specialization、large/不稳定 shape、autotune 不可靠、编译或精度不满足时必须回到现有 fallback。不使用仅匹配测试 shape 的硬编码 whitelist 冒充通用能力。
- 内存与性能：设计文档必须解释 standalone additional peak 可能来自哪一层，如何在正式 scheduler/wrapper 中复测；large 在 profiler/tile 闭环前不进入首批 gate。
- 验证设计：需要列出 pattern 触发/不触发、generated graph/code、fallback 注入、三 dtype、contiguous/transposed、aligned/unaligned、dynamic first/replay、AOTAutograd forward/backward、空维/数值压力、paired p50/p99/内存和失败回滚。设计评审完成前不进入 implementation。
- 产物：`report/t021_mmplus_different_k_integration_design_20260821.md`，并回填 P-005 的拟修文件、无需/需要 wheel 重建判断、回滚边界和 T-022 前置条件。
- E-047（设计结论）：当前 pattern 的 `is_valid_mm_plus_mm()` 本身允许 K1!=K2，实际退回发生在 `kernel/mm_plus_mm.py:tuned_mm_plus_mm()` 的完整 size equality guard。上游 Triton template 没有独立 K2 loop，且 template heuristic 没有为 device type `npu` 注册；NPU 当前只获得 device-agnostic `aten_mm_plus_mm` extern choice。选定的首批方案是 torch_npu default backend 中默认关闭开关保护的 NPU-only/static/different-K duplicate pattern，调用新 `NPUTritonTemplate`，并始终保留支持 different K 的 `aten_mm_plus_mm` C++ composite extern 作为第一 fallback。不全局 monkeypatch 上游 handler，不在 lowering 中直调 eager Triton，不修 PyTorch/Triton Ascend/C++。拟修为 5 个 torch_npu Python 文件和新 UT；迭代不需重编 PyTorch，但当前正式环境最终验证必须重建 torch_npu wheel 并 `--no-deps` 安装。根据 T-020 风险，implementation 前先执行 T-022 large profiler/tile/内存分解。

### T-022：mm_plus_mm different-K large profiler、tile 与内存分解

- 状态：`completed-large-supported-neutral-hold`；三个 tile 均正确且为单 task，device p50 以 `128x128x128` 的 `14.41 μs` 最优，但三组 paired p50 改善仅 `6.64%/6.97%/7.55%`，都低于 10% 门槛；large 继续排除在首批自动 capability gate 外。内存分解证明 candidate 稳态 allocated peak 比 baseline 多一个逻辑输出大小，不再沿用 T-020 被长生命周期首轮输出放大的 5.90 MB 数值。仅修改审计脚本、结果和文档，未修改功能源码或共享环境；host launcher 使用审计垫片的结果必须始终带环境限制标签。
- 目标：解释 T-020 large 的高方差/长尾是 NPU task 本体、tile 选择还是 host/同步采样造成，并把 candidate additional peak 拆分为 pure output allocation、kernel/runtime 附加与 baseline 临时量。
- cohort：固定 fp16/contiguous/static large `(M,K1,N,K2)=(512,768,640,384)`；候选 tile 为 `64x64x64`、`64x128x64`、`128x128x128`。中间 tile 用来区分 output grid 并行度与 K-loop 成本，不一次性扩展大规模 autotune 空间。
- profiler：每个 tile fresh process，先正确性和普通 warmup 10，再 Level0/AiCoreNone、profile warmup 1、active 10、每 step 同步。必须得到 10/10 单 task，报告 kernel duration mean/stdev/p50/p99；不用含显式同步的 step gap 代替 task duration。
- paired 性能：profiler 正确的 tile 再与当前 fallback 各做 fresh-process warmup 10、runs 100、3 轮交错。报告每轮 mean±stdev/p50/p99，同时比较 profiler device duration 与 host synchronized sample；如果 device duration 稳定而 host 长尾，large 性能 gate 仍不自动放行，但根因分流到 host/runtime 稳定性。
- 内存：同一 worker 内分别采集 `torch.empty(M,N)`、baseline 和每个 candidate tile；每项 warmup 后 reset peak，记录 allocated/reserved before、max allocated/reserved 和 delta，输出每次删除并同步。若 runtime API 不提供 reserved peak，明确记为 unavailable，不伪造 0。
- 隔离：每个配置前后复查目标卡进程表；出现外部进程则保留原结果但不用于 verdict，不终止他人任务。测试从 `/home/z50063656/tmp` 启动，compiled baseline 设置 `TORCH_COMPILE_DEBUG=1`；成功不读 `output_code.py`。
- 停止条件：任一 tile 精度失败、非单 task、profile 不完整或隔离失效时不对该 tile 作性能结论。T-022 结束只决定 large 是否继续 hold 以及首个 formal config，不直接授权 T-021 源码实施。
- 产物：`t022_mmplus_different_k_large_profile.py`、`t022_mmplus_different_k_large_benchmark.py`、`results/t022_mmplus_different_k_large_20260821/` 与独立 Markdown 报告。
- E-048（large `64x64x64` profiler）：fp16 `(512,768,640,384)` 在空闲 NPU 7 完成，profile 前后 candidate max/mean absolute error 均为 `0.0625/8.7023e-7`，通过 fp16 预设容差。10 个 active step 导出 10 条同名 `different_k_mm_plus_mm_kernel`，每 step 一个 task；duration mean/stdev `29.280±0.598 μs`，p50/p99 `29.27/30.04 μs`。ACL→NPU flow 关联告警不影响完整 `kernel_details.csv`；不使用包含显式同步的 step gap。
- E-049（large `64x128x64` 两次受限执行无设备权限）：在运行前无进程的 NPU 7/NPU 6 各启动一次受限 fresh process，但 `libascend_hal.so:drvGetDevNum` 返回 `DRV_ERROR_INNER_ERR(7)`，`torch.npu.is_available()` 为 false；同一调用在显式 NPU 设备权限下可枚举 `Ascend910B2`，所以该现象属于执行沙箱设备节点权限，不是 tile 或共享运行时结论。原始证据保留在 `profile_bm64_bn128_bk64/result.json` 和 `profile_bm64_bn128_bk64_rerun1/result.json`，均不得计入正确性或性能结论。
- E-050（large `64x128x64` fresh launcher 编译环境阻塞）：显式设备权限下已能分配输入并进入 Triton launcher 初始化，但当前 PyTorch 为指向 `/home/z50063656/Benchmark/pytorch-upstream` 的 editable 安装，该源码树的 `torch/include` 不存在，首轮报 `ATen/ATen.h` 缺失。仅对单次进程设置 wheel 现有头文件 `CPATH` 后，编译继续暴露两个既有版本合同：PyTorch 2.14 `ATen` 要求 C++20，而安装的 Triton Ascend 3.2.0 launcher 固定传 `-std=c++17`；torch_npu installed header 还引用 CANN 9.0.1 headers 中没有声明的 `aclmdlRICondHandle/aclmdlRICondTaskParams`。证据分别保留在 `profile_bm64_bn128_bk64_rerun2/result.json` 和 `profile_bm64_bn128_bk64_rerun3/result.json`。没有 kernel 被执行，不能把它判为 tile 不支持；未重装、未建软链、未修改共享缓存或源码。
- E-051（large `128x128x128` 既有 launcher cache 不可复用）：在显式设备权限和运行前空闲 NPU 6 上设置官方 `TRITON_DISABLE_PRECOMPILE=1`，尝试跳过 GCH 并复用旧 launcher SO；当前 cache 中没有对应 SO，仍进入 launcher 编译并因 editable 源码树缺少 `ATen/ATen.h` 失败。原始证据保留在 `profile_bm128_bn128_bk128/result.json`，没有 kernel 执行，不计入 tile 结论。为继续只读性能审计，登记一个非产品、非安装态 workaround：在审计目录增加 compiler wrapper，仅将 Triton launcher 命令中的 `-std=c++17` 替换为 `-std=c++20`，并 `-include` 一个只含 `aclmdlRICondHandle` 与不透明 `aclmdlRICondTaskParams` 前置声明的兼容头；同时通过 `CPATH` 指向当前 wheel 头文件、`TRITON_CACHE_DIR` 指向本任务独立目录。两个 CANN 类型只为解析 torch_npu 间接头文件，launcher 不得引用或调用相关图 API；若编译、加载、正确性或 profiler 任一失败立即停止。该垫片不能作为产品修复或环境已兼容的证据。
- E-052（large `64x128x64` 审计垫片 profiler）：在运行前后进程表均为空的 NPU 6，以 `CC=t022_launcher_cc_wrapper.sh`、wheel headers `CPATH`、`TRITON_DISABLE_PRECOMPILE=1` 和 T-022 独立 `TRITON_CACHE_DIR` 完成；未修改安装环境或共享 cache。profile 前后 max/mean absolute error 均为 `0.0625/8.7023e-7`，通过 fp16 容差。10 个 active step 导出 10 条同名 kernel，每 step 一个 task；duration mean/stdev `24.408±0.349 μs`，p50/p99 `24.33/25.14 μs`，低于 `64x64x64` 的 `29.280±0.598 μs`。首次 compile+run `17,204.80 ms` 仅诊断隔离 cache 编译，不计稳态性能。结果位于 `profile_bm64_bn128_bk64_audit_shim/result.json`；由于 host launcher 使用了审计垫片，设备 kernel 数值与 duration 可用于 tile 比较，但不能证明正式环境 fresh compile 已可用。
- E-053（large `128x128x128` 审计垫片 profiler）：在与 E-052 相同的进程局部 wrapper、headers 和隔离 cache 合同下，于运行前后进程表均为空的 NPU 6 完成。profile 前后 max/mean absolute error 同为 `0.0625/8.7023e-7`；10/10 active step 均为一个 `different_k_mm_plus_mm_kernel`。duration mean/stdev `14.448±0.329 μs`，p50/p99 `14.41/14.98 μs`，明显低于 `64x128x64` 的 `24.408 μs` 与 `64x64x64` 的 `29.280 μs`。首次 compile+run `15,635.05 ms` 仅诊断。结果位于 `profile_bm128_bn128_bk128_audit_shim/result.json`。三 tile 的 task duration 均稳定，说明 T-020 large 的数百微秒 host synchronized 时延和毫秒长尾不来自 candidate device kernel 本体；仍需 paired host 采样确认 wrapper/runtime 影响。
- E-054（large `64x64x64` paired 与内存分解）：在运行前后进程表均为空的 NPU 6，baseline/candidate 各 warmup 10、runs 100、3 轮交错；p50 中位数 `0.299175/0.279320 ms`，改善 `6.64%`，p99 `0.324950/0.299940 ms`，改善 `7.70%`，低于预设 10% gate。三轮 baseline mean±stdev 为 `0.304933±0.007842`、`0.304547±0.017242`、`0.300130±0.008764 ms`，candidate 为 `0.281282±0.006143`、`0.279953±0.006106`、`0.280669±0.006696 ms`，未复现 T-020 毫秒级长尾。baseline benchmark 前后误差为 0，candidate 为 `0.0625/8.7023e-7`，均通过。additional allocated peak 的 pure output/baseline/candidate 为 `655,872/1,311,744/1,967,104 B`；三者 additional reserved peak 都为 0，因采样前 allocator 已预留 `73,400,320 B`，不能解释为零 workspace。编译期提示当前设备 SM 数不足以采用 max-autotune GEMM，数据代表实际 fallback；baseline 成功，按图模式规则未读取 `output_code.py`。结果位于 `benchmark_bm64_bn64_bk64_audit_shim/result.json`。
- E-055（large `64x128x64` paired 与内存分解）：同一运行前后空闲 NPU 6 和相同采样合同下，p50 中位数 `0.272655/0.253640 ms`，改善 `6.97%`；p99 `0.300030/0.274450 ms`，改善 `8.53%`，仍低于 10% gate。baseline 三轮 mean±stdev `0.280695±0.027991`、`0.276082±0.008541`、`0.275959±0.009252 ms`，candidate `0.256691±0.005624`、`0.255918±0.005711`、`0.255787±0.009913 ms`；只有 baseline 第 1 轮单点 max `0.54721 ms`，没有毫秒级长尾。正确性与 E-054 一致。pure output/baseline/candidate additional allocated peak 仍为 `655,872/1,311,744/1,967,104 B`，reserved delta 为 0。profiler device p50 `24.33 μs` 对应 paired host p50 `253.64 μs`，说明约九成同步样本是 host launch/synchronize/runtime 固定成本，tile 本体改善被稀释。baseline 成功，未读取 `output_code.py`。结果位于 `benchmark_bm64_bn128_bk64_audit_shim/result.json`。
- E-056（large `128x128x128` paired、内存与 T-022 verdict）：同一运行前后空闲 NPU 6 和相同采样合同下，p50 中位数 `0.275920/0.255100 ms`，改善 `7.55%`；p99 `0.292130/0.271480 ms`，改善 `7.07%`，仍低于 10% gate。baseline 三轮 mean±stdev `0.276201±0.004962`、`0.277245±0.004658`、`0.278002±0.003981 ms`；candidate `0.257691±0.015490`、`0.255106±0.003757`、`0.256136±0.003797 ms`，首轮仅一个 max `0.40326 ms`，没有 T-020 多毫秒长尾。正确性与前两 tile 一致，内存三项也完全一致。`128x128x128` device p50 仅 `14.41 μs` 而 host p50 `255.10 μs`，device kernel 约占同步样本 5.6%；不能把 tile 的 2 倍 device 加速直接外推为端到端收益。三 tile candidate additional allocated peak 都为 `1,967,104 B`，baseline `1,311,744 B`，差值 `655,360 B` 恰为 logical fp16 output bytes；pure output allocator 实测 `655,872 B`（含对齐）。T-020 采样时首个 baseline/candidate output 在 paired rounds 前持续存活，使峰值被放大；T-022 在采样前删除并同步后得到更可解释的 steady decomposition。reserved API 可用但三项 delta 都为 0，因为采样前已 reserved `73,400,320 B`，不得宣称无 workspace。结论：large 功能支持且 device kernel 稳定，但端到端仅 `supported-neutral`，继续 hold；若未来采用该 tile，`128x128x128` 是 device-profile 首选而非 large 自动启用证据。baseline 成功，未读取 `output_code.py`。结果位于 `benchmark_bm128_bn128_bk128_audit_shim/result.json`。

### T-023：mm_plus_mm different-K 首批 default-off 正式接入

- 状态：`installed-functional-performance-verified-memory-tradeoff-environment-pending`；修复版 source-built wheel 已 `--no-deps` 安装，结构 UT 6/6、row/column 正向、AOTAutograd backward 与五项 capability negative 已通过；shape-A/unaligned 集成 p50 改善 `15.29%/18.04%`，但 Ascend Triton workspace 使 strict memory gate 失败。实现保持默认关闭、output 上限与 extern fallback；只剩匹配 headers 的正式无 shim launcher 复验。
- 目标：把 standalone different-K kernel 变成 torch_npu default Inductor backend 的可选 `NPUTritonTemplate` choice；不改变 same-K、CPU、experimental backend 或 rollout-off 行为，并始终保留支持 different K 的 `aten_mm_plus_mm` extern fallback。
- 首批开关：新增 `TORCHINDUCTOR_NPU_ENABLE_DIFFERENT_K_MM_PLUS_MM`，默认 false。新增 `TORCHINDUCTOR_NPU_DIFFERENT_K_MM_PLUS_MM_MAX_OUTPUT_ELEMENTS`，默认 `131072`；无效或非正值退回默认值并告警。该 power-of-two 成本 gate 覆盖 T-016/T-020 已验证的约 61K-output beneficial cohort，排除 T-022 的 327,680-output neutral large，不绑定具体 M/N/K shape 白名单。未来扩门必须新增性能证据。
- Pattern capability：仅 backend=`default`、NPU、4 个 2D meta tensor、同 device/dtype、dtype 为 fp16/bf16/fp32、static positive M/N/K、K1!=K2、M/N 相同、每个输入为精确 row-major 或 column-major stride、`M*N<=max_output_elements` 且上游 `is_valid_mm_plus_mm()` 通过时触发。dynamic、empty、mixed dtype/device、arbitrary stride、same-K、large 和其他 backend 均保持原图/fallback。
- Template：新增独立 K1/K2 reduction loop 与各自 tail mask；四组真实 stride；两个 fp32 accumulator 分别 cast 到 output dtype 后相加；首个且唯一候选 config 为已验证 `BLOCK_M/N/K=128/128/128`、`num_stages=1`、`num_warps=4`。使用 `NPUTritonTemplate` 和 scheduler/wrapper，不在 lowering 中直接调用 eager Triton。
- Choice/fallback：`tuned_npu_mm_plus_mm()` 先无条件绑定 `aten_mm_plus_mm` extern，再尝试 append NPU template。template render/compile 无效时 choices 仍含 extern；`NoValidChoicesError` 回到 extern output。不得依赖 generic ATEN backend 配置才能获得 fallback。
- 幂等与 backend 切换：新 `LoweringPatternEntry` 通过 handler identity 在 `post_grad.pass_patterns[1]` 中查重；pattern 即使在同进程 backend switch 后仍存在，extra check 也要求 `TORCHINDUCTOR_NPU_BACKEND=default`，不得污染 `triton_experimental`/CPU。
- 拟修文件：新建 `torch_npu/_inductor/kernel/mm_plus_mm.py`；修改 `kernel/__init__.py`、`fx_passes/post_grad.py`、`torch_npu/_inductor/__init__.py`、`config.py`；新建 `test/_inductor/test_mm_plus_mm.py`。不改 PyTorch、Triton Ascend、C++ dispatcher、AOTI shim和 T-011 `strict_sum`。
- 验证顺序：先 `py_compile`、lint 和无 NPU kernel 的结构/extra-check/幂等/fallback unit；再从 `/home/z50063656/tmp`、`TORCH_COMPILE_DEBUG=1` 运行 rollout off/on、static positive、same-K/dynamic/large/empty/stride negatives 和 AOTAutograd。失败 compiled case 必须读本轮 `output_code.py`；成功不读。当前 fresh launcher 需审计 shim 才能运行的结果只能作为开发诊断，最终必须在 headers/编译标准匹配的环境重建 torch_npu wheel、`--no-deps` 安装并无 shim 复验。
- 性能 gate：正式 compiled template 需重测 T-016/T-020 beneficial cohort，各 warmup 10、runs 100、3 轮交错，报告 mean±stdev/p50/p99、allocated/reserved peak、device task；任一首批配置 p50 不超过 10%、p99/内存明显回退或 selector 不能可靠 fallback，则保持开关默认关闭并缩小/撤销 capability。
- 回滚：默认关闭开关是运行时最小回滚；代码回滚只撤销上述 5 个 torch_npu Python 文件的新内容与新 UT，不撤销 lowering.py 的 T-011 修改，不清理既有未跟踪生成物或 Triton 外部修改。

- E-057（T-023 源码复核）：上游 pattern 注册 API 支持对同一 `pass_patterns[1]` 追加独立 lowering handler；当前 torch_npu default loader 在 `patch_pattern_mm_plus_mm()` 后安装 algorithm selector，适合在两者之间注册 NPU pattern。`NPUTritonTemplate.generate()` 已提供 NPU benchmark request、32-bit indexing和 wrapper render；`maybe_append_choice()` 可让模板生成错误不删除既有 extern choice。`restore_inductor_baseline()` 只恢复 lowering/scheduler hook，不恢复 post-grad registry，因此新 pattern 必须按 handler identity 查重并在 extra check 中显式限制 default backend。当前相关 tracked diff 仍只有 T-011 `torch_npu/_inductor/lowering.py`，本任务不会覆盖已有修改。
- E-058（T-023 首轮结构单测）：从 `/home/z50063656/tmp` 启动 Benchmark 环境，先加载已安装 torch_npu native package，再仅在测试进程内把源码 `torch_npu` 目录前置到 package path；没有写入 site-packages 或替换 wheel。6 项结构/extra-check/template/fallback 单测中 5 项通过，幂等测试因断言“一个 pattern 等于一个 registry entry”失败：实际 matcher 会为同一 handler 展开 17 个 `LoweringPatternEntry`。handler identity 查重已生效，错误只在 UT 期望；修正为首次计数大于 0、再次调用返回 false 且计数不增长，然后重跑完整目标文件。该轮没有执行 NPU kernel，不生成或检查 `output_code.py`。
- E-059（T-023 结构单测闭环）：修正幂等断言后，同一无持久修改的源码 path overlay 下目标文件 6/6 通过，用时 `5.72 s`。本轮验证的是 Python 结构合同而非已安装 wheel；正式 compiled 验证仍需重建。静态检查累计为六文件 `py_compile`、`git diff --check`、FLAKE8/NEWLINE/SPACES/TABS/COPYRIGHT 定向 lint 和结构 UT 全部通过。
- E-060（T-023 wheel 构建边界）：最终运行环境以 `/home/z50063656/Benchmark/env.sh` 激活的 `benchmark-py311` 为准，不能直接复用硬编码旧 `Pass` Conda 路径的 T-011 helper。新增 T-023 独立 helper，只把 PyTorch install view 与 torchgen packaged 路径切换到 `envs/benchmark-py311`，仍从 `/home/z50063656/tmp` 发起、仅在构建进程排除旧路径遮蔽、临时创建并最终移除 `torchgen/packaged` symlink、禁用原 wheel 不含的 TorchAir/RPC、隔离 legacy egg-info、恢复 ACL 子模块。覆盖同名 dist wheel 前，把 SHA256 为 `263ffec23...2792704` 的当前 wheel 复制为任务内回退 artifact；构建后必须比较 archive entry 集、确认 T-011 与 T-023 Python 修改均入包，验收通过后才允许 `pip install --no-deps --force-reinstall`。
- E-061（T-023 首次 wheel 重建阻断）：构建在 codegen/CMake 前因 `tools.setup_helpers.version` 被 Benchmark PyTorch 的顶层 `tools` package 遮蔽而退出。旧 dist wheel SHA256 仍为 `263ffec23...2792704`，临时 torchgen link 与 `/tmp/t023_torch_npu_build_*` 均不存在，三个 ACL 子模块恢复到登记 commit。与 T-011 不同，本轮 Benchmark 路径正是 PyTorch editable source，不能在导入 torch 前删除；helper 改为先导入并校验 `torch` 来自该 source，再仅从当前父构建进程 `sys.path` 移除 Benchmark root、前置 torch_npu root，随后执行 setup。只读探针确认导入 torch 后顶层 `tools` 尚未进入 `sys.modules`，所以不需删除已加载模块或修改 `.pth`。
- E-062（T-023 第二次 wheel 重建主动中止）：修正 package 遮蔽后已完成 codegen 并进入 Ninja，但配置日志显示 include/PYTORCH_INSTALL_DIR 已切到 `benchmark-py311`，`TORCH_LIBRARY`、`c10_LIBRARY`、`Torch_DIR` 和 Python cache 仍保留旧 `Pass` 绝对路径。两环境 `libtorch.so` SHA256 相同，但混合 cache wheel 不满足最终环境证据要求，因此在 Ninja `20/1224` 主动中止，dist wheel 仍为旧 SHA。中止信号先于 helper `finally` 完成；经精确枚举后，仅移除本轮创建的 torchgen symlink 和只含源码软链的 `/tmp/t023_torch_npu_build_94fa77h5`，并用 `git submodule update --init --checkout` 恢复三个 ACL 子模块。未触碰 Tensorpipe/DVM 或既有未跟踪生成物。
- E-063（T-023 CMake cache 对齐）：保留现有 compiled objects，但在 setup 前用 CMake `-U` 精确清除可跨环境污染的 Torch/Caffe2/Python/ninja/pkg-config cache key，以及 TorchAir/Tensorpipe 旧开关；同次 configure 显式传入当前 `benchmark-py311` 的 `PYTORCH_INSTALL_DIR`、`CMAKE_PREFIX_PATH`、`TORCHAIR_TARGET_PYTHON`。重跑必须先在日志或 cache 中确认 Torch/native library/Python 路径不再指向 `envs/Pass`；若仍混用则停止，不安装。
- E-064（T-023 wheel 构建与安装前验收）：第三次构建在 `benchmark-py311` 对齐 cache 下完成 1224 个 Ninja 目标、链接和 bdist_wheel，退出码 0。新 wheel SHA256 为 `39834f4fe176e640e69ac0da56ec5074f616ae7a3af9963a5235f9c3c35fa8fa`；旧回退 artifact 仍为 `263ffec23...2792704`。旧/新 archive entry 为 `1317/1318`，排序路径集合唯一新增 `torch_npu/_inductor/kernel/mm_plus_mm.py`；不含 TorchAir、`libtensorpipe.so` 或 legacy `torch_npu.egg-info`。六个目标 Python 文件逐字节等于当前 source，wheel 同时包含 `_C` 与 `libtorch_npu.so`，T-011 `lowering.py` 也保留。CMake cache 的 ninja、Caffe2、PyTorch install、Torch/c10 native library 与 target Python 均指向 `envs/benchmark-py311`。临时 torchgen link/workspace不存在，ACL 子模块已自动恢复。安装闸门通过，下一步只允许在 Benchmark 环境执行该 wheel 的 `pip install --no-deps --force-reinstall`。
- E-065（T-023 安装与 installed-package smoke）：新 wheel 已在 `benchmark-py311` 使用 `pip install --no-deps --force-reinstall` 安装，pip 未解析或替换依赖。`/home/z50063656/tmp` 中显式 NPU 访问 smoke 确认 torch 为 `2.14.0a0+git8e86e0a` editable Benchmark source、torch_npu 为 site-packages 中的 `2.14.0a0+git83cc452`、`torch.npu.is_available()` 为 true；未设置 rollout 环境变量时 `enable_different_k_mm_plus_mm=False`，默认上限为 `131072`。随后不使用 source overlay 的 installed-package 结构 UT 为 6/6 通过（`9.96 s`）；未构建 TorchAir 的 warning 与原 wheel 组件边界一致。
- E-066（T-023 compiled 功能探针设计）：新增独立 fresh-process 探针，首轮按 rollout-off positive、rollout-on positive/column-major、same-K/large/empty/arbitrary-stride/dynamic negatives 与 AOTAutograd backward 顺序执行。每个 worker 在 import 前设置 rollout，固定 default backend、`max_autotune/max_autotune_gemm=True`，比较 eager 输出（backward 同时比较四个输入梯度），并用进程局部 wrapper 记录新 template 的 append choice 与 selector timing，不修改产品代码。每项使用独立 Inductor/Triton cache 和 `TORCH_COMPILE_DEBUG=1`；失败项读取本轮 `output_code.py`，成功项不读。当前所有物理 NPU 均有外部进程，NPU 7 仅有约 596 MiB worker 但仍不视为空闲；先完成脚本静态检查，不启动 NPU case，不终止他人任务。
- E-067（T-023 rollout-off/no-shim smoke）：探针 `py_compile`、120 列检查、`--help` 和源码 `diff --check` 通过后，在当时占用最小但仍有外部 worker 的物理 NPU 1 只运行非性能 small positive。fresh isolated cache、debug on、无 compiler shim/CPATH，结果为 `compile-correct`：rollout/effective config 均 false，template event 为空，输出 max/mean error 均 0，首次 compile+run `5343.17 ms`，峰值 allocated `1,380,864 B`。成功 case 按规则未读 `output_code.py`；该共享设备耗时和内存只作功能诊断，不作性能结论。worker 退出后复查 NPU 1 只剩原外部 PID；此时物理 NPU 4 已完全空闲，后续功能 case 切到 NPU 4。探针原字段只取 distribution metadata，显示 Triton `3.5.0`；为避免与实际 import module 混淆，后续同时记录 `triton.__version__/__file__` 和 distribution version，不回写或伪造已完成原始结果。
- E-068（T-023 rollout-on 首轮 candidate 编译失败与安全 fallback）：物理 NPU 4 空闲时，small positive 在 rollout on、fresh isolated cache、debug on、无 shim 下成功匹配新 handler，并把 choices 从 1 个 extern 扩到 2 个；但生成模板把一维 `offs_m/offs_n` 直接传给 `store_output`，渲染出一维 `xindex = offs_n + 320*offs_m`，随后要求 `broadcast_to([128,128])`，Triton 报 `Cannot broadcast, rank mismatch: [128], [128, 128]`。autotune 把 candidate 记为 `inf`，extern 为 `0.1327 ms`，最终图输出误差仍为 0。该结果只能判为 `candidate-failed-safe-fallback-correct`，不能判 rollout-on 可用。按失败规则已读取本轮 `output_code.py`：最终代码只调用一个 `extern_kernels._mm_plus_mm`，确认坏 template 未进入执行图；同时读取失败 cache 中生成模板定位根因。修正为显式 `idx_m = offs_m[:, None]`、`idx_n = offs_n[None, :]` 后再传 `store_output`；UT 增加二维 index 断言，探针把 selector 返回 choice 记入事件，并要求 eligible positive 实际选择 `TritonTemplateCaller`，防止安全 fallback 再被误算成功。修复需再次重建/验收/`--no-deps` 安装 wheel。
- E-069（T-023 store index 修复静态闭环与重建边界）：修复后目标三文件 `py_compile`、源码 `diff --check`、探针 120 列检查通过；kernel/UT 定向 FLAKE8/NEWLINE/SPACES/TABS/COPYRIGHT lint 无问题；源码 path overlay 结构 UT 6/6 通过（`7.99 s`）。当前 site-packages 仍是首版坏模板，不用于复验。覆盖 dist 前将 SHA256 `39834f4f...5fa8fa` 的首版 T-023 wheel 保存为 `artifacts/torch_npu_t023_before_store_index_fix.whl`；随后复用 E-063 的 Benchmark-aligned helper，重新比较 archive 路径、目标 Python 内容与组件边界，通过后再 `--no-deps --force-reinstall`。
- E-070（T-023 store index 修复版 wheel 验收）：Benchmark-aligned 重建退出码 0，新 wheel SHA256 为 `d0ee10794f8cb63d528c86f27294a2a52a4b8b5f484eb6be53323d22b2157718`。与首版 T-023 wheel 都是 1318 个 archive entries，排序路径集合无差异；六个目标 Python 文件逐字节等于当前 source，且仍不含 TorchAir、`libtensorpipe.so` 或 legacy egg-info。临时 torchgen link/workspace 已清除，ACL 子模块恢复。修复版安装前闸门通过，允许 `--no-deps --force-reinstall`，随后先重跑 installed-package UT 和同一 no-shim rollout-on positive fresh case。
- E-071（T-023 修复版 installed/no-shim 环境阻断）：修复版已 `--no-deps` 安装，installed-package UT 6/6 通过（`15.14 s`）。随后 shared NPU 7 上的 positive fresh case 已越过二维 index/Triton IR 编译，生成模板明确为 `idx_m[128,1] + 320*idx_n[1,128]` 的二维 `xindex`；但 launcher GCH 从 editable PyTorch 的不存在路径 `/home/z50063656/Benchmark/pytorch-upstream/torch/include` 查找 `ATen/ATen.h`，命令仍固定 `-std=c++17`，candidate timing 为 `inf`、extern 为 `0.1154 ms`。探针因 selector 未返回 `TritonTemplateCaller` 正确标为 error。按失败规则读取 `output_code.py`，最终图仅含单个 `extern_kernels._mm_plus_mm`，安全 fallback 仍正确。该阻断与 E-050/E-051 的环境合同相同，重建 torch_npu wheel 不能补齐 editable PyTorch include view 或修改 Triton launcher 标准。下一轮仅作开发态功能复验：复用 T-022 已登记的 audit-only C++20/header wrapper、当前 Benchmark torch/torch_npu wheel headers 和全新 cache；不得把结果记为正式 no-shim 环境通过。

- E-072（T-023 audit-shim 正向融合与探针延迟选择修正）：shared NPU 7 上复用 T-022 的进程局部 C++20/header wrapper、wheel headers、`TRITON_DISABLE_PRECOMPILE=1` 和全新隔离 cache 后，small positive 的 choices 从一个 extern 扩展为 extern+NPU template；autotune 诊断值为 template `0.0089 ms`、extern `0.0908 ms`，图输出 max/mean absolute error 均为 0。selector wrapper 返回 `selected_choice=None`，探针原断言因此把本轮标为 error；按失败规则读取本轮 `output_code.py`，最终 Runner 明确调用 `triton_npu_different_k_mm_plus_mm.run(..., 6, 1, 1, stream=...)`，没有 extern 执行调用，证明 template 已经由 `MultiTemplateBuffer` 延迟选择并实际执行。该异常属于审计脚本判定错误，不是产品功能错误。修正探针：eligible case 仍必须 append 至少两个 choices；selector 显式返回 choice 时必须是 `TritonTemplateCaller`，返回 `None` 时记录 `selection_mode=deferred` 并由编译正确性继续约束，不再误报。由于设备共享且使用 audit shim，autotune 数值只作功能诊断，不作正式性能结论；修正后必须用新输出目录重跑，成功 case 按图模式规则不再读取 `output_code.py`。
- E-073（T-023 延迟选择探针复验通过）：修正后的 fresh small positive 在运行前无进程的物理 NPU 7 上完成，结果为 `compile-correct`；rollout 环境与 effective config 均为 true，choices 为 extern+`TritonTemplateCaller`，`selection_modes=["deferred"]`，输出 max/mean absolute error 均为 0。autotune 诊断为 template `0.0078 ms`、extern `0.1021 ms`，首次 compile+run `24,934.74 ms`，编译期峰值 allocated `202,758,144 B`；这些值不属于正式 steady benchmark。worker 退出后 NPU 7 出现非本任务 PID、占用 124 MiB，因此本轮只计功能通过，不计隔离性能证据。case 成功，按图模式规则未读取 `output_code.py`。下一步在共享最小卡上逐个运行 negative capability；所有 negative 都必须没有 template append，正向 column/backward 才使用 audit shim。
- E-074（T-023 capability negative 批次）：在 rollout on、default backend、fresh 独立 cache、debug on 且无 launcher shim 的条件下，物理 NPU 7 顺序完成 same-K、large、empty、arbitrary-stride 和 dynamic 五项；全部为 `compile-correct`，输出 max/mean absolute error 均为 0，五项 `template_events` 都为空，证明首批 pattern 没有越过 K 不同、output 上限、非空、精确 row/column stride 和 static gate。dynamic 首轮 `(192,320,256,128)` 与 replay `(200,328,264,136)` 都正确且没有 template append；empty 输出为 `(0,320)`；large `(512,768,640,384)` 继续走原路径。五项首次 compile+run 分别约 `4.35/3.46/3.53/4.11/6.29 s`，仅作功能诊断。批次期间 NPU 7 始终有外部小进程且结束时占用 546 MiB，因此不作性能或内存 verdict；成功 case 均未读取 `output_code.py`。
- E-075（T-023 column-major 与 AOTAutograd）：在 shared NPU 7、audit-only launcher shim、fresh 独立 cache 和 debug on 下，真实 column-major 正向及 backward 两项均为 `compile-correct`。column 四输入 stride 为 `[1,192]`、`[1,256]`、`[1,192]`、`[1,128]`；其输出 max/mean absolute error 为 0，choices 从一个 extern 扩到 extern+template，selection 为 deferred。backward 前向同样追加并延迟选择 template；前向输出及四路输入梯度的 max/mean absolute error 全为 0，AOTAutograd 同时生成 forward/backward 图且未暴露梯度路径错误。两项 autotune 仅诊断：column template/extern 为 `0.0108/0.0944 ms`，backward 的前向 template/extern 为 `0.0120/0.0404 ms`；首次 compile+run 为 `27.63/20.77 s`，编译期 peak allocated 为 `202,758,144/204,067,328 B`，均不作 steady 性能或内存结论。成功 case 按规则未读取 `output_code.py`。至此 T-023 功能边界批次完成，剩余闸门为独占卡上的集成 profiler、paired 100-run×3-round 性能与 steady memory，以及正式无 shim 环境合同。
- E-076（T-023 集成性能验证合同）：新增一个审计目录内的统一 worker，不修改产品源码。覆盖 fp16 contiguous static 的 shape-A `(192,256,320,128)` 与 unaligned `(191,255,319,127)`；二者都在首批 output gate 内且已有 T-015/T-016 standalone 预算。worker 在 import 前固定 default backend、关闭 Inductor 磁盘 cache，注册一次 selector 事件记录；同进程先把运行时 rollout config 设为 false 编译独立 baseline callable，再设为 true 编译独立 candidate callable，分别断言 baseline 无 template append、candidate 有 extern+template choice 且显式选 template或 deferred，避免跨进程 allocator/host 噪声。profile 模式 warmup 10、profile warmup 1、active 10，每 step 同步，要求 10/10 单一 device task，并报告 kernel duration mean/stdev/p50/p99。benchmark 模式 baseline/candidate warmup 10、runs 100、3 轮交错，报告每轮 mean±stdev/p50/p99；随后删除所有首轮输出并同步，再分解 pure output、baseline、candidate 的 allocated/reserved before/max/delta。p50 改善至少 10%、p99 不回退、candidate additional allocated 不超过 baseline 加一个 pure output allocation，才可通过首批性能 gate。所有运行必须从 `/home/z50063656/tmp` 发起、使用 fresh debug/cache、前后检查目标卡进程；仅独占结果进入 verdict。当前 audit-only C++20/header shim 只影响 fresh host launcher 编译，允许用于 device task 和 steady runtime 诊断，但不能关闭正式 no-shim 可用性 blocker。compiled case 成功不读 `output_code.py`，失败才按图模式规则读取。
- E-077（T-023 shape-A 集成 profiler）：新增 worker 的 `py_compile`、120 列检查和 `--help` 通过后，在运行前后都无进程的物理 NPU 7 完成首个正式 profile。rollout-on candidate 追加 extern+template choice、selection 为 deferred，profile 前后输出 max/mean absolute error 均为 0；10 个 active step 精确得到 10 条单一 `triton_npu_different_k_mm_plus_mm` task。device duration mean/stdev `8.430±0.199 μs`，p50/p99 `8.47/8.68 μs`，范围 `8.02–8.68 μs`，与 T-015 standalone shape-A `8.714±0.337 μs` 一致，证明正式 Inductor template 没有引入额外 device task。首次 compile+run `20,885.03 ms` 是关闭 cache 后的编译诊断，不计稳态。ACL→NPU flow 关联告警不影响 10 条完整 kernel CSV。使用 audit shim，因此本条可作为 device task/duration 和集成功能证据，但不关闭 no-shim blocker。case 成功，未读取 `output_code.py`；允许继续 unaligned profiler。
- E-078（T-023 unaligned profiler 隔离失效、保留不计）：unaligned worker 启动前物理 NPU 7 无进程，但结束复查时出现非本任务 PID 2372300、占用 378 MiB，且 AICore 为 100%；因此本轮 autotune 和 profiler 不进入正式 verdict。受干扰的 autotune 把 extern/template 计为 `0.0829/0.0993 ms`，`MultiTemplateBuffer` 延迟选择了 extern；10 个 active step 共 30 个 task，分组为 10 个 `aclnnMm`、10 个 addmm 内部 matmul 和 10 个 add，profile gate 为 false。profile 前后输出 max/mean absolute error仍为 0，说明安全 fallback 正确，但不能据此判 template 在 unaligned 上回退。结果原样保留在 `profile_unaligned_audit_shim/result.json`，成功 compiled case 未读取 `output_code.py`。下一步必须在前后都空闲的设备用全新输出/cache 重跑；重跑未得到 10/10 单 template task前，不启动 unaligned paired benchmark。
- E-079（T-023 unaligned 集成 profiler 有效重跑）：迁移到运行前后都无进程的物理 NPU 1，以全新 output/cache 重跑。autotune 的 template/extern 为 `0.0160/0.0952 ms`，与 E-078 的反向选择形成清晰隔离对照；最终 10 个 active step 精确得到 10 条单一 `triton_npu_different_k_mm_plus_mm` task，profile 前后输出 max/mean absolute error均为 0。device duration mean/stdev `13.264±0.149 μs`，p50/p99 `13.29/13.46 μs`，范围 `13.06–13.46 μs`，与 T-015 standalone unaligned `13.824±0.541 μs` 一致。首次 compile+run `20,985.90 ms` 不计稳态；ACL→NPU flow 告警不影响 CSV。至此 shape-A/unaligned 集成 profiler 都通过一个 task/step gate，允许按 E-076 在独占卡上运行 paired benchmark。成功 case 未读取 `output_code.py`；audit shim 证据边界不变。
- E-080（T-023 shape-A 首轮 paired worker 自污染、合同修正）：物理 NPU 1 运行前后均无进程，首轮结果数值正确，但不能进入性能 verdict。worker 先以 rollout false 编译 baseline，再在同一进程把 config module 改为 true 编译 candidate；测量阶段首次调用 baseline callable 时，Dynamo 对该全局 config guard 触发重编译，控制台出现第二次 `npu_different_k_mm_plus_mm` autotune 和第三个 compiled graph。结果中 baseline/candidate additional allocated peak 都为 `517,120 B`，p50 只差 `2.33%`，实际比较已被污染为 candidate 对 candidate。成功图不读取 `output_code.py`；证据原样保留在 `benchmark_shape_a_audit_shim/result.json`，不解释为集成收益回退。修正 E-076 的 paired 实现：每个 shape 的每个 variant/round 使用独立 fresh process，rollout 环境变量在 import torch 前固定且进程内禁止切换；顺序为 `baseline1→candidate1→candidate2→baseline2→baseline3→candidate3`，每个 worker warmup 10、runs 100，并各自记录正确性和 pure-output/variant memory。独立纯 Python aggregator 只读取六个 JSON，汇总三轮 mean±stdev/p50/p99 和内存中位数；任何 worker 发生 late compile、选择 fallback、设备隔离失效或 schema 不一致则整组不出 verdict。
- E-081（T-023 shape-A fresh-process paired 与 memory gate）：修正 worker/aggregator 的 `py_compile`、120 列检查和帮助入口通过；物理 NPU 1 在六个 worker 的批次前、各阶段和批次后均无进程。三轮 baseline mean±stdev 为 `0.299987±0.007356`、`0.308665±0.006629`、`0.306043±0.008170 ms`，p50 为 `0.298025/0.306455/0.303280 ms`；candidate 为 `0.272525±0.009607`、`0.243928±0.004880`、`0.257973±0.004274 ms`，p50 为 `0.268820/0.242185/0.256920 ms`。三轮中位数 baseline/candidate p50 `0.303280/0.256920 ms`，改善 `15.29%`；p99 `0.327100/0.268340 ms`，改善 `17.96%`，性能门通过。所有 worker 前后正确性 max/mean absolute error 为 0；三个 candidate worker 各自额外 profile 一个 active step，均为单一 `triton_npu_different_k_mm_plus_mm` task，duration `8.24/8.54/8.44 μs`，排除 deferred selector 退回 extern。steady additional allocated peak 三轮完全一致：pure output `123,392 B`、baseline `246,784 B`、candidate `517,120 B`；reserved delta 都为 0，但 candidate 的 `517,120 B` 高于 baseline+一个 pure output 的 `370,176 B`，内存 gate 失败，因此 shape-A 总 gate 暂为 false。该差异不得被 15.29% 加速掩盖，需与 unaligned 一并确认并定位 planner/template 临时分配。结果位于 `workers_shape_a/` 与 `aggregate_shape_a/result.json`；成功 case 均未读取 `output_code.py`，audit shim 边界不变。
- E-082（T-023 unaligned fresh-process paired 与重复 memory blocker）：同一物理 NPU 1 在六 worker 前、中、后均无进程。baseline 三轮 mean±stdev `0.302327±0.007373`、`0.296357±0.006289`、`0.307756±0.007472 ms`，p50 `0.299215/0.294015/0.305460 ms`；candidate `0.247445±0.004905`、`0.244399±0.004229`、`0.269649±0.229779 ms`，p50 `0.245530/0.243485/0.245225 ms`。candidate 第 3 轮含一个 `2.54338 ms` max 孤立长尾，原样保留；该轮 p99 为 `0.29205 ms`，前两轮为 `0.26255/0.25872 ms`。三轮中位数 baseline/candidate p50 `0.299215/0.245225 ms`，改善 `18.04%`；p99 `0.324840/0.262550 ms`，改善 `19.18%`。三个 candidate 末尾单-step profiler 都是唯一融合 task，duration `13.36/13.32/13.42 μs`；所有正确性误差为 0。additional allocated peak 稳定为 pure output `122,368 B`、baseline `244,736 B`、candidate `516,096 B`，reserved delta 仍为 0；candidate 再次超过 baseline+一个 output，内存 gate 失败，整组总 gate false。shape-A candidate peak `517,120 B` 接近四个输入 logical bytes `393,216 B` 加一个 aligned output `123,392 B`；unaligned 也呈相同量级，优先沿 template wrapper 的 `copy_if_misaligned` 输入复制路径做只读根因定位。结果位于 `workers_unaligned/` 和 `aggregate_unaligned/result.json`；成功 case未读取 `output_code.py`。
- E-083（T-023 memory-history 诊断合同）：上游 `copy_if_misaligned` 的 C++ fast path只在实际 `data_ptr % 16 != 0` 时 clone；从 `/home/z50063656/tmp` 在空闲 NPU 1 新分配 shape-A/unaligned 共 8 个 fp16 输入，所有实际地址 `%16 == 0`、storage offset 为 0，因此 E-082 的峰值算术相似性不能当作输入复制证据。下一步只扩展审计 worker 的 `memory-trace` 模式，不改产品源码：baseline/candidate 各用 import 前固定 rollout 的 fresh process，编译并 warmup 10 后启用 NPU allocator history，围绕一次同步调用放置 snapshot 边界，记录边界内 alloc/free 事件的 size/address 与截断 Python frames，同时记录 allocated/reserved before/max/delta 和正确性。若 history 显示 candidate 真实申请额外临时块，再按栈定位 template/wrapper；若只是一块输出但 peak API 受 awaiting-free/stream 影响，则修正 memory gate 解释而不是改 kernel。成功 compiled case仍不读 `output_code.py`。
- E-084（T-023 memory-history 首轮审计脚本边界失败）：shape-A baseline 已编译并正确运行，但 NPU allocator snapshot 没有像 CUDA 文档合同那样在 `device_traces` 中留下两个 `action=snapshot` 边界，worker 因审计断言报 `allocator history did not contain two snapshot boundaries`；不是产品图或数值失败。按失败规则读取本轮 `output_code.py`：baseline 明确先分配 `buf0` 执行 `extern_kernels.mm(..., out=buf0)`，再由 `aten.addmm` 返回 `buf1`，解释其 additional peak 恰为两个 output allocation；四个 `copy_if_misaligned` 只是 16-byte runtime fast check，实际输入已验证 aligned。修正诊断脚本：history 在 warmup 后才启用，本来就没有早期 alloc/free 事件，因此不再依赖 snapshot marker，直接过滤启用后到最终 `_snapshot()` 之间的全部非-snapshot trace；保留 action/size/address/frame。使用全新输出目录重跑 baseline，再跑 candidate。
- E-085（T-023 memory-history correctness 污染修正）：修正边界后的 baseline trace 完成，但审计 worker 在读取 max/snapshot 前调用 `_correctness()`，`torch.testing.assert_close` 及误差统计额外申请 `61,440 B`、多个 `245,760 B` 和若干小块，使本轮 additional peak 放大为 `862,208 B`，不能与 E-081 的 steady peak比较。trace 开头属于 compiled graph 的分配只有两个请求大小均为 `122,880 B` 的块，frame 分别落在 generated wrapper 的 mm output 和 `aten.addmm` output，与 E-084 生成图一致。修正顺序：同步 graph call 后立即读取 max、生成 snapshot并关闭 history，再执行正确性和删除 output；这样 trace/peak 只覆盖 graph call，正确性仍保留但不污染诊断。首轮 `memory_trace_shape_a_baseline_rerun1` 只作脚本诊断，使用新目录重跑 baseline/candidate。
- E-086（T-023 candidate workspace 根因闭环）：无 correctness 污染的 baseline trace 复现 additional allocated peak `246,784 B`，history 只有两个 requested-size `122,880 B` 的 graph allocation，分别是 mm 中间 `buf0` 和 addmm 输出；candidate trace 复现 peak `517,120 B`，history 也只有两个 allocation：generated wrapper 的输出 `122,880 B`，以及 `triton/backends/ascend/driver.py:140` launcher 内的 `393,216 B`。candidate 编译 metadata 明确 `workspace_size=65,536 B`、grid 为 `6×1×1`，launcher 按 `workspace_size * blockNum` 分配，恰为 `65,536×6=393,216 B`；因此不是 `copy_if_misaligned`、Inductor planner或泄漏，而是 Triton Ascend device workspace。两 trace 正确性 max/mean absolute error均为 0。E-081/E-082 的 candidate allocated peak由一个 output 加该 workspace（再含 allocator 对齐）完整解释。下一步只读检查当前 Triton Ascend workspace 生成和编译选项；任何减 workspace 方案必须重新通过单-task、正确性和 paired 性能，不能直接关闭内存 gate。

### T-024：mm_plus_mm different-K workspace/tile 审计

- 状态：`completed-no-both-gates-candidate`；control 与编译选项/大 tile/grouped-program 共七个配置完成筛选，没有配置同时通过预登记的显存和 task-duration gate；未修改 torch_npu 产品源码，未重建或重装 wheel。
- 目标：在保持 T-023 单 Triton task、数值正确和端到端性能收益的前提下，把 Ascend launcher 的总 workspace 降到严格显存门槛内；若无法同时满足，则保留 default-off 并把性能收益与显存代价写成明确取舍，不静默放宽门槛。
- 源码结论：安装版 Triton Ascend 由编译后二进制 callback 回填每 block `workspace_size`，driver 在 launch 前按 `workspace_size * gridX * gridY * gridZ` 分配。`set_workspace_multibuffer` 的文档语义是为 workspace 数据搬运增加 2/4 档多缓冲，不能作为减内存开关；当前 auto-blockify 仅在环境开关启用且逻辑 grid 需要折叠到物理核数时减少实际 block，本例 shape-A 的 grid=6，小于物理核数，不能解决问题。
- 首轮候选：以当前 `128x128x128, multibuffer=True` 为 control；审计 `256x128x128`、`128x256x128`、`256x256x128` 三种减 grid tile，以及 `128x128x128, multibuffer=False` 对照。每个配置必须 fresh compile，记录 metadata 中 per-block workspace、grid 和总 workspace；编译资源不足属于该配置不支持，不改编译器或共享环境绕过。
- 分阶段闸门：先在 shape-A/fp16/contiguous 做 compile、正确性、单-task profiler与 isolated memory；只有 total candidate additional allocated 不高于 `baseline additional + one pure output`，且 task duration 未相对当前 T-023 profile 明显回退的配置，才进入 unaligned 复核和 fresh-process integrated paired benchmark。不得用 direct launcher 的 host 数值代替集成性能结论。
- 正确性与性能合同：沿用 fp16 `rtol=atol=0.01`，profile warmup 1/active 10并要求 10/10 每 step 唯一同名 task；最终 paired 仍为 baseline/candidate fresh process、warmup 10/runs 100/3 轮，主判据为三轮 p50 中位数改善超过 10%，同时报告 mean±stdev、p99、正确性、allocated/reserved peak和 task duration。
- 隔离与产物：所有 NPU 运行从 `/home/z50063656/tmp` 启动，运行前后检查目标物理卡进程表；不清理其他任务。产物使用 fresh `results/t024_mmplus_different_k_workspace_20260821/` 子目录。失败 compiled case 才检查本轮 debug/output code；成功 case 不读取。产品 tile 只有在完整闸门通过后才另行登记修改提案。
- E-087（T-024 实验登记与选项排除）：已完成 Triton Ascend compiler/driver/官方迁移文档只读核对并登记上述合同。`enable_ubuf_saving` 用于节省片上 UB，可能把数据转移到 workspace，暂不作为首轮减 device workspace 候选；`disable_tightly_coupled_buffer_reuse=True` 从语义上会禁用复用，也不进入首轮。下一步新增 audit-only fresh worker，对 tile/grid 和 `multibuffer=False` 做最小筛选。
- E-088（T-024 首轮 tile/编译选项筛选）：新增 `t024_mmplus_different_k_workspace_screen.py` 并通过 `py_compile`、帮助入口和 100 列检查。shape-A control 完整复现 per-block workspace `65,536 B`、grid 6、total workspace `393,216 B`、additional allocated peak `517,120 B`，正确性误差 0，10/10 单 task p50 `9.02 μs`，证明筛选链与 T-023 一致。`128³, multibuffer=False` 的 workspace/peak 完全不变，task p50 `8.97 μs`；关闭该编译优化不是减 workspace 手段。`256x128x128` 与 `128x256x128` 均因 L0C 需求 `2,097,152 bit > 1,048,576 bit` 编译失败，`256x256x128` 需求 `4,194,304 bit`，因此不能直接扩大 accumulator tile 来减 grid。
- E-089（T-024 grouped-program 二阶段登记）：下一步仍只修改 T-024 审计脚本，新增 `128x128x128, tiles_per_program=2` 的 grouped kernel：每个物理 program 在内层顺序处理两个独立 logical output tile，K1/K2 loop、mask、fp32 accumulator与 store 语义不变，shape-A/unaligned 的 logical grid 6 映射为 launch grid 3。预期 per-block workspace 保持 `65,536 B`、total workspace降到 `196,608 B`，candidate peak约为 output 加 workspace，从而进入 `370,176 B` 门槛；但并行度减半可能增加 task duration。先只在 shape-A fresh compile 做正确性、metadata、memory和10-step single-task profile；编译资源溢出、输出覆盖不全、峰值不降或 task p50 超过筛选阈值即停止，不接入产品。若筛选通过，再登记 unaligned 与集成 paired 验证。
- E-090（T-024 grouped-program 首轮结果与 locality 复筛）：linear group2 正确编译，logical grid 6 降为 launch grid 3，per-block workspace仍为 `65,536 B`，total workspace `196,608 B`；实测 candidate additional allocated peak `320,512 B`，低于 `370,176 B` 严格门槛，正确性 max/mean absolute error均为 0，10/10 每 step 单一 grouped task。其 task p50/p99 为 `10.42/11.18 μs`，比同批 control p50 `9.02 μs` 高约 15.5%，且超过预设 `10 μs` 筛选线，故 linear 映射不进入集成。允许最后一个同等显存、同等计算量的 audit-only locality 复筛：physical pid 固定 N tile，每个 program 顺序处理两个 M tile，使 B/D tile 在两次计算间具备复用机会并避免 linear 映射中 logical tile 2→3 的跨 M/N 跳转。若 M-group2 仍高于 `10 μs`，停止 T-024 kernel 结构搜索，不再通过放宽阈值进入集成。
- E-091（T-024 关闭与 T-023 最终取舍）：M-group2 同样正确、10/10 单 task，workspace/peak 保持 `196,608/320,512 B`，但 task p50/p99 为 `10.54/11.12 μs`，仍高于 `10 μs` 门槛，固定 N 的局部性没有补回并行度损失。按 E-090 停止 kernel 结构搜索，不运行 unaligned 或集成 paired，不修改产品 tile。T-023 结论固定为 default-off 条件性性能替代：shape-A/unaligned fresh-process paired p50 改善 `15.29%/18.04%`、p99 改善 `17.96%/19.18%`，正确性误差 0且每 step 单一融合 task；candidate additional allocated peak比 baseline均多 `270,336 B`，根因为 `65,536 B × 6` Ascend Triton workspace。严格 memory gate未通过，不能升级为默认开启；保持 output 元素上限 `131072`、large/dynamic/empty/arbitrary-stride/same-K 排除和 extern fallback。正式交付状态仍带无 shim headers/launcher 环境复验限制。

### T-025：pad_mm/pad_bmm/pad_addmm NPU capability 与收益审计

- 状态：`completed-capability-available-performance-rejected`；mm/bmm/addmm 的positive替换与aligned不触发均已闭环，三个positive shape的三轮配对性能均明显回退；未修改PyTorch/torch_npu/Triton产品源码，未重建或重装wheel。
- 源码根因：上游 joint-graph `pad_mm.py:can_pad()` 在性能决策前调用 `check_device()`，只接受 CUDA/XPU；`force_shape_pad=True` 只让 `_should_pad()` 跳过 benchmark/heuristic，不能越过 device gate。fp16/bf16 对齐粒度为 8、fp32 为 4；replacement 为 `constant_pad_nd(mat1/mat2) -> mm/bmm/addmm -> slice original M/N`。torch_npu triton experimental 还关闭 `shape_padding`，首轮只审计 default backend。
- 方法：每个 family/variant 使用独立 fresh process。baseline 保持原 `check_device`，candidate 仅在 worker 内把该函数改为“两个输入都是 NPU”；两路都设置 `shape_padding=True, force_shape_pad=True`，从而只隔离 device capability差异。再包装模块全局 `should_pad()` 记录 family、输入 shape/stride、返回值，并安装 `joint_custom_post_pass` 只读记录 pattern 后的 call_function target和 meta shape。
- 首轮 smoke：复用既有 fp16 contiguous mm正例 `(255,257)@(257,259)`。baseline必须记录 targeted `should_pad=False`且 joint graph只有原 mm；candidate必须记录 targeted `should_pad=True`，joint graph同时出现 padding、mm与slice，compiled输出通过 eager `rtol=atol=0.01`。aligned `(256,256)@(256,256)` 后续作为 negative，candidate也不得引入padding。
- 扩展闸门：只有 mm正例真实替换且正确，才扩到 bmm `(4,127,129)@(4,129,131)` 和 vector-bias addmm，以及 aligned negatives；只有三个 family功能图均闭环，才登记 fresh-process paired benchmark。成功 case不读取 `output_code.py`；失败或图不符时才读本轮 debug artifact定位。
- 性能合同：后续 baseline/candidate各 warmup 10、runs 100、3轮交错，报告 mean±stdev/p50/p99、首次编译、allocated/reserved peak、profiler task组成和设备隔离。p50改善超过10%且p99/显存可接受才提议放宽 gate；无收益则从 `unsupported` 收敛为 NPU `not-applicable/supported-regression`，不手写 Triton复制padding。
- 隔离：所有测试从 `/home/z50063656/tmp` 启动，运行前后确认目标物理卡无外部进程；出现占用则结果保留但不用于性能 verdict，不终止他人任务。结果目录使用 fresh `results/t025_pad_family_20260821/`。
- E-092（T-025 只读设计登记）：已复核 runtime PyTorch `pad_mm.py`、joint graph lazy registration、旧 P0 force-shape-pad结果和三 family正例 shape。旧结果中 active config确为 `force_shape_pad=True`，但 mm/addmm pattern count为0，bmm的一个通用 matcher count无法归因到 padding；这些证据不能替代 capability bypass。下一步实现 audit-only graph/event worker并先跑 mm baseline/forced。
- E-093（T-025 mm capability bypass 闭环）：新增 `t025_pad_family_capability.py`，`py_compile`、帮助入口和100列检查通过。物理NPU 1上的两个fresh process均无需launcher shim：baseline positive的targeted `should_pad=False`，joint graph只有`aten.mm.default`；forced worker只把进程内`check_device`从原始false改为两个输入均为NPU时true，targeted `should_pad=True`，joint graph精确出现2个`aten.constant_pad_nd.default`、padded `(256,264)@(264,264)->(256,264)` mm和2个slice，最终恢复`(255,259)`。两路输出max/mean absolute error均为0，forced matcher计数为1。成功case未读取`output_code.py`。mm replacement的NPU功能承接通过，按合同扩到bmm/addmm正例和三个forced aligned negative，仍不采性能。
- E-094（T-025 bmm通过与addmm launcher环境分流）：同一物理NPU 1的fresh bmm baseline/forced均功能通过；baseline targeted `should_pad=False`且joint graph仅原bmm，forced为2个constant-pad、padded bmm和2个slice，输出max/mean error为0。addmm baseline也通过且不pad；forced在图替换后、执行前因Triton pointwise launcher找不到editable PyTorch source view的`ATen/ATen.h`失败，属于E-071同一环境合同。按失败规则读取本轮`output_code.py`：它已生成bias补5个零、mat1/mat2 pad到`(256,264)/(264,264)`、padded `aten.addmm`和padded-stride输出view；阻断点是生成5个零的`triton_poi_fused_addmm_0` host launcher，不是addmm/pad语义。修正audit worker在安装hook后立即把events挂入result，即使后续编译失败也保留joint graph证据；允许用T-022登记的audit-only C++20/header shim在fresh目录复跑addmm forced，只判开发态capability，不判正式无shim环境通过。
- E-095（T-025 addmm开发态capability闭环）：物理NPU 1空闲时，在fresh cache中仅加入T-022已登记的audit-only C++20/header shim与`TRITON_DISABLE_PRECOMPILE=1`复跑addmm forced。targeted `should_pad=True`，joint graph含2个constant-pad、bias的`full+cat`、padded `(256,264)@(264,264)->(256,264)` addmm和2个slice；matcher计数为1，最终输出shape `(255,259)`，max/mean absolute error均为0，graph gate通过。该成功case未读取`output_code.py`。结论严格限定为开发环境capability通过；正式无shim状态仍由E-094的launcher header失败约束。下一步运行mm/bmm/addmm三个forced aligned negative，要求targeted `should_pad=False`且不得出现padding/slice，然后才允许登记性能筛选。
- E-096（T-025 aligned负例闭环）：物理NPU 1空闲时顺序启动三个fresh forced worker，mm `(256,256)@(256,256)`、bmm `(4,128,128)@(4,128,128)`、addmm vector-bias `(256,)` 均记录targeted `should_pad=False`；三个joint graph分别只含一个原始mm/bmm/addmm，constant-pad和slice计数均为0，graph gate通过，输出max/mean absolute error均为0。三个成功case均未读取`output_code.py`。功能结论为：绕过上游CUDA/XPU device gate后，NPU default backend可承接三类static fp16 contiguous正例，且对已对齐shape保持不触发；这只证明capability，不证明值得放宽产品gate。
- E-097（T-025性能阶段登记）：下一步新增audit-only `t026_pad_family_performance.py` 与聚合器，不改产品源码。每个family/variant/round都使用独立fresh process/cache、固定seed和同一positive shape；baseline保留原device gate，candidate只做E-092的NPU capability bypass，两路都使用`shape_padding=True, force_shape_pad=True`。为保持编译环境对称，所有worker统一带T-022 audit shim；结果只能形成开发态性能结论，正式无shim限制不变。三轮执行顺序固定为`B-C / C-B / B-C`，每轮warmup 10、runs 100并逐次同步，报告mean±stdev/p50/p99、首次编译和正确性；每个family首轮两侧另做warmup 1/active 10的NPU task profile，三轮均测allocated/reserved peak。主gate为三轮p50中位数改善严格超过10%，p99不得回退超过5%；显存报告candidate相对baseline的峰值增量及理论padding字节，不以隐藏中间buffer换取通过。任一family不满足性能gate即定为`capability-available-performance-rejected`，不手写Triton复制padding；通过者才进入更多shape/dtype与产品capability设计。
- E-098（T-026 mm性能否决）：新增worker与聚合器通过`py_compile`、帮助入口、100列检查，随后在物理NPU 1按`B1-F1/F2-B2/B3-F3`完成六个fresh process。三轮p50中位数从baseline `0.213605 ms`回退到forced `0.368780 ms`（改善率`-72.65%`），p99从`0.232840 ms`回退到`0.407260 ms`（`-74.91%`）；allocated peak中位数从`132,608 B`增至`411,136 B`，差值`278,528 B`，与padding buffer理论上界`277,638 B`仅有allocator对齐差。10-step profile中baseline每步只有1个`aclnnMm`，forced每步为2个PadV3 memset、2个PadV3和1个mm，matmul本身p50由`6.78 μs`略降至`6.48 μs`，但padding task成本远大于这点收益。所有六轮图gate和正确性均通过，成功case未读取`output_code.py`。该shape的mm结论为`capability-available-performance-rejected`，不放宽device gate、不手写Triton复制padding；继续用同一合同测bmm/addmm，避免把单family结果外推。
- E-099（T-026 bmm性能否决）：物理NPU 1在family开始前为空闲，按同一六fresh-process合同完成bmm。三轮p50中位数由baseline `0.215645 ms`回退到forced `0.356485 ms`（`-65.31%`），p99由`0.232380 ms`回退到`0.394350 ms`（`-69.70%`）；allocated peak从`133,632 B`增至`428,032 B`，差值`294,400 B`，与理论padding上界`293,400 B`基本一致。profile同样由每步1个`aclnnBatchMatMul`变为4个padding task加1个bmm，且padded bmm本身p50也由`7.16 μs`回退到`7.53 μs`。六轮图gate和正确性均通过，成功case未读取`output_code.py`。该shape的bmm同样定为`capability-available-performance-rejected`；剩余只测addmm，仍不提前外推。
- E-100（T-025/T-026关闭）：物理NPU 1在family开始前为空闲，addmm六个fresh process全部通过图gate和正确性。三轮p50中位数由baseline `0.233980 ms`回退到forced `0.516225 ms`（`-120.63%`），p99由`0.249340 ms`回退到`0.535490 ms`（`-114.76%`）；allocated peak从`132,608 B`增至`412,160 B`，差值`279,552 B`。profile由每步1个addmm变为2个PadV3 memset、2个PadV3、1个bias-zero Triton、1个cat和1个addmm，共7个task；padded addmm本身p50从`7.58 μs`略降至`7.16 μs`，远不足以覆盖额外任务。forced首次编译中位数`17,594.01 ms`，显著高于baseline `2,612.72 ms`，由bias补零Triton编译支线放大。所有性能worker统一使用audit shim，成功case未读取`output_code.py`。T-025/T-026最终结论：上游device gate仍使当前产品状态为`unsupported`；测试侧绕过后，三family在static fp16 contiguous上功能可承接但性能分别回退72.65%/65.31%/120.63%，故replacement状态为`rejected-performance-regression`。不放宽gate、不修改产品、不用Triton复制padding；若未来大shape数据证明对齐GEMM收益，候选应是把masked load融合进专用GEMM而非独立padding kernel，并重新走功能/性能/显存gate。

### T-027：P1 B2 首批 dynamic-shape FX 结构验收

- 状态：`structure-complete-npu-compile-pending`；现有device-independent UT 32/32通过，7条pass只推进到结构触发确认；未修改产品源码、测试源码、wheel或环境。
- 范围：`test/_inductor/test_dynamic_shape_fx_passes.py` 直接覆盖的7个custom pass：`fold_expand`、`view_fold_pass`、`fold_reduce`、`fold_slice`、`repeat_to_expand_pass`、`fold_four_op_pass`、`cat_to_view_pass`，以及这些pass共用的symbolic-shape utility。
- 方法：从`/home/z50063656/tmp`激活Benchmark环境，以绝对测试文件路径运行完整unittest；运行时必须导入site-packages中的source-built torch_npu，不通过`PYTHONPATH`覆盖到源码package。记录positive transform、undecidable/partial negative、switch-off和static regression的测试数量与失败。
- 判定边界：该文件只构造/变换FX图，不执行NPU kernel。通过最多把7条矩阵记录推进到`structure-trigger-confirmed`，不能填写NPU codegen、fallback、性能或最终supported verdict；下一阶段仍需default backend fresh NPU compile。
- E-101（T-027登记）：已只读核对文件包含17个symbolic utility测试、11个dynamic pass正负例和4个switch/static regression，共32个test方法；其中直接归因7个pass的结构case为15个。下一步先做`py_compile`，再运行完整文件；失败时分类为测试/源码接口/环境，不在本任务中直接修改产品。
- E-102（T-027沙箱环境分流）：测试文件`py_compile`通过，但受限沙箱内导入site-packages torch_npu时，`torch_npu/_inductor/config.py`查询SoC触发`aclInit`，因设备访问受限报507008；失败发生在unittest收集前，0个test执行，不能归因任何pass。保持命令、Benchmark环境和`/home/z50063656/tmp`工作目录不变，只在允许NPU设备访问的执行权限下复跑；不加入package overlay、mock或源码修改。
- E-103（T-027结构层闭环）：同一Benchmark环境、site-packages source-built torch_npu和绝对测试路径在允许驱动访问的执行权限下完成32/32，耗时`0.926 s`。其中7个pass的15个直接结构case全部通过：`fold_expand` 2项、`view_fold_pass` 1项、`fold_reduce` 2项、`fold_slice` 4项、`repeat_to_expand_pass` 3项、`fold_four_op_pass` 1项、`cat_to_view_pass` 2项；另17项symbolic utility通过。结果覆盖symbolic positive、不可证明/partial negative、switch-off和static regression，没有执行NPU kernel。矩阵只更新applicability/trigger/correctness为结构层通过，codegen/fallback/performance/verdict保持`not-run`。下一步另行登记default backend fresh NPU compile worker，不能复用本结论冒充端到端支持。

### T-028：P1 B2 首批 custom pass NPU compile

- 状态：`fold-reduce-functional-complete-expansion-pending`；fold_reduce positive/negative已完成default backend NPU触发/图/正确性，另外6个pass待扩展；只新增audit worker、结果和文档，未修改产品源码、测试源码或wheel。
- 首轮：先用`fold_reduce` positive `sum(x, dim=1, keepdim=True)`、输入`(6,1,4)`验证harness。worker在default backend已注册POST pass列表中按函数identity临时包装目标函数，记录调用前后FX call target计数；不改变原函数逻辑、不关闭其他产品pass。
- Graph gate：目标wrapper必须恰实执行，进入前存在一个`aten.sum.dim_IntList`，退出后该sum为0；compiled输出与eager在fp16容差内一致。随后才扩到另外6个positive和每pass至少一个negative。
- 隔离：每个case独立fresh process/cache，`TORCH_COMPILE_DEBUG=1`，从`/home/z50063656/tmp`运行。当前fresh Triton launcher统一用T-022 audit shim；成功case不读取`output_code.py`，失败case才读本轮artifact。结果只能形成开发态default backend证据。
- 性能边界：T-028只关闭触发、图、正确性与runtime codegen；不采paired性能。功能闭环后另行登记单pass enabled/disabled三轮paired，不能用eager作为性能baseline。
- E-104（T-028设计登记）：已确认7个目标都注册在`PassType.POST`，default backend通过`AscendCustomPostPass`按level执行registry。审计wrapper只替换registry中目标函数所在slot并转调original，可同时捕获该pass真正看到的前后图；若目标节点在更早阶段已被其他pass删除，将如实判为`not-reached`，不通过直接调用函数伪造端到端触发。
- E-105（T-028首轮harness推理模式修正）：fold_reduce首轮compiled输出与eager误差0，但目标wrapper调用数为0，graph gate正确失败。按失败规则读取本轮`output_code.py`，最终图仍保留`aten.sum.dim_IntList`并生成`triton_poi_fused_sum_0`，证明pass确实未执行而非observer漏记。根因是POST driver的`is_inference_check()`直接返回`not torch.is_grad_enabled()`；worker虽无requires-grad输入，却在默认grad-enabled上下文编译，inference-only pass被合法跳过。仅修改audit worker，把reference、compile、首次执行和正确性置于`torch.no_grad()`，不改产品mode判断；用fresh结果目录复跑。原失败保留为harness证据，不覆盖。
- E-106（T-028 fold_reduce positive闭环）：audit worker通过`py_compile`、帮助入口和100列检查。在物理NPU 1的fresh cache、default backend、no-grad推理和统一audit shim下，registry wrapper恰调用1次；入口图含1个`aten.sum.dim_IntList`，原`fold_reduce`执行后为0，graph gate通过。输入`(6,1,4)` fp16的compiled输出shape不变，eager对比max/mean absolute error均为0，首次compile+run `2812.37 ms`。成功case未读取`output_code.py`。矩阵中fold_reduce推进到`npu-trigger-confirmed`、runtime codegen正确，performance与最终verdict仍`not-run`；下一步先加symbolic-dim negative，要求pass被调用但sum保持，再扩另外6个pass。
- E-107（T-028 fold_reduce negative登记）：只扩展audit worker的`--shape-profile negative`，输入`(6,4)`并对symbolic dim0做keepdim sum。目标wrapper仍必须恰调用1次，入口/出口`aten.sum.dim_IntList`计数都为1；compiled与eager正确。positive默认与既有命令兼容，产品源码不改；fresh目录运行，失败才读本轮`output_code.py`。
- E-108（T-028 fold_reduce功能闭环）：negative fresh worker在物理NPU 1、default backend、no-grad推理和audit shim下完成。目标wrapper恰调用1次，`aten.sum.dim_IntList`入口/出口计数均为1，证明无法静态判定size=1的symbolic reduce没有被误删；compiled输出shape `(1,4)`，max/mean absolute error均为0。首次compile+run `21,361.85 ms`受fresh Triton编译环境影响，只记录不作性能结论。成功case未读取`output_code.py`。fold_reduce现具结构UT与NPU positive/negative功能证据，performance/final verdict仍`not-run`；T-028下一步扩`fold_expand`等6个pass。
- E-109（T-028 六个 pass 扩展登记，2026-08-24）：继续扩展 audit-only
  `t028_b2_custom_pass_compile.py`，覆盖 `fold_expand`、`view_fold_pass`、
  `fold_slice`、`repeat_to_expand_pass`、`fold_four_op_pass` 和
  `cat_to_view_pass`，不修改 PyTorch、torch_npu、Triton 产品源码或当前 wheel。
  每个 pass 建立一个 positive 和一个 negative：目标 registry wrapper 必须恰调用
  1 次；positive 要求目标节点按 pass 语义消除或替换，negative 要求节点保持；
  `repeat_to_expand_pass` positive 还必须确认 repeat 消失且 expand 出现。所有 case
  使用 default backend、`torch.no_grad()`、dynamic compile、独立 fresh process/cache，
  从 `/home/z50063656/tmp` 发起并与 eager 做 fp16 数值比较。首次 compile+run 只记录，
  不作性能结论；功能闭环后另行登记单 pass on/off 三轮 paired benchmark。失败时只读
  本轮 debug artifact 分类为未触发、前序 pass 改写、图门禁、lowering/codegen、精度或
  环境，不用 monkeypatch 产品逻辑伪造通过。
- E-110（T-028 首批扩展与 harness 分类修正，2026-08-24）：受限权限下的首个
  `fold_expand` positive 在设备枚举阶段报 `aclInit 507008`，0 个 pass 执行，保留为
  environment-blocked；相同命令在允许驱动访问的物理 NPU 1 重跑后通过。随后 12 个
  positive/negative 中，`fold_expand`、`repeat_to_expand_pass`、`cat_to_view_pass`
  两侧均直接通过，`fold_slice` negative 与 `fold_four_op_pass` negative 通过；所有已执行
  case 的 eager/compiled max/mean absolute error 均为 0。四个原始 error 均为审计门禁
  或 reachability 分类，不是数值/codegen 失败：`view_fold_pass` positive 和
  `fold_slice` positive 的目标节点在进入目标 wrapper 前已不存在；`view_fold_pass`
  negative 在 wrapper 前已被规范成 `aten.reshape.default`，而旧门禁只统计
  `aten.view.default`；`fold_four_op_pass` positive 的 `x + zeros_like(x)` 在进入 Inductor
  后端前已化为恒等图，registry wrapper 调用数为 0。按失败规则已只读对应四个 debug
  目录：前两者的 `fx_graph_readable.py` 已只剩 relu，view negative 的 readable graph
  含 view、wrapper event 含 reshape、最终 transformed graph 保留 reshape；four-op positive
  没有生成 Inductor graph artifact。下一步只修改 audit worker：view 门禁聚合
  view/reshape/_unsafe_view；数值正确但目标在 wrapper 前消失或整个 backend 未进入时明确
  记为 `npu-compile-target-not-reached`，不伪装为目标 pass 成功，也不把它继续报成产品
  error。使用新目录只复跑这四项，产品源码和 wheel 仍不改。
- E-111（T-028 输出 alias 语义补充合同，2026-08-24）：首批数值/图结果不能直接关闭
  功能结论。`fold_reduce(keepdim=True)` 在源码中可直接用输入替换 sum，而 eager sum
  返回新 storage；`cat_to_view_pass` 可直接用 parent 替换 eager cat，新旧 storage alias
  也可能不同。两者都存在“数值相同但输出 mutation/alias 语义变化”的风险。只扩展
  audit worker，在 eager reference 与 compiled output 上记录 dtype、stride，以及 output
  是否与原输入共享 storage；要求 compiled 三项与 reference 完全一致，否则状态为功能
  error。用 fresh 目录复跑 `fold_reduce` 和 `cat_to_view_pass` 的 positive/negative；
  alias 失败时按规则读取本轮 `output_code.py` 确认 wrapper 是否直接返回输入，并保持
  performance/verdict 为 `not-run`。不通过修改测试输入、插入 clone 或放宽断言掩盖问题。
- E-112（T-028 alias blocker 闭环，2026-08-24）：`fold_reduce` positive 与
  `cat_to_view_pass` positive 均复现真实功能失败。两者 eager/compiled 的数值误差为 0，
  dtype/stride一致，但 eager output 都不与输入共享 storage，compiled output 都与输入共享；
  两份 `output_code.py` 均明确 `return (arg2_1,)`。对应 negative 保持原节点，数值、dtype、
  stride、alias 全部匹配。结论不是性能中性，而是触发路径 correctness blocker；两条矩阵
  记录在修复前标为 `unsupported-correctness-alias`，禁止进入 paired performance。
  `fold_expand` 与 `repeat_to_expand_pass` 的输出 consumer 分别是 relu/mul，现有语义合同
  通过；`view_fold_pass`/`fold_slice`/`fold_four_op_pass` 当前 positive 未到达目标 pass，
  只记 reachability-neutral；不得把这些中性结果写成目标 pass 已优化成功。

### T-029：B2 alias correctness 修复提案

- 状态：`completed-intermediate`。目标只修复 T-028 已实测的
  `fold_reduce` 与 `cat_to_view_pass` 输出 storage 语义，不扩大其他 pass 范围，不使用
  Triton，也不调整 Benchmark 环境版本。
- `fold_reduce`：当前 `_get_fold_result()` 对 `keepdim=True` 直接返回输入，对
  `keepdim=False` 返回输入的 squeeze view；两者都把 eager reduction 的新 storage 变成
  alias。候选是在删除 size-one reduction 时先建立 contiguous clone，再按 keepdim 决定
  直接返回 clone 或 squeeze clone；同时补 dtype 参数边界，不能把带 dtype conversion 的
  sum 无条件替换成同 dtype clone。
- `cat_to_view_pass`：full-cover identity 分支当前直接返回 parent，而 eager cat 必须产生
  新 storage。候选把这两个 identity 分支改为 contiguous clone(parent)；rotation 分支已用
  roll 产生新 storage，本轮不改。这样仍可删除 slice/cat 结构，但是否比原 cat 更快必须
  后续 paired benchmark 决定。
- 拟修文件：
  `torch_npu/_inductor/fx_passes/utils/get_binary_fold_result.py`、
  `torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py` 和
  `test/_inductor/test_dynamic_shape_fx_passes.py`。先补结构断言，确保两个 positive 出现
  clone、negative 不误改；再用源码覆盖的纯 FX UT 验证，最后重建 torch_npu wheel 并
  `--no-deps` 安装，从 `/home/z50063656/tmp` 重跑四个 NPU alias case。
- 验收：数值、dtype、stride、output-input alias 全部与 eager 相同；目标 pass 确实调用，
  原 sum/cat 被删除且 clone 出现。功能通过后才登记单 pass on/off 三轮 paired；若 clone
  使性能无收益，则结论为 correctness-fixed-supported-neutral，不为了性能再次破坏 alias。
- 回滚：只撤销上述三文件的 T-029 增量；不撤销 T-011、T-023、动态 shape 既有实现，
  不清理未知生成文件或他人工作树修改。

### E-113：T-029 安装态验证门禁（2026-08-24）

- `fold_reduce` 正例除要求 `aten.sum.dim_IntList` 从 1 降为 0 外，还要求
  pass 后至少保留 1 个 `aten.clone.default`；负例继续要求 reduce 不被折叠。
- `cat_to_view_pass` 全覆盖正例除要求 `aten.cat.default` 从 1 降为 0 外，
  还要求 pass 后至少保留 1 个 `aten.clone.default`；非全覆盖负例继续保留 cat。
- 四个正负例都必须同时通过数值、dtype、stride 和输出/输入 storage alias
  一致性门禁；只满足图结构或数值一致仍判失败。
- 验证对象必须是新构建并以 `--no-deps --force-reinstall` 安装的 wheel，
  wheel 内两份修复文件的哈希需与源码一致，避免源码覆盖测试造成假阳性。

### T-030：B2 alias 修复后的单 pass 性能合同

- 状态：`completed`。只新增 audit worker/aggregate 和结果文档，
  不再修改产品源码或环境版本。
- 对象：`fold_reduce` 正例 `(1024, 1, 1024)` 与 `cat_to_view_pass` 全覆盖正例
  `(2048, 1024)`，fp16、contiguous、default backend、`torch.no_grad()`、dynamic compile。
- baseline 在 registry 原位置用观测 wrapper 跳过目标 pass，candidate 调用原 pass；
  两侧都要求 wrapper 恰好调用一次。baseline 必须保留 sum/cat 且不新增 clone，
  candidate 必须删除 sum/cat 并新增 clone。
- 每个 family 使用 3 轮 fresh process paired，顺序 `B1,C1,C2,B2,B3,C3`，
  每个 worker warmup 10、runs 100；记录 mean/stdev/p50/p99、输出分配扣除后的
  peak allocated/reserved、首次 compile+run。每侧第 1 轮额外采集 10 active steps 的
  NPU task profile。
- 两侧都必须在计时前后通过数值、dtype、stride、output-input alias 门禁。p50
  改善超过 10%、p99 不回退超过 5%、显存无不可接受增长才标 beneficial；否则按数据
  标为 supported-neutral 或 performance-regressed，不以一次 compile 时间作性能结论。
- 测试从 `/home/z50063656/tmp` 发起，使用物理 NPU 1；开始、结束均检查进程表。
  审计 launcher shim 继续只作为已登记环境兼容手段，不提升无 shim 环境结论。

### T-031：fold_reduce 性能回退后的产品收敛提案

- 状态：`completed-verified`。T-030 已证明 alias-safe clone 相对保留
  size-one sum 的 p50/p99 分别回退 3.06%/6.72%，且没有 task 或显存收益。
- 产品策略：让 `fold_reduce` 保留原 reduction，不再执行 size-one reduce→clone/squeeze
  替换；保留 `_get_fold_result` 的 alias-safe 实现，避免未来复用时重新引入直返输入错误。
  `cat_to_view_pass` 的 clone 修复不回滚，因为它 task 3→1 且显存减少约 4 MiB。
- 测试调整：fold_reduce 静态 size-one 正例改为断言 sum 保留、clone 不出现；dtype 和
  symbolic negative 同样保留。安装态 NPU 正例门禁改为 sum 1→1、clone 0，alias 仍须
  与 eager 一致。cat 正负例门禁保持 T-029 合同。
- 完成源码静态/33 个 FX 测试后重建并 `--no-deps --force-reinstall` wheel；最终 wheel
  需重跑 fold_reduce/cat 四个 NPU case。若均通过，fold_reduce 记为
  `supported-pass-disabled-performance-rejected`，cat 记为
  `supported-neutral-resource-beneficial`。
- 回滚只涉及 fold_reduce 的 no-op 收敛与对应测试/audit 门禁；不改环境版本，不清理
  生成文件，不触碰 T-011/T-023 和 cat 的正确性修复。

### E-114：T-029/T-030 执行结果（2026-08-24）

- T-029 wheel SHA256 为
  `51f484457e555544c171167ff5a652478995f4acc3d749f6accf0a091a00e4df`，安装态
  33/33 FX UT 和 fold_reduce/cat 四个 alias case 通过；首次受限安装态 UT 的
  `aclInit 507008` 在驱动可见层重跑通过，保留为环境中性证据。
- T-030 两个 family 均完成 3 轮 fresh-process、warmup 10/runs 100 与一轮 10-step
  profiler。fold_reduce clone p50/p99 回退 3.06%/6.72%，task 1→1、显存不变；
  cat clone p50 +2.29%、p99 -0.78%，task 3→1、allocated peak 减少 4,195,840 B。
- fold_reduce clone 是“功能成功、性能失败”的中间方案；cat clone 是“延迟中性、资源
  有益”的保留方案。两者不得合并写成统一的 pass 优化成功。

### E-115：T-031 最终 wheel 验证（2026-08-24）

- 源码静态检查、lintrunner 与源码覆盖 FX UT 33/33 通过；最终 wheel SHA256 为
  `29c3c105453a36d8f2eb648eeb0a2d35cfd0cb871c34697c6aaf17fb1a96a6f5`，wheel/source
  文件哈希一致并以 `--no-deps --force-reinstall` 安装。
- 最终安装态 FX UT 33/33 通过；fold_reduce/cat 正负四个 NPU case 均数值误差 0，
  dtype、stride、storage alias 与 eager 一致，图门禁通过。
- fold_reduce 最终 sum `1→1`、clone `0`，verdict 为
  `supported-pass-disabled-performance-rejected`；cat positive 为 cat `1→0`、
  clone `0→1`，negative cat `1→1`，verdict 为
  `supported-neutral-resource-beneficial`。
- T-029 性能失败 candidate wheel 已备份为
  `artifacts/torch_npu_t029_alias_safe_clone_candidate.whl`；最终环境安装的是 T-031 wheel。

### T-032：B2 第二批冗余规约 pass 结构与 NPU 可达性审计

- 状态：`completed`。本轮覆盖
  `fold_cast`、`fold_cat`、`fold_clone`、`fold_detach` 四条尚未形成动态结论的 custom
  pass；先补 device-independent 结构正负例，再用 default backend、fresh process 做
  NPU 编译与 registry 观察。本轮初始阶段不修改四个 pass 的产品实现、不重建或重装
  torch_npu wheel。
- 已完成只读图实验：`fold_cast` 的同 dtype `prims.convert_element_type` 从 1 降为 0，
  fp16→fp32 转换保持 1；`fold_cat` 的同维、单用户嵌套 cat 从 2 个展平为 1 个，多用户
  inner cat 保持 2 个；`fold_clone` 的 `clone→relu` 从 1 降为 0，而直接图输出的 clone
  保持 1。三组变换后的 CPU GraphModule 与 eager 数值、dtype、stride 和输出/输入 alias
  关系一致。
- `fold_detach` 边界：`make_fx` 已在目标 pass 前把显式 `aten.detach` 规范化为
  `aten.alias`，因此常规后端图很可能属于 `target-node-not-reached`，不能通过直接调用 pass
  冒充 NPU 优化成功。直接返回 detach 时，eager 输出虽与输入共享 storage，却是
  `requires_grad=False` 的不同 Tensor 对象；若错误直返输入，数值和 storage alias 都可能
  通过，但 `requires_grad`/对象身份会变化。本轮必须额外记录这些语义字段。
- 拟修改审计资产：在
  `test/_inductor/test_dynamic_shape_fx_passes.py` 增加前三条 pass 的结构正负例，并为
  `fold_detach` 增加“前序规范化/直接调用语义边界”测试；新增独立
  `t032_b2_redundancy_compile.py`，支持多输入、tuple 输出以及 dtype/stride/storage alias/
  object identity/`requires_grad` 合同。只有测试暴露真实产品缺陷时，才另行登记修复提案；
  不在本条登记下直接改 pass 实现。
- NPU 合同：每个 case/variant 使用独立 fresh process/cache，从
  `/home/z50063656/tmp` 启动，`torch.no_grad()`、dynamic compile、default backend；目标
  registry wrapper 若执行必须恰为 1 次。positive/negative 图计数、eager/compiled 数值和
  完整语义合同必须同时通过。目标节点在 wrapper 前被消除时记
  `npu-compile-target-not-reached`；lowering/codegen、精度或语义失败分别记录，不混写为
  pass 不支持。
- 性能边界：T-032 先关闭结构、可达性与功能，不采性能结论。只有 positive 真正到达且
  功能通过的 pass 才进入单 pass on/off、warmup 10/runs 100、3 轮 paired benchmark；
  报告 mean±stdev/p50/p99、首次编译、任务数和峰值显存。设备被外部进程占用时不采性能，
  不终止其他任务。

### E-116：T-032 结构与 NPU 编译结果（2026-08-24）

- `test_dynamic_shape_fx_passes.py` 新增 8 个结构/语义边界测试，完整文件从 33/33
  提升为 41/41；测试从 `/home/z50063656/tmp` 启动，使用 Benchmark 环境已安装的
  T-031 source-built torch_npu wheel。新增测试源码尚未重建进 wheel，但调用的 pass 实现
  与当前安装态一致。
- 物理 NPU 1 在批次前后均无外部进程。8 个 fresh worker 全部以
  `npu-compile-complete` 或预期的 `npu-compile-target-not-reached` 结束；所有 tensor/tuple
  输出的数值误差为 0，dtype、stride、每个输入的 storage alias、对象身份和
  `requires_grad` 均与 eager 一致。
- `fold_cat` 正例真正到达：同维、单用户 nested cat 从 2 个变为 1 个；多用户负例保持
  2 个。它是本批唯一满足单 pass 性能准入条件的对象。
- `fold_cast` 同 dtype positive 在目标 pass 前已消失，fp16→fp32 negative 到达并保持
  cast 1→1；`fold_clone` 的内部 clone positive 在目标 pass 前已消失，直接输出 clone
  negative 到达并保持 1→1；两者记 partial reachability，不归因性能收益。
- `fold_detach` 的 detach→relu positive 在目标 pass 前已消失，直接 detach 输出整图绕过
  registry；compiled 仍正确保持 output/input 共享 storage、不同 Tensor 对象和
  `requires_grad=False`。因此当前记 reachability-neutral，不把直接调用 pass 的潜在语义
  风险误写成实际 NPU pipeline failure。
- 证据位于 `results/t032_b2_redundancy_compile_20260824/`；逐条矩阵已更新但最终 verdict
  仍为 `not-run`，其中 `fold_cat` 等待性能，另外三条等待能实际到达目标 pass 的代表图。

### T-033：fold_cat 单 pass paired 性能合同

- 状态：`completed-supported-beneficial`。只新增 audit worker、聚合器、结果和文档，
  不修改 `fold_cat` 产品实现，不重建或重装 wheel。
- workload：fp16 contiguous，三个输入分别为 `(2048,256)`、`(2048,256)`、
  `(2048,512)`，计算 `cat([cat([a,b], dim=1), c], dim=1)`；最终输出为
  `(2048,1024)`。该 shape 使 baseline 内层 cat 产生约 2 MiB 中间输出，便于观察减少一次
  cat 的任务与显存效果。
- 对照：baseline 在 registry 原位置包装并跳过 `fold_cat`，要求 cat `2→2`；candidate
  调用原 pass，要求 cat `2→1`。两侧 wrapper 必须恰调用一次，并同时通过数值、dtype、
  stride、storage alias、对象身份与 `requires_grad` 合同。
- 采样：3 轮 fresh-process paired，固定顺序 `B1,C1,C2,B2,B3,C3`；每个 worker
  warmup 10、runs 100，逐次 NPU synchronize，记录 mean±stdev/p50/p99、首次 compile、
  allocated/reserved peak。第 1 轮两侧另采 10 active steps 的 NPU task profile。
- 判定：candidate p50 改善严格超过 10%、p99 不回退超过 5%、峰值显存不恶化且图门禁
  稳定，才记 `supported-beneficial`；若延迟未过门槛但 task/显存明确减少，记
  `supported-neutral-resource-beneficial`；任何正确性或长尾失败优先判失败。所有性能结论
  需记录 CANN 9.0.1、Ascend910B2 和完整采样参数。

### E-117：T-033 fold_cat 性能关闭（2026-08-24）

- 六个 fresh worker 按 `B1,C1,C2,B2,B3,C3` 全部完成；每轮 warmup 10、runs 100，
  B1/C1 各有 warmup 1、active 10 的 NPU profile。物理 NPU 1 在批次开始和结束时均无
  外部进程；六轮的图门禁、测量前后完整语义合同和数值正确性全部通过。
- 三轮中位 baseline/candidate mean 为 `0.300154±0.006694 ms` /
  `0.269550±0.005401 ms`；p50 为 `0.298020/0.267790 ms`，改善 `10.14%`；p99 为
  `0.322720/0.289430 ms`，改善 `10.32%`。candidate p50 严格超过预登记 10% 门槛，
  p99 无回退。
- profiler 中 baseline 每 step 2 个 `aclnnCat_ConcatD_ConcatD`，candidate 每 step 1 个；
  10 个 active step 的 device task duration 合计由 `95.94 μs` 降为 `56.16 μs`。compiled
  additional allocated peak 的三轮中位数由 `6,292,480 B` 降为 `4,194,816 B`，减少
  `2,097,664 B`；additional reserved peak 两侧均为 0。
- 结论：`fold_cat` 在已登记的 fp16、contiguous、static shape 上为
  `supported-beneficial`。这是已有产品 pass 的验证成功，不需要手写 Triton 或产品源码
  修改；结论尚不外推到其他 dtype、非连续输入、dynamic width 或多层/多用户组合。
- 证据位于 `results/t033_fold_cat_performance_20260824/`；矩阵 verdict 已从 `not-run`
  更新为 `supported-beneficial`。

### T-034：B2 第三批 view/copy/where 冗余 pass 审计

- 状态：`closed-functional-audit-complete`。本轮覆盖
  `fold_sink_view`、`fold_squeeze`、`fold_to_copy`、`fold_where`、
  `fold_redundant_ops` 五条 custom pass；先补结构正负例，再做 default backend NPU
  registry 可达性与完整语义验证。初始阶段不修改产品实现、不重建或重装 wheel。
- 只读结构实验均符合源码设计：view→relu 被改写为 relu→view，多用户 view 不改；
  同 dim unsqueeze→squeeze 被删除，dim 不同则保持；内部同 dtype `_to_copy` 被删除，真实
  dtype conversion 保持；`where(mask,x,x)` 变为 clone，分支不同的 where 保持；
  view→squeeze 输出 shape 回到原输入时整链删除，否则保持。
- 重点语义：`fold_squeeze` 和 `fold_redundant_ops` 若最终直返图输入，可能把 eager 的
  “不同 Tensor 对象但共享 storage 的 view”变成输入对象本身；这与数值和 storage alias
  都无关，必须比较 object identity。`fold_where` positive 必须保留 eager where 的新 storage，
  预期由 clone 承接；`fold_to_copy` 直接图输出必须保持 copy，不得被内部-copy规则误删。
- 拟修改审计资产：在 `test/_inductor/test_dynamic_shape_fx_passes.py` 增加 10 个结构
  正负例；新增 `t034_b2_view_copy_compile.py`，复用 T-032 的多输入/tuple 输出完整语义
  合同。若目标节点在 registry 前被消除，记 `target-node-not-reached`；若实际到达后出现
  object/alias/dtype/stride/`requires_grad` 不一致，登记独立产品修复提案后才可修改 pass。
- NPU 方法：每 case/variant 使用独立 fresh process/cache，`torch.no_grad()`、dynamic
  compile、default backend，从 `/home/z50063656/tmp` 启动；目标 wrapper 执行时必须恰为
  1 次，图计数和完整语义同时通过。T-034 不采性能；只有真正到达且功能通过的 positive
  才进入后续单 pass 三轮 paired。

### T-035：fold_where 单 pass 性能验证

- 状态：`closed-supported-neutral`。T-034 已证明 `where(mask, x, x)` 在 default
  backend 的 custom registry 中真实由 1 个 `aten.where.self` 改写为 1 个
  `aten.clone.default`，且数值、dtype、stride、storage alias、对象身份和
  `requires_grad` 全部与 eager 一致；本任务只测该既有 pass，不修改产品实现、不重建 wheel。
- workload 固定为 fp16 contiguous `x=[2048,2048]` 与 bool contiguous 同 shape mask，静态
  shape、forward/no-grad、直接输出 `where(mask,x,x)`。baseline 只在 registry wrapper 中跳过
  `fold_where`，candidate 执行原 pass；两侧其他编译配置完全相同，图门禁分别要求
  `where 1→1, clone 0→0` 与 `where 1→0, clone 0→1`。
- 每侧 3 个 fresh worker，执行顺序 `B1,C1,C2,B2,B3,C3`；每轮 warmup 10、runs 100，记录
  mean±stdev、p50、p99、首次编译和 allocated/reserved peak。B1/C1 另采 warmup 1、active 10
  的 NPU task profile；每轮测量前后均复查 T-034 完整语义合同。
- 判定沿用 T-033：candidate p50 改善严格超过 10%、p99 不回退超过 5%、allocated peak 不增，
  才记 `supported-beneficial`；若延迟未过门槛但 task/显存明确减少且 p99 合格，记
  `supported-neutral-resource-beneficial`；p50 或 p99 回退超过 5% 则记性能回退。所有性能
  结论必须记录 CANN 9.0.1、Ascend910B2、warmup、runs 和三轮统计。

### E-118：T-034 第三批功能关闭（2026-08-24）

- device-independent 结构测试新增 10 条，完整文件由 41/41 扩为 51/51。10 个最终有效
  default-backend NPU 正负例均通过数值、dtype、stride、storage alias、对象身份和
  `requires_grad` 合同；物理 NPU 1 在批次开始和结束时均无外部进程。
- `fold_sink_view` positive 真实完成 reshape→relu 到 relu→reshape 的拓扑交换，多用户
  negative 保持；`fold_squeeze` 和 `fold_redundant_ops` positive 分别删除匹配组合，negative
  保持；编译边界仍保持 eager 的共享 storage/不同对象语义。`fold_where` 真实由 where
  `1→0`、clone `0→1`，保持新 storage；distinct-branch negative 保持 where `1→1`。
- `fold_to_copy` 的 same-dtype positive 在目标 pass 前已消失，dtype-conversion negative 在
  目标 pass 前已规范化为 `prims.convert_element_type`；两侧语义正确，但不得把收益归因给
  `fold_to_copy`，记 reachability-neutral。
- 首轮 4 个 graph-gate error 来自审计假设错误：动态 pipeline 把 view 规范化为 reshape，
  sink-view 正例还是恒等 view。修正 workload/门禁后的 v2 fresh worker 全部通过；原始错误
  目录保留为中性尝试，不计产品失败。证据见
  `results/t034_b2_view_copy_compile_20260824/` 与 T-034 报告。

### E-119：T-035 fold_where 性能关闭（2026-08-24）

- 六个 fresh worker 按 `B1,C1,C2,B2,B3,C3` 完成；每轮 warmup 10、runs 100，B1/C1
  另有 warmup 1、active 10 的 NPU profile。全部图门禁和测量前后完整语义合同通过。
- 三轮中位 baseline/candidate mean 为 `0.249119±0.005972 ms` /
  `0.246128±0.005860 ms`；p50 为 `0.247985/0.245115 ms`，改善 `1.16%`；p99 为
  `0.274870/0.266300 ms`，改善 `3.12%`。未达到预登记 p50 严格超过 10% 的门槛。
- 两侧均为每 step 1 个 Triton task，additional allocated peak 均为 `8,389,120 B`，reserved
  delta 均为 0。10 个 active step 的 device duration 从 `97.14 μs` 降到 `71.44 μs`，
  但 task/显存未减少，kernel 收益被端到端固定开销掩盖。
- 结论：`fold_where` 在该 fp16/contiguous/static cohort 为 `supported-neutral`。保留既有
  pass，不为此场景手写 Triton；baseline 的 `tl.where` int8 condition 弃用 warning 另作
  lowering 兼容性事项。证据位于 `results/t035_fold_where_performance_20260824/`。

### T-036：B2 layout/搬运第三批结构与 alias 审计

- 状态：`blockers-confirmed-repair-registered`。覆盖 PRE pass
  `cat_slice_cat_fold_pass` 与 `pad_slice_fold`；先用精确的 FX built-in target 构造正负例，
  再做 default-backend NPU 可达性与跨输出/storage alias 合同。初始阶段不修改产品实现、
  不重建或重装 wheel。
- 源码风险 1：`cat_slice_cat_fold_pass` 把第二个 cat 及其 slices 直接替换为第一个 cat；直接
  单输出时两者都不 alias 原输入，但若第一个 cat 同时可观察，eager 的两个 cat 输出 storage
  独立，改写后两个返回值会指向同一 storage。当前实现没有检查 cat1 的外部 users，T-036
  必须增加“输出之间的 pairwise alias/对象身份”，不能只检查相对输入的 alias。
- 源码风险 2：`pad_slice_fold` 把 pad storage 上的 slice 改为原输入上的 slice。eager slice
  alias pad 的新 storage、并不 alias 输入；改写后会直接 alias 输入，因此即使数值为 0 误差
  也可能违反语义。positive 必须以直接 slice 输出暴露该边界，negative 使用切片触及 padding
  区域并要求保留 pad。
- 拟新增结构测试：cat 完整连续覆盖/存在 gap，以及 cat1 同时输出的 alias 负例；pad slice
  完全位于原数据区/触及 padding 区域，以及 eager alias 边界。若动态实测确认 blocker，先
  登记独立修复提案和性能基线，再决定“限制改写”或“clone 保语义”，不得直接手写 Triton。
- NPU 方法沿用 T-034：fresh process/cache、`torch.no_grad()`、dynamic compile、default
  backend、从 `/home/z50063656/tmp` 启动；PRE registry wrapper 只执行 1 次。完整输出合同
  增加所有 tensor 输出之间的 storage alias 与 Python identity 矩阵。

### E-120：T-036 NPU alias blocker 确认（2026-08-25）

- 新增 6 个结构/边界测试后完整 FX 文件为 57/57；随后在物理 NPU 1 执行 6 个 fresh
  worker。cat 直接单输出 positive 真实完成 cat `2→1`、getitem `2→0`并通过；gap negative
  保持 `2→2/2→2`。pad→slice→relu positive 真实完成 pad `1→0`并通过；触及 padding 的
  negative 保持 pad `1→1`。
- cat1 同时作为第二个返回值时，目标 pass 仍把 cat `2→1`，两个输出各自相对输入的数值、
  dtype、stride、alias均看似正确且 max error 为 0；新增 cross-output 合同却发现 eager
  `output[0]/output[1]` storage 和对象均独立，compiled 两者 storage alias 且为同一对象，
  状态为 `npu-compile-semantic-failed`。
- pad slice 直接输出时，目标 pass 仍把 pad `1→0`；max error 为 0，但 eager 输出 stride
  `(6,1)` 且不 alias 输入，compiled stride `(4,1)` 且 alias 输入，状态同样为
  `npu-compile-semantic-failed`。这两个结果证明问题是可观察 alias/layout 语义，不是数值精度。
- 证据位于 `results/t036_b2_layout_alias_compile_20260825/`；修复前不得把两条 pass 记为
  NPU 可用或进入性能结论。

### E-121：P-006 最小源码实施与 wheel 安装（2026-08-25）

- 按 P-006 修改 `ascend_graph_pass.py`：cat 仅在 cat1 users 恰好等于待删除 slice 集合时
  fold；pad 仅在 slice 的每个直接 user 都可被显式证明物化新 storage 时 fold。允许集合只
  含已知非原地 elementwise/activation/matmul 方法或函数；output、view、未知和原地 user
  一律保持原图。没有增加 clone、Triton 或 C++ 修改。
- 新增 cat 可观察中间结果、pad direct output、pad view output 三个修复测试；连同本批初始
  6 个测试，完整 FX 文件由 51 条扩为 60 条。从 `/home/z50063656/tmp` 运行结果 60/60；
  两个修改文件的 `lintrunner` 为 `ok No lint issues.`。
- 保留 T-031 wheel 至
  `artifacts/torch_npu_t031_before_t036_layout_alias_fix.whl`，SHA256 为
  `29c3c105453a36d8f2eb648eeb0a2d35cfd0cb871c34697c6aaf17fb1a96a6f5`。随后从源码
  构建 T-036 wheel，SHA256 为
  `d745cf3afd6a2859a68d6c31dd02a46498264e82dedff34d726c2be2609c6b9d`，以
  `pip install --no-deps --force-reinstall` 安装。运行时导入指向 site-packages，且安装文件
  包含三个新增 guard。

### E-122：T-036 修复后 NPU 功能关闭（2026-08-25）

- 在 CANN 9.0.1、Ascend910B2、物理 NPU 1 上以 fresh process/cache 重跑 6 个 worker，
  最终全部为 `npu-compile-complete`，图门禁、shape/dtype/stride、相对输入 alias、对象身份、
  跨输出 alias 和 `requires_grad` 全部通过，所有 max/mean absolute error 为 0。
- cat safe positive 仍为 cat `2→1`、getitem `2→0`；gap negative 保持 `2→2/2→2`；
  observable cat1 场景修复后也保持 `2→2/2→2`，两输出 storage/对象继续独立。pad 的
  relu positive 仍为 pad `1→0`；触及 padding 与 direct-output 场景都保持 pad/getitem
  `1→1/1→1`，direct 输出 stride 保持 `(6,1)` 且不 alias 输入。
- 修复后首次批处理因 shell `set -e` 与 `env.sh` 探测命令组合而在 worker 前退出；不属于
  产品测试。首个 cat-alias 复测还曾把 `CPATH` 错指到 editable 源码的 `torch/include`，
  launcher 因 `ATen/ATen.h` 缺失失败；observer 已显示 pass 保持图，但未执行设备，故没有
  计为通过。改用 site-packages wheel headers 和全新目录后 6/6 关闭。失败证据保留在
  `results/t036_b2_layout_alias_fix_20260825/`，最终证据位于
  `results/t036_b2_layout_alias_fix_header_20260825/`。
- 两条 pass 当前只关闭功能可用度；性能尚未 paired 测量，矩阵 verdict 保持 `not-run`。
  下一步必须另登记 safe positive 的 pass-off/pass-on 三轮性能，不能用首次 compile/run
  或“误差为 0”宣称性能收益。

### T-037：cat-slice-cat / pad-slice 单 pass paired 性能登记

- 状态：`complete`。不修改产品源码；新增 audit-only worker/aggregate，并只测 T-036
  已通过完整语义的 safe positive。baseline 在 PRE registry wrapper 中仅跳过目标 pass，
  candidate 执行安装 wheel 的原 pass；其他 pass、default backend、dynamic/fullgraph 和输入
  完全相同。
- `cat_slice_cat_fold_pass` 使用 fp16 contiguous `a,b=(2048,512)`，构造第一个 cat 后按
  `[0:512]`、`[512:1024]` 完整切回并执行第二个 cat。图门禁要求 baseline cat/getitem
  `2→2/2→2`，candidate `2→1/2→0`；完整语义要求输出 shape/dtype/stride、不 alias 两个输入。
- `pad_slice_fold` 使用 fp16 contiguous `x=(2048,2048)`，末维右 pad 256、只切回原数据区，
  slice 后接 `relu` 物化新 storage。图门禁要求 baseline pad/getitem `1→1/1→1`，candidate
  `1→0/1→1`；完整语义要求输出不 alias 输入且 stride 保持。
- 每条 pass 独立执行 `B1,C1,C2,B2,B3,C3` 六个 fresh worker；warmup 10、runs 100，
  memory warmup 3、runs 10。B1/C1 另做 profiler warmup 1、active 10，记录 task/step、
  device duration 与 kernel 名。每轮测量前后均重新验证 T-036 完整数值/alias 合同。
- 结论必须记录 PyTorch/torch_npu、CANN 9.0.1、Ascend910B2、mean±stdev、p50/p99、
  allocated/reserved peak 和首次 compile+run。p50 改善严格超过 10%、p99 不回退超过 5%、
  allocated peak 不增加才记 `supported-beneficial`；延迟未过门槛但 task/显存明确改善且 p99
  合格时记 `supported-neutral-resource-beneficial`；p50 或 p99 回退超过 5% 记性能回退。
  首次 compile+run、单轮或 profiler kernel duration 均不能单独决定 verdict。

### E-123：T-037 两条 layout pass 性能关闭（2026-08-25）

- 每条 pass 的六个 fresh worker 均按 `B1,C1,C2,B2,B3,C3` 完成；warmup 10、runs 100，
  B1/C1 另采 warmup 1、active 10 profiler。12 个 worker 的 pass-on/off 图门禁和测量前后
  完整数值/alias合同全部通过，max/mean absolute error 均为 0；NPU 1 批次前后无外部进程。
- `cat_slice_cat_fold_pass` 三轮中位 baseline/candidate mean 为
  `0.318666±0.006658/0.241956±0.004486 ms`，p50 `0.317155/0.241035 ms`
  （+24.00%），p99 `0.333430/0.257190 ms`（+22.87%）；task 2→1，10 active steps
  device duration `107.80→56.04 μs`，additional allocated peak 两侧均 `4,194,816 B`。
  B3 baseline 有明显长尾，但每轮 p50 改善均超过 10%，前两轮也分别为 15.94%/18.60%。
- `pad_slice_fold` 三轮中位 baseline/candidate mean 为
  `0.365093±0.006282/0.250833±0.004924 ms`，p50 `0.363850/0.249770 ms`
  （+31.35%），p99 `0.383640/0.267240 ms`（+30.34%）；task 3→1，10 active steps
  device duration `487.86→105.30 μs`，additional allocated peak
  `18,874,368→8,389,120 B`（-10,485,248 B）。三轮 p50/p99 均一致改善。
- 两条均通过预登记 p50、p99 与内存 gate，关闭为 `supported-beneficial`。收益来自删除
  第二次 cat 或 pad/MemSet/storage；保留 T-036 alias guard 和既有 pass，不手写 Triton
  替身。原始证据与聚合位于 `results/t037_layout_pass_performance_20260825/`。

### T-038：B2 dtype/index/mask 首批静态与结构语义审计

- 状态：`static-risk-review-started`。优先覆盖 `dtype_optimal_pass`、
  `fold_iota_arithmetic_pass` 与 `broadcast_const_mask_compress`；先从
  `/home/z50063656/tmp` 做 device-independent FX/eager 最小正负例，不修改产品源码、不重建
  wheel。只有结构 blocker 被可重复确认后，才登记独立修复提案。
- `dtype_optimal_pass` 静态风险：PRE 实现会把 int64 arange 在端点可表示时直接改成 int32，
  也会把 float32/int32/bool/int16/int8 的 `.to(int64)` 直接改为 `.to(int32)`；当前没有检查
  结果是否作为图输出、下游是否要求 int64，或 float 值是否超出 int32。最小反例必须同时
  检查值与输出 dtype，不能把数值相等视为语义通过。
- `fold_iota_arithmetic_pass` 静态风险：iota downcast 已有 transparent/closing closure，
  但 `cmp(sub(a,b),0)→cmp(a,b)` 没有 dtype/range gate。float 的 `inf-inf` 产生 NaN，而
  `inf>=inf` 为真；定宽整数 subtraction 还可能溢出。必须加入有限浮点、Inf/NaN 与整数极值
  边界，区分数学恒等式和实际 dtype 语义。
- `broadcast_const_mask_compress` 静态风险：它把
  `cast(where(mask, full(shape,c1), full(shape,c2)))` 改为 cast(mask) 或
  logical_not(mask)，并明确删除显式 broadcast；当前没有证明 mask shape 等于 where 输出
  shape，也没有证明较小结果会在所有下游重新广播。最小反例使用小 mask 与大 full shape
  直接输出，检查 shape/stride/value。
- 若 CPU FX 已确认上述 blocker，下一步先补结构测试和 default-backend NPU observer；修复
  候选优先缩窄 capability gate（输出/未知 consumer 保持原图），不得用 Triton 掩盖 dtype、
  overflow 或 shape 语义问题。
- NPU observer 在 wheel 安装态从 `/home/z50063656/tmp`、物理 NPU 1 上运行，每个 profile
  使用独立 debug/cache/output 目录并要求目标 pass 恰好被调用一次。`dtype_optimal_pass`
  覆盖 int64 arange 直出保持、arange→bool comparison 降级、float32→int64 直出保持与
  int32→int64→comparison 降级；`fold_iota_arithmetic_pass` 覆盖 safe iota comparison 的
  int64→int32 和 `sub→ge(0)` 的 Inf/int32-overflow 保持；
  `broadcast_const_mask_compress` 覆盖 equal-shape 压缩及 `(1,N)→(B,N)` broadcast 保持。
- 图门禁同时记录目标节点 before/after 和相关 dtype；输出合同检查嵌套结构、shape、dtype、
  stride 与数值，整数/bool 要求严格相等。目标图在 pass 前已被其他阶段消除时单列
  `target-not-reached`，不得算 pass 通过。执行前 `npu-smi info` 显示物理 NPU 1 无外部进程；
  本环节只做功能与 reachability，不采性能、不据单次首次编译时间判优。

### E-124：T-038 CPU FX blocker 确认（2026-08-25）

- 从 `/home/z50063656/tmp` 对安装态 pass 做手工 FX+ShapeProp 前后执行。int64
  `arange(4)` 的值仍为 `[0,1,2,3]`，但输出 dtype 从 int64 变成 int32，证明零数值误差
  不能覆盖 dtype 合同；float32 `[1.9,3_000_000_000]→int64` 被改为 int32 后，第二个值从
  `3,000,000,000` 变成 `2,147,483,647`，同时 dtype 错误。
- bool mask `(1,3)` 与两个 full `(2,3)` 构成 where 再 cast 时，原输出 shape 为 `(2,3)`；
  `broadcast_const_mask_compress` 删除显式广播后输出变为 `(1,3)`。值前缀相同不能弥补
  shape/stride 语义错误。
- `fold_iota_arithmetic_pass` 的 cmp-sub 初次 ShapeProp 反例未触发，因为手工图只有
  `tensor_meta` 而真实 POST 判断读取 `meta['val']`；补齐该 metadata 后目标确实从
  `aten.ge.Scalar(sub(a,b),0)` 改为 `aten.ge.Tensor(a,b)`。float32 `inf/inf` 结果
  `False→True`；int32 `INT_MIN-1` 溢出例结果 `True→False`。这不是 observer 假阴性，而是
  无 dtype/range gate 的实际语义 blocker。
- 当前三条 pass 均不得直接记 NPU 可用或进入性能；先按 P-007 增加保守 guard 与结构回归，
  再重建 wheel 做 NPU 正负例。原反例是 device-independent 证据，不冒充 NPU 编译结果。

### E-125：T-038 源码态加载中性尝试（2026-08-25）

- P-007 的最小 guard 与 7 个结构/语义回归已写入登记的两个源文件，`py_compile` 与
  lintrunner 均通过。首次从 `/home/z50063656/tmp` 直接把 torch_npu 源码根加入
  `PYTHONPATH` 运行测试时，在 PyTorch backend entry-point 自动加载阶段因源码树没有已编译
  的 `torch_npu._C` 而退出；尚未进入任何 pass 测试，因此这条记录是环境/加载方式失败，
  不是功能失败，也不得计入通过率。
- 下一次只改变加载隔离方式：设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，让测试文件已有的
  namespace fallback 从源码路径加载纯 Python pass；若 67 项结构测试通过，再保留当前
  T-036 wheel、重建 source wheel 并按用户指定方式 `--no-deps` 安装，最终以安装态和 NPU
  fresh worker 为准。不得把 namespace fallback 的结果冒充正式 wheel 验证。

### E-126：T-038 namespace config stub 补齐（2026-08-25）

- 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 后已越过 `_C` 自动加载，但测试文件的轻量
  `torch_npu._inductor.config` stub 只提供 `log`；当前 `ascend_graph_pass.py` 同时导入
  `is_ascend950`，因此仍在 unittest 收集前以 `ImportError` 退出，没有执行任何 pass case。
- 允许的最小测试基础设施修改是在已有 config stub 上增加 `is_ascend950=False`。该值只服务
  CPU/device-independent FX 测试，不修改产品 config 或 NPU capability；正式结论仍要求
  source wheel 安装态测试。若补齐后仍有加载依赖，继续按“收集前环境失败”单独记录，不能
  删除正负例来换取通过。
- 首次补齐后，导入继续到 `register_custom_pass.py`，确认同一 config stub 还缺
  `enable_fused_matmul_relu`；这同样发生在收集前。该模块只用它决定默认关闭测试无关的
  matmul+relu pass，因此把 stub 设为 `False`，与产品默认关闭策略一致，并再次运行完整集合。
- 补齐两个 stub 后，同一源码态命令完成 67/67：既有 60 项与 P-007 新增 7 项全部通过，
  日志为 `results/t038_source_fx_tests_20260825.log`。这证明纯 Python 图改写的结构合同通过，
  但仍只作为重建 wheel 前的快速门禁。
- 重建前将当前 T-036 安装来源 wheel（SHA256
  `d745cf3afd6a2859a68d6c31dd02a46498264e82dedff34d726c2be2609c6b9d`）保留为
  `artifacts/torch_npu_t036_before_t038_dtype_mask_fix.whl`，然后才清空 `dist` 产出 T-038
  wheel。这样失败时能精确恢复，不覆盖更早 T-031 基线。
- 源码 wheel 构建完成，退出码 0；新 wheel SHA256 为
  `dffad49056538fc4250b444b2c40a619db3b0897b00f8906f53757a857b167d8`。构建日志中的
  Kineto、setuptools/AOTI snapshot 与 C++ reorder 均为 warning，未导致编译失败；wheel
  内文件检查确认包含三处 P-007 guard。下一步只用 `pip install --no-deps --force-reinstall`
  安装这一 wheel，然后从 `/home/z50063656/tmp` 校验 import 来源、源码标记与 67 项测试。

### E-127：T-038 首批 NPU worker 头文件环境失败（2026-08-25）

- 新 wheel 已按 `pip install --no-deps --force-reinstall` 成功安装；运行时 torch_npu 来自
  Conda site-packages，三处 guard 均存在，安装态 FX 测试 67/67 通过。
- 未设置额外 include path 的首批 NPU batch 中，前 4 个 dtype worker 均已真实调用目标
  pass 一次，observer 的图门禁分别显示危险直出保持 int64、安全 comparison 闭包改为
  int32；但随后每个 worker 都在 Triton launcher GCH 编译时从 editable PyTorch 的空
  `torch/include` 查找 `ATen/ATen.h` 并失败，未执行生成 kernel。第五个 worker在确认重复
  根因后主动中断，其余未启动；这些结果不能计入 NPU 功能通过或失败。
- 原始失败保留在 `results/t038_dtype_index_mask_compile_20260825/`。复测沿用 T-036 已验证
  的环境修正：只对 fresh worker 的 `CPATH` 前置 Conda site-packages 下 PyTorch wheel
  headers，并使用新结果根 `results/t038_dtype_index_mask_compile_header_20260825/`；不建
  软链、不修改共享安装或产品源码。若暴露新的产品/版本合同，必须重新分类而不能继续堆
  环境 workaround。

### E-128：T-038 fresh-cache 与强制禁用缓存冲突（2026-08-25）

- 补齐 wheel headers 后，首个 arange worker 已不再报缺头文件；但审计脚本同时设置
  `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`，Ascend Triton 为同一个 hash 的多个 launcher
  candidate 反复串行生成约 179 MiB 的 `precompiled.h.gch`，4 分钟后仍未进入设备执行。
  `ps` 显示 cc1plus 持续占用 CPU，属于重复预编译而非 hang 或 pass 错误；为避免污染后续
  8 项，该 batch 已主动中断，且没有写成完成 result。
- 每个 worker 本来就使用全新的独立 Inductor/Triton cache 目录，足以避免旧产物污染。
  因此审计脚本把 `TORCHINDUCTOR_FORCE_DISABLE_CACHES` 改为 `0`，只允许同一 fresh worker
  内复用它刚生成的 GCH/launcher，不跨 profile 共用 cache。复测另用
  `results/t038_dtype_index_mask_compile_cached_header_20260825/`，不得把缩短编译时间解释为
  pass 性能收益。

### E-129：T-038 fresh launcher C++ 标准合同确认（2026-08-25）

- 将 force-disable 改为 0 后首个 arange worker 仍对同一 GCH 反复启动编译器。进一步读取
  driver 与进程状态确认：cache 目录确实固定，但每次 GCH 编译都使用 Triton Ascend 写死的
  `-std=c++17`，不满足当前 PyTorch 2.14 headers 的 C++20 合同；heuristic 捕获该 config
  失败后继续尝试下一个 config，导致未等到汇总 stderr。该轮在约 4 分钟后中断，E-128
  关于“仅由 force-disable 导致”的假设被证伪并保留，不覆盖为成功经验。
- 继续功能审计复用已在 T-022 登记的两个 audit-only 文件：以环境变量 `CC` 指向
  `t022_launcher_cc_wrapper.sh`，它只把 launcher 参数 `-std=c++17` 改为 C++20，并前置
  `t022_cann_header_compat.h` 中两个当前 CANN 缺失、且 launcher 不调用的 conditional graph
  类型；同时设置 `TRITON_DISABLE_PRECOMPILE=1` 跳过 GCH。新证据根为
  `results/t038_dtype_index_mask_compile_audit_shim_20260825/`。
- 该垫片不改 device Triton kernel、torch_npu wheel 或产品源码，适合判断 pass reachability、
  图门禁与设备数值；但任何通过只能记为 `development/audit-shim` capability，正式 fresh
  launcher 环境缺口仍存在。若 wrapper 下出现新错误，停止扩展 workaround 并按实际根因
  分类。

### E-130：T-038 修复后 NPU 功能关闭（2026-08-25）

- 在 CANN 9.0.1、Ascend910B2、物理 NPU 1 上，9 个 fresh process/cache worker 均以
  `development/audit-shim` 方式完成 NPU 编译执行，结果全部为 `npu-compile-complete`；批次
  前后 NPU 1 均无外部进程。所有输出都严格逐元素相等，mismatch count 为 0，shape、dtype、
  stride、`requires_grad` 与相对输入 alias 合同全部一致。
- `dtype_optimal_pass` 四项图门禁全部通过：int64 arange 直出保持 int64；arange→comparison
  为 int64→int32；float32→int64 直出保持 int64，并正确保留 `±3,000,000,000`；
  int32→int64→comparison 为 int64→int32。两个 comparison 输出均保持 bool。
- `fold_iota_arithmetic_pass` 的 safe iota comparison 为 int64→int32；Inf/Inf 与
  `INT_MIN-1` 两个反例都保持 `aten.sub.Tensor 1→1`、`aten.ge.Scalar 1→1`，输出严格匹配，
  证明停用不安全子改写后 NPU 边界恢复。
- `broadcast_const_mask_compress` 的 `(32,4096)` equal-shape 正例把 where `1→0`、full
  `2→0`；`(1,4096)` mask 广播到 `(32,4096)` 的反例保持 where `1→1`、full `2→2`。
  两者输出都是 float32 contiguous `(32,4096)`、stride `(4096,1)` 且不 alias 输入。
- 原始结果位于 `results/t038_dtype_index_mask_compile_audit_shim_20260825/`。首次 compile+run
  约 17.19–30.42 秒，只反映 fresh compile 与 launcher 开销，不用于性能 verdict。正式环境
  仍有 E-129 的 C++17/C++20/CANN header 缺口，因此当前关闭的是 pass 功能修复和开发态
  device capability，不宣称无 shim fresh launcher 已通过。

### T-039：dtype/index/mask safe-positive 单 pass paired 性能登记

- 状态：`completed-development-audit-shim`。未再修改产品 pass；仅新增 audit-only
  worker/aggregate。三条
  candidate 均使用 T-038 已通过完整语义的 safe positive，baseline 在目标 pass 的 registry
  wrapper 中跳过且仍调用其余所有 pass，candidate 正常执行目标 pass。两侧都使用 T-022
  audit-only C++20/CANN launcher 垫片，性能 verdict 必须标注 `development/audit-shim`。
- `dtype_optimal_pass`：int32 input `(1,048,576)`；构造 int64 arange 与 input `.to(int64)` 后
  比较。baseline 两节点保持 int64，candidate 两节点改 int32，输出 bool；意图测量 index
  算术/访存降宽，不把输出 dtype 改变纳入收益。
- `fold_iota_arithmetic_pass`：int32 input `(1,048,576)`；直接构造 prims iota int64 并与
  `.to(int64)` input 比较。baseline iota 保持 int64，candidate iota 改 int32；PRE dtype pass
  在两侧相同运行，隔离 POST iota rewrite。
- `broadcast_const_mask_compress`：bool mask `(1024,1024)`，两个同 shape fp16 full(1/0)、
  where 后 convert 到 fp32。baseline 保持 where `1`/full `2`，candidate 删除为 `0/0`；输出
  shape/dtype/stride 与 alias 必须一致。即使 task 数相同，也要检查 device duration 与稳态
  host latency，不能仅凭删节点宣称获益。
- 每条 pass 独立执行 `B1,C1,C2,B2,B3,C3` 六个 fresh worker；warmup 10、runs 100，memory
  warmup 3、runs 10。B1/C1 另做 profiler warmup 1、active 10。每次测量前后重新检查严格值、
  shape/dtype/stride/alias 合同与图门禁；物理 NPU 1 运行，批次前后检查外部进程。
- 记录 PyTorch/torch_npu、CANN 9.0.1、Ascend910B2、mean±stdev、p50/p99、首次
  compile+run、allocated/reserved peak、task/active-step 和 device duration。三轮中位 p50
  改善严格超过 10%、p99 不回退超过 5%、additional allocated peak 不增加才记
  `supported-beneficial-development-shim`；延迟未过门槛但 task/显存明确改善且 p99 合格时
  可记 `supported-neutral-resource-beneficial-development-shim`；其余按 neutral/regressed
  记录。首次编译与单轮结果不参与放行。

### E-131：T-039 三条 safe-positive paired 性能关闭（2026-08-25）

- 物理 NPU 1 上按 `B1,C1,C2,B2,B3,C3` 完成 18/18 fresh worker；批次前后 NPU 1 均无
  外部进程。每个 worker 的图门禁、测量前后完整输出合同均通过，严格逐元素 mismatch
  count 都为 0。环境为 PyTorch `2.14.0a0+git8e86e0a`、torch_npu
  `2.14.0a0+git83cc452`、Triton 3.2.0、CANN 9.0.1、Ascend910B2，证据范围为
  `development/audit-shim`。
- `dtype_optimal_pass` 三轮中位 p50 `0.550140→0.263725 ms`，改善 `52.06%`；p99
  `0.571360→0.282910 ms`，改善 `50.48%`。profiler active 10 总 device duration
  `3591.94→67.78 µs`，每步 task `1→1`，additional allocated peak
  `1,049,088→1,049,088 B`。判定 `supported-beneficial-development-audit-shim`。
- `fold_iota_arithmetic_pass` 三轮中位 p50 `0.557745→0.246625 ms`，改善 `55.78%`；
  p99 `0.577820→0.262860 ms`，改善 `54.51%`。profiler active 10 总 device duration
  `3584.54→54.88 µs`，每步 task `1→1`，additional allocated peak
  `1,049,088→1,049,088 B`。判定 `supported-beneficial-development-audit-shim`。
- `broadcast_const_mask_compress` 三轮中位 p50 `0.244325→0.243580 ms`，只改善
  `0.30%`；p99 改善 `1.01%`，task `1→1`，additional allocated peak
  `4,194,816→4,194,816 B`。baseline 已将 full/where/convert 融为单 task，删 FX 节点没有
  形成端到端收益，判定 `supported-neutral-development-audit-shim`，不做 Triton 替身。
- 初版 aggregate 曾把单次 profiler device duration 略降视为 resource 改善；复核 T-039
  预登记后收紧为只接受 task 或显存明确下降，mask verdict 从中间态 resource-beneficial
  修正为 neutral。原始 18 个 worker 未重跑、未挑选。最终结果在
  `results/t039_dtype_index_mask_performance_20260825/*_aggregate_final/aggregate.json`，报告为
  `report/t039_dtype_index_mask_performance_20260825.md`。

### T-040：剩余 dtype/mask/hamming 三 pass 语义与结构登记

- 状态：`planned-static-and-fx-first`。目标为 `masked_add_compose_pass`、
  `bool_cast_mul_to_where_pass`、`sign_diff_hamming_fuse_pass`；先不修改产品源码、不构建
  wheel、不运行性能。只新增/扩展目标 FX 测试与 audit-only 观察脚本；若发现反例，必须在
  本节补登记 capability 修复、回滚边界与验证计划后才能改 pass。
- `masked_add_compose_pass`：正例要求两个 where 的 mask 是同一 bool 源的一正一反、两
  where 只有 add 一个用户、alpha=1；验证 where `2→1`、add `1→0`。负例覆盖非互补 mask、
  multi-user 与 alpha≠1。额外检查浮点 signed zero：原式选中值仍会与另一支的 `+0` 相加，
  新式直接选择值，逐元素普通相等可能看不出 `signbit` 差异。
- `bool_cast_mul_to_where_pass`：正例覆盖 direct cast 与单用户 view/expand chain；负例覆盖
  dtype 不一致与 cast multi-user。必须先检查 `false * Inf/NaN` 和浮点 signed zero：乘法
  的 IEEE 结果不一定等于 where false 分支的常量 `+0`，若形成反例，浮点路径不能按普通
  有限随机样本放行。
- `sign_diff_hamming_fuse_pass`：正例覆盖有限实数、dim 与 keepdim；负例覆盖链条 multi-user
  或非目标拓扑。必须检查 NaN：原 `sign→relu→sub→abs→sum` 会传播 NaN，而
  `gt(NaN,0)` 为 false，可能把 NaN 静默变成有限计数；同时检查输入 dtype 与输出 dtype。
- 结构测试从 `/home/z50063656/tmp` 启动，记录变换前后目标节点计数和严格值、dtype、shape、
  stride、alias、`signbit`/NaN 合同。只有源码态与安装态目标测试通过，才登记 fresh NPU
  compile；只有 NPU 功能与 generated code 通过，才为 safe positive 预登记单 pass paired
  性能。手写 Triton不能修复 IEEE/shape/alias 语义，不在本阶段使用。

### E-132：T-040 CPU eager 与当前 installed-pass 边界反例（2026-08-25）

- 从 `/home/z50063656/tmp` 使用当前 installed wheel 构造真实 aten FX 图。
  `masked_add_compose_pass` 确实执行 where `2→1`、add `1→0`；输入两支选中值为 `-0.0`
  时，原式输出 signbit 为 false（`-0 + +0 = +0`），重写输出 signbit 为 true。普通数值比较
  仍会把二者视为相等，因此这是新的“数值误差 0 但语义失败”。
- `bool_cast_mul_to_where_pass` 确实执行 cast `1→0`、mul `1→0`、where `0→1`；mask=false
  对应 `Inf/NaN` 时原乘法输出 NaN，重写却输出 `0.0`。这不是 NPU 精度容差，而是 IEEE
  分类改变；signed zero 也存在相同风险。
- `sign_diff_hamming_fuse_pass` 在包含 NaN、±Inf、±0 的 float32 样本上执行 sign `2→0`、
  gt `0→2`、ne `0→1`，原/新输出均为 float32 标量 4。当前 PyTorch `sign(NaN)` 与
  `gt(NaN,0)` 对此模式一致，所以该样本没有形成反例；仍需 integer、keepdim、multi-user
  与真实 NPU 关闭。

### P-008：mask arithmetic IEEE capability 缩窄

- 状态：`verified-functional-performance-pending-development-audit-shim`。只修改 `ascend_graph_pass.py` 与
  `test_dynamic_shape_fx_passes.py`，不修改 PyTorch/Triton/C++、不新增 kernel。回滚边界是
  一个整数/布尔 exact-zero dtype allowlist、前两条 pass 的两个 guard 和对应新增测试。
- `masked_add_compose_pass` 只有两支 value dtype 都属于 bool/uint8/int8/int16/int32/int64
  时才折叠；浮点与复数保持两 where 加 add，因为直接选择不能保持加正零后的 signbit/NaN
  完整语义。`bool_cast_mul_to_where_pass` 同样只允许上述 exact-zero dtype；浮点与复数保持
  cast×x，因为 `0×Inf/NaN` 不能替换为常量零。
- `sign_diff_hamming_fuse_pass` 本轮不增加 dtype guard；先用特殊浮点、整数、keepdim 和
  multi-user 测试验证其现有等价域。若 NPU 或扩展用例出现反例，再单独登记，不与前两个
  已证实问题捆绑推断。
- 目标测试至少新增：masked integer positive、非互补 negative、float signed-zero
  negative；bool-cast integer direct/view positive、float Inf/NaN/signed-zero negative；
  hamming special-float positive、integer keepdim positive、multi-user negative。测试必须检查
  节点计数和严格 dtype/shape/value，浮点边界另查 `isnan`/`signbit`。完成源码态与安装态
  全文件、source wheel `--no-deps` 安装和 fresh NPU 正/负例后，才进入性能候选选择。

### E-133：T-040 exact-zero guard、wheel 与 9/9 NPU 功能关闭（2026-08-25）

- P-008 已按登记范围实施：`ascend_graph_pass.py` 新增 bool/uint8/int8/int16/int32/int64
  exact-zero allowlist，`masked_add_compose_pass` 检查两支 value dtype，
  `bool_cast_mul_to_where_pass` 检查乘法另一输入 dtype；没有修改 PyTorch、Triton、C++ 或
  vendor kernel。目标测试文件新增 9 项，完整源码态与安装态均为 76/76。
- 旧 T-038 wheel 已归档为
  `artifacts/torch_npu_t038_before_t040_mask_ieee_fix.whl`，SHA256
  `dffad49056538fc4250b444b2c40a619db3b0897b00f8906f53757a857b167d8`。新 source-built
  wheel SHA256 为 `b273aeedcb9d1367de65328bea78a448ff1eb81fa4a85dca3f910e556c7b2460`，已用
  `--no-deps --force-reinstall` 安装；wheel 内检查确认 allowlist 与两处 guard 存在。
- 物理 NPU1 在运行前后均无其他进程。default backend、fullgraph、inference、fresh cache
  的 9/9 audit-shim worker 全部为 `npu-compile-complete`，所有图门禁与完整输出合同通过，
  累计 mismatch 0：masked 整数 where/add `2/1→1/0`，非互补与浮点边界保持；bool 整数
  direct/view 的 cast/mul `1/1→0/0` 并新增 where，浮点非有限路径保持；hamming 特殊浮点与
  整数 keepdim 的 sign/relu/sub/abs 被 gt/ne 替换，multi-user 保持原链。
- 第一次 NPU worker 的 `CPATH` 误指 editable PyTorch 源码 include view，launcher 因
  `ATen/ATen.h` 缺失失败；改为 site-packages wheel headers 后在新目录通过。第一次
  installed 测试同时关闭 autoload 并在 stub 后显式加载 native torch_npu，导致
  `_npu_dtype_cast` schema 重复注册；正常 autoload 下 76/76。两者均保留为环境/启动方式
  中性证据，不计 pass 失败。
- 聚合结果为
  `results/t040_mask_hamming_compile_20260825/aggregate/aggregate.json`，详细报告为
  `report/t040_mask_hamming_semantic_fix_20260825.md`。三条矩阵记录已更新到功能通过、性能
  `not-run`；下一步必须另登记 T-041 单 pass paired 性能，优先 sign-Hamming，再测 masked
  add 与 bool-cast direct/view。功能通过不能提前写成 beneficial。

### T-041：mask/hamming safe-positive 单 pass paired 性能登记

- 状态：`approved-audit-only-performance`。不修改产品源码、wheel、PyTorch 或 Triton；只
  新增 T-041 worker/聚合器与报告。所有 baseline/candidate 都加载同一 T-040 wheel，baseline
  只让目标 registry pass 返回而不执行，candidate 执行原 pass；其余 backend、输入和编译
  配置相同。由于 fresh launcher 仍需 T-022 垫片，证据范围统一为
  `development-audit-shim`。
- 四个 representative case：masked-add 为 1,048,576 元素 int32；bool-cast direct 为
  1,048,576 元素 int32；bool-cast view-chain 为 `(262144,)` mask 经 unsqueeze 乘
  `(262144,4)` int32；sign-Hamming 为 `(1024,1024)` float32 沿 dim1 reduce。只测 T-040
  已验证的 safe positive，不用危险浮点 mask 路径做性能候选。
- 每 case 三轮、每轮两个 fresh process/cache，顺序固定 `B-C / C-B / B-C`；warmup 10、
  runs 100、逐次同步。每 case 第一轮两侧另做 warmup1/active10 NPU profiler；每个 worker
  做 memory warmup3/runs10。运行前后都要确认物理 NPU1 无外部进程。
- worker 必须在测量前后通过完整 tensor contract；图门禁分别证明 baseline 不改写、candidate
  按预期改写。报告 mean±stdev、p50/p99、首次 compile+run、task 数、device duration 与
  allocated/reserved peak。首次编译不计 steady verdict。
- 主性能 gate 为三轮 p50 中位数改善严格超过 10%，且 p99 不得回退超过 5%；resource
  beneficial 只接受 task 数或峰值显存明确下降，不能把一次 profiler duration 略降单独算作
  resource 收益。未过 gate 而功能正确记 `supported-neutral`，不手写与已有单 kernel 重复的
  Triton；若多 task/profile 显示明确 launch/中间访存瓶颈，再单独登记 replacement。

### E-134：T-041 四 case、24/24 worker 性能分流（2026-08-25）

- 物理 NPU1 运行前后均无其他进程；四 case 的三轮 baseline/candidate worker 全部完成，
  图门禁与测量前后完整语义均通过。结果位于
  `results/t041_mask_hamming_performance_20260825/`，报告为
  `report/t041_mask_hamming_performance_20260825.md`。
- masked-add p50/p99 `0.260785/0.295460→0.251115/0.266380 ms`，改善
  `3.71%/9.84%`；task 1→1、allocated peak不变，为 `supported-neutral`。
- sign-Hamming p50/p99 `0.266480/0.290000→0.256770/0.274090 ms`，改善
  `3.64%/5.49%`；task 1→1、allocated peak不变，为 `supported-neutral`。两条 active10
  device duration 虽分别从 81.10→64.18 μs、126.46→86.10 μs，但不能单独升级 verdict。
- bool-cast direct p50 回退 `0.69%`、p99 回退 `19.02%`，task/显存无收益；active10
  duration 又从 62.56 增至 73.86 μs，为 `supported-performance-regressed`。view-chain
  p50/p99 改善 `36.30%/39.90%`，active10 duration 2377.58→96.28 μs，task/显存不变，
  为 `supported-beneficial`。同一 pass 的 capability 必须据此继续缩窄，不能整体标一个
  不带条件的成功。

### P-009：bool-cast-mul 只保留 view-chain 性能正域

- 状态：`verified-supported-beneficial-view-chain-development-audit-shim`。只修改 `ascend_graph_pass.py` 与
  `test_dynamic_shape_fx_passes.py`；不修改 PyTorch/Triton/C++、不新增 kernel。回滚边界是
  一个 `chain` 非空 guard 和 direct 结构断言。
- `_walk_back_view_chain_to_cast` 对 direct cast 返回空 list，对 unsqueeze/squeeze/view/
  reshape/expand 等路径返回非空 chain。`bool_cast_mul_to_where_pass` 只有 chain 非空时才
  rewrite；direct cast×x 保持原图，避免 T-041 已证实的 p99/device 回退；view-chain 的
  dtype、single-user 与 fake-meta guard 全部保持不变。
- direct integer 测试改为 cast/mul `1/1→1/1`、where `0→0`；view-chain 正例继续要求
  cast/mul `1/1→0/0`、where `0→1`、unsqueeze `1→1`；浮点边界保持。修改后运行源码态与
  安装态 76/76，重建 source wheel 并 `--no-deps` 安装，再跑 direct/view/float NPU
  observer。view 路径代码未变，T-041 的 view paired 数据可沿用；若功能图门禁改变则重测。

### E-135：T-042 view-chain guard、wheel 与安装态关闭（2026-08-25）

- 已加入 `if not chain: continue`，direct integer 测试改为保持 cast/mul，view-chain 与
  float 测试不变；源码态和安装态全文件均为 76/76。T-040 wheel 已归档，SHA256
  `b273aeedcb9d1367de65328bea78a448ff1eb81fa4a85dca3f910e556c7b2460`；T-042 新 wheel
  SHA256 `ea801e791373b0bd3adf9d4bfb6253ace75afa800c71b0451c9b206e4664fe5a`，已
  `--no-deps --force-reinstall` 安装。
- 物理 NPU1 前后空闲，direct int32、view-chain int32、float Inf/NaN 三 worker 为 3/3
  `npu-compile-complete`，图门禁与完整输出合同通过、累计 mismatch 0。direct/float 保持，
  view cast/mul→where 继续触发。聚合结果位于
  `results/t042_bool_view_guard_compile_20260825/aggregate/aggregate.json`。
- view candidate 代码未变，沿用 T-041 的 p50/p99 +36.30%/+39.90%；direct 现在等于同轮
  baseline，不再执行已证实回退的 where rewrite。因此 pass 最终 scope 为
  `supported-beneficial`（exact-zero integer/bool + non-empty single-user view chain）；不需
  Triton replacement。详细报告为 `report/t042_bool_view_guard_integration_20260825.md`。

### T-043：B2 最后三条复合 pass 的结构与语义审计登记

- 目标只覆盖 `batch_embedding_fusion_pass`、`fused_matmul_relu_pass` 与
  `fusion_attention_v3_pass`。先新增 audit-only CPU/FX 反例脚本与报告，不改产品源码、
  不重建 wheel、不运行性能；若确认 blocker，再另立提案后才允许修改
  `ascend_graph_pass.py` 和目标 FX 测试。
- batch-embedding 正例使用同一权重、同一父 indices、两个完整等长连续 slice、默认
  embedding 参数和同一种默认 reduce，要求 embedding/reduce 由 2→1 且 eager 前后值、
  dtype、shape 一致；cat-collapse 另核对输出 stride/alias。负例至少覆盖 slice `step!=1`、
  negative dim、gap/overlap、非默认 embedding 参数、reduce `dtype=`、multi-user 与不同权重。
  特别检查 pass 是否忽略 slice step 或 reduce dtype；普通样本误差为 0 不能替代这些合同。
- fused-matmul-relu 在当前 910B2 上受 `is_ascend950` 与 default-off 双门禁；本轮先证明 B2
  不注册/不改图并静态核对 fp16/bf16、rank、bias、single-user、mutation guards。不得在 B2
  monkeypatch A5 capability 后宣称设备可用或有性能；A5 真实功能/性能留作对应硬件验证。
- fusion-attention-v3 先核对旧/新 schema、tuple 输出和 PRE runner 调用次数。正例仅在能
  证明参数与被使用输出等价时允许 old→v3；负例覆盖旧 op 的第 4–6 个 scalar auxiliary
  输出、`dropout_mask/seed/offset`、list 型 actual-seq 参数以及 inference runner 重复执行。
  任何 schema/tuple 不等价都先判 correctness blocker，不能用只读取 output[0] 的模型掩盖。
- 所有审计命令从 `/home/z50063656/tmp` 启动，使用 T-042 已安装 source wheel；脚本先
  `py_compile`，结果保存到 `results/t043_b2_composite_static_20260825/`。结构层通过后再登记
  fresh NPU positive/negative observer；只有功能闭环后才登记 pass-on/pass-off paired 性能。

### E-136：T-043 只读源码风险定位（2026-08-25）

- `batch_embedding_fusion_pass` 的 coverage 证明只读取 slice 的 start/end，当前没有校验第
  5 个 `step`；`_reduce_call_args()` 只重建 input/dim，未转发 reduce 的 `dtype=`。这两处均
  可能让图结构看似成功而改变取值或输出 dtype，必须由登记反例实证后再决定 guard/透传。
- `fusion_attention_v3_pass` 当前把旧节点的全部 `args/kwargs` 原样送入 v3 并复制 meta，
  但源码 schema 显示旧 op 返回 4 Tensor + 3 int，v3 返回 6 Tensor；旧 op 还多出
  `dropout_mask/seed/offset`，actual-seq 参数类型也不同。当前不能把它视为无条件等价替换。
- `run_register_pre_custom_passes()` 在 inference 下先运行全部 PRE pass，随后又按名字运行
  `fusion_attention_v3_pass`；变换本身可能因第二次无旧节点而幂等，但 pass observer/编译
  开销会执行两次。以上均为只读结论，尚未修改源码或矩阵 verdict。

### E-137：T-043 CPU/FX 反例确认（2026-08-25）

- audit-only 脚本 `t043_b2_composite_static.py` 已从 `/home/z50063656/tmp` 在 T-042 安装态
  运行，结果为 `results/t043_b2_composite_static_20260825/result.json`。普通 embedding
  正例确实把 embedding/sum 由 2/2→1/1、数值误差为 0，但两个原本不 alias 的 reduce 输出
  变为同一 combined reduce 的 select，`_is_alias_of` 从 false→true；因此“误差为 0”仍不
  足以证明功能正确。
- slice `step=2` 反例同样被 2/2→1/1，两个输出最大绝对误差为 `9/129`；显式
  `sum(dtype=float32)` 也被改写，输出 dtype 从 float32 变为 float16，虽然该特定可表示输入
  的逐元素误差恰为 0。三项分别确认 value、dtype 与 storage-alias blocker。
- attention schema 运行态确认旧/v3 为 `24→21` 参数、`7→6` 返回；当前 pass 会生成
  `getitem(v3, 6)`，也会把旧 op 的 24 个位置参数原样送给只有 21 个参数的 v3，并复制旧
  tuple meta。隔离 registry spy 又确认 inference runner 调用该 pass 2 次。
- 当前 910B2 下 `is_ascend950=false`，`fused_matmul_relu_pass` 不在 POST registry，fused op
  解析为 None，直接调用 pass 后 mm/relu 都保持 1→1；当前机器结论为 device-gated
  not-applicable，不冒充 A5 功能/性能结果。

### T-044：B2 composite 修复的 installed-wheel NPU 功能登记

- 使用 P-010 构建并安装的 wheel；每个 profile 使用物理 NPU1 的独立 fresh process/cache、
  default backend、`fullgraph=True`、静态 no-grad。运行前后检查设备空闲，目标 pass wrapper
  必须恰好记录一次 before/after，不读取成功 case 的 `output_code.py`。
- batch-embedding 运行 default tuple positive、cat-collapse positive、slice-step negative 与
  reduce-dtype negative 四项。正例要求 embedding/sum 2/2→1/1；tuple positive 还要求两个
  输出与 eager 一致且彼此不 alias，cat positive 不得插 clone；两个 negative 要求 2/2
  保持。所有输出核对 shape/dtype/stride、input alias 与 output-output alias。
- attention 先用 legacy 与 v3 eager 对照验证前四个 tensor 输出，再编译只消费 0–3 的 safe
  profile，要求 old/v3 1/0→0/1、v3 fake meta 为 6-tuple、输出在 fp16 容差内且 NaN 分类
  一致。旧 scalar aux/full legacy args 已由 84/84 FX 负例固定，本轮若设备算子接口不接受其
  代表 shape，保留为结构证据，不用不相关运行失败否决 safe scope。
- `fused_matmul_relu_pass` 在 910B2 不注册、不解析 fused op，维持 device-gated
  not-applicable；不通过 monkeypatch 绕过 A5 gate。T-044 只关闭功能，batch/attention paired
  性能必须在通过后另登记；任何图门禁、alias 或数值失败先回到 P-010，不写 Triton 替身。

### E-138：T-043/T-044 wheel 与 NPU 功能关闭（2026-08-25）

- P-010 修改后 source/installed 完整 FX 均为 84/84。T-042 wheel 已归档为
  `artifacts/torch_npu_t042_before_t043_b2_composite_guard.whl`，SHA256
  `ea801e791373b0bd3adf9d4bfb6253ace75afa800c71b0451c9b206e4664fe5a`；新 wheel SHA256
  `44f2aad2465d59d6285fcd17739186a9560f90483dfa4e5de92948e848e461d8`，archive 1318 条、
  两个修改模块与 source byte-equal、不含 TorchAir/Tensorpipe/legacy egg-info，已按
  `--no-deps --force-reinstall` 安装。
- installed CPU/FX 复验中，batch default 仍 2/2→1/1 且 output-output alias false→false，
  step/dtype 两反例不再改写；attention safe output0 升级且生成 6-tuple meta，aux index6 和
  24 legacy args 保留，inference runner 调用次数从 2 降为 1。
- T-044 使用 NPU1；首个无 include-shim worker 复现已知 `ATen/ATen.h` 缺失并作为中性环境
  失败保留，随后统一使用登记的 C++20/site-packages-header audit shim。5/5 fresh worker、
  全部图门禁和输出合同通过、累计 mismatch 0：batch tuple/cat 正例 2/2→1/1，tuple 增加
  2 clone并保持独立 storage；step/dtype 负例保持 2/2；attention old/v3 1/0→0/1，前四
  tensor 输出 shape/dtype/stride/alias 与值完全一致。聚合证据为
  `results/t044_b2_composite_compile_20260825/aggregate/aggregate.json`。

### T-045：B2 batch-embedding 与 attention-v3 paired 性能登记

- 不再修改产品源码。使用同一 T-043 wheel、NPU1、default backend与统一 audit launcher，
  baseline wrapper 跳过单个目标 pass，candidate 执行当前 pass；每个 variant/round 均为独立
  fresh process/cache。三轮顺序固定 B-C/C-B/B-C，warmup 10、runs 100、memory 3/10；
  round1 两侧另做 warmup1/active10 profiler。
- batch 分两个代表 scope：`batch_default_clone` 使用同一 `(8192,128)` fp16 weight、
  `(1024,64)` int64 indices 的 4 个等长 slice+embedding+sum，四路结果最终相加为单 tensor
  以兼容统一 memory sampler，要求 4/4→1/1 且 candidate 有 4 个 alias-safe clone；
  `batch_cat_collapse` 在相同输入后把四个
  reduce 沿末维 cat，要求 4/4→1/1、cat 1→1 且无 clone。两路都逐轮复核 output-output
  alias、stride/dtype/shape与数值。
- attention 使用 BSH `(4,256,1024)` fp16、16 heads，只消费 legacy/v3 的 output0；baseline
  保留 legacy op，candidate 替换 v3。输出必须零 mismatch/NaN 分类一致，profile 记录 vendor
  task 数与 duration；不得把首次编译时间计入 steady verdict。
- 主 gate 沿用三轮 p50 中位数改善严格超过 10%，p99 不回退超过 5%；task/allocated peak
  明确下降可单独形成 resource-beneficial。未过 gate的安全 scope记 neutral；明显回退则另立
  guard 停用该 scope。vendor attention 优先保留 vendor op，不写 Triton；batch 已有设备
  kernel缺口前不得用手写 Triton替代图语义修复。

### E-139：T-045 三类 paired 性能结果（2026-08-25）

- 三类共 18 个有效 fresh worker 全部通过图门禁与测量前后输出合同；另保留最初使用
  `1e-3` 误判 fp16 sum/add rounding 的 `batch_default_clone/baseline_r1` 中性失败，重试改用
  标准 fp16 `rtol=atol=1e-2`，但仍严格检查 dtype/shape/stride、NaN 分类、输入别名和图结构。
- `batch_default_clone` 三轮中位数 P50 `0.528130→0.404005 ms`（改善 `23.50%`），P99
  `0.634600→0.664170 ms`（回退 `4.66%`，在 5% 门限内），每步设备任务 `9→3`；但
  additional allocated peak `17,041,920→18,873,856 B`（增加 `1,831,936 B`），首次编译
  `20.98→45.41 s`。按预注册完整 gate 为 `supported-neutral-resource-beneficial`，表示稳态
  延迟/任务数受益，但不能称为内存和编译均受益。
- `batch_cat_collapse` P50 `0.757215→0.424760 ms`（改善 `43.90%`），P99
  `0.915350→0.472880 ms`（改善 `48.34%`），每步任务 `13→3`；additional allocated peak
  `5,245,440→18,873,856 B`（增加 `13,628,416 B`），首次编译 `20.92→46.54 s`。同样因
  memory gate 未过记 `supported-neutral-resource-beneficial`，保留 pass 但明确其 trade-off。
- `fusion_attention_v3` 的旧/v3 两侧均为每步 1 个
  `aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore`，allocated peak 同为
  `49,286,144 B`；P50 `0.333150→0.349320 ms`（回退 `4.85%`），P99
  `0.416890→0.549120 ms`（回退 `31.72%`）。candidate P50 三轮均慢，P99 在 2/3 轮明显
  更差，判 `supported-performance-regressed`。这不是缺 kernel，故不写 Triton attention；
  进入 P-011，在非 A5（当前 910B2）禁用此接口替换，A5 保留待真机验证。

### T-046：fusion-attention-v3 B2 性能拒绝落地与 wheel 验证

- 按 P-011 仅增加非 A5 early-return 与对应 FX 测试；源码测试必须通过隔离引导同时证明
  `ascend_graph_pass.py` 来自 source、`_C` 来自已安装扩展。任何 autoload/schema 启动失败
  作为中性环境证据保留，不计入测试结果。
- 归档 P-010 wheel 后增量构建；archive 检查条目唯一、产品模块与 source byte-equal、核心
  动态库存在且不含 TorchAir/`libtensorpipe.so`/legacy egg-info。只允许
  `pip --no-deps --force-reinstall`，installed 完整 FX 通过后才进入 NPU。
- NPU1 fresh default/fullgraph worker 只验证当前 B2 门禁：legacy/v3 必须
  `1/0→1/0`，输出 value/dtype/shape/stride/alias 合同通过。它不是新一轮性能测试；T-045
  已给出关闭依据。完成后更新矩阵、报告、入门指南与公开文档仓库。

### E-140：T-046 wheel、安装态与 B2 关闭态验证（2026-08-25）

- P-010 wheel 已归档为 `artifacts/torch_npu_t043_before_t046_attention_b2_gate.whl`，SHA256
  `44f2aad2465d59d6285fcd17739186a9560f90483dfa4e5de92948e848e461d8`。P-011 新 wheel
  SHA256 为 `beee993d4c803ed72d26284dcdc06eac97cedaf450a54398ec11285d2711d54b`；archive 1318
  条且唯一，产品 pass/runner 与 source byte-equal，包含 `_C`/`libtorch_npu.so`，不含
  TorchAir、`libtensorpipe.so` 或 legacy egg-info，构建临时链接已清理。
- 新 wheel 已按 `--no-deps --force-reinstall` 安装；source 隔离引导与 installed 完整 FX
  均为 85/85。两次错误 source 启动分别因 backend autoload 抢先导入无 `_C` 源码包、以及
  stub/native schema 重复注册中止，原始日志保留且未进入测试结果。
- NPU1 fresh worker 使用 Ascend910B2/CANN 9.0.1/default/fullgraph/audit launcher，确认
  legacy/v3 `1/0→1/0`；四个输出 value/dtype/shape/stride/input alias/output alias 全通过，
  mismatch 合计 0，首次 compile+run `2774.42581 ms`。结束后 NPU1 无残留进程。
- 矩阵更新为 231 `not-run`、1 `not-applicable`、3 `unsupported`、7
  `supported-beneficial`、1 `conditional-supported-beneficial`、4 `supported-neutral`、2
  `supported-neutral-resource-beneficial`、2 `supported-pass-disabled-performance-rejected`。
  B2 27 条至此关闭；主线下一步是 B3 DVM/MLIR。

### T-047：B3 DVM/MLIR 八条 pass 的结构与环境路由登记

- 目标覆盖矩阵 B3 的 8 条：DVM `dvm_graph_fusion` 及 5 个 `fx_pass.py` 子 pass，MLIR
  `DvmMlirPostGradPass` 与 `fold_sum_cast_to_dtype`。本任务先只读源码、现有测试和运行依赖，
  新增 audit-only CPU/FX 结构脚本与报告；不修改 torch_npu/PyTorch/Triton 产品源码。
- 每条先明确调用者、输入 IR、返回/原地语义、正例、负例、symbolic/dtype/alias 风险；聚合
  pass 与子 pass 分开记，不能用 `DvmMlirPostGradPass` 调用一次冒充其内部变换都命中。
- 环境 Gate 0 分三路：纯 FX 可直接验证；DVM backend 需要 `torch.compile(backend="dvm")`
  或对应 loader 能导入；MLIR 需要 `ascend_npu_ir`/torch-mlir 编译依赖完整。缺依赖记
  environment-blocked，不把 import skip 当 pass，也不为测试修改共享环境。
- 结构层至少覆盖：mm transpose flag 正/负、K=1 matmul 与 K≠1/symbolic、fp16/bf16 sum
  cast 与 fp32 negative、混合 dtype promotion 与同 dtype negative、expand-as-reshape 与真广播、
  sum+cast 单用户与多用户/dtype negative。后端层只有在结构通过后才从 NPU1 fresh process
  验证生成 kernel/fallback 和数值；性能仍另登记，当前不提出 Triton 替身。

### E-141：T-047 安装态 CPU/FakeTensor 首轮证据（2026-08-25）

- `results/t047_b3_dvm_mlir_static_20260825/result.json` 使用当前 source-built wheel 的安装态
  模块完成首轮结构审计。transpose flag、K=1 mm/bmm→mul、K≠1/动态 K 保持、混合 dtype
  promotion、同 dtype 保持、DVM add→mul 子图融合、MLIR sum-cast fold 及 wrapper 链式调用均按
  预期；已执行样例的 value/dtype/shape/stride 合同通过。
- `insert_sum_fp32_prepost_cast_prims` 的直接调用域存在确定反例：int64
  `[16_777_217, 1]` 经 FP32 累加后与 eager 相差 2；float64 显式 sum 与 eager 相差 1。当前
  `GraphFusionPartitioner` 会先由 DVM dtype rule 排除这两类输入，所以不是已证实的现网 aggregate
  逃逸；但子 pass 本身没有同等 guard，独立调用和未来调用者不安全。
- `expand_to_reshape` 的 identity expand 会把 graph target 改成 reshape，但函数返回 `None`，也
  不 lint/recompile；真实 broadcast 正确保持。当前 aggregate 的后续 graph/codegen 流程会直接读
  graph，尚未观察到数值错误，但该函数声明与其他子 pass 的返回合同不一致，独立调用会看到 stale
  `GraphModule.code`。
- 当前环境没有 `torch_mlir`，因此纯 FX 的 `fold_sum_cast_to_dtype` 和 wrapper 可测，但完整 MLIR
  backend 必须记 environment-blocked，不能以结构通过冒充设备编译通过。DVM backend 不沿用这个
  torch-mlir Gate，后续仍可在 NPU1 单独验证。

### E-142：P-012 wheel 与 B3 DVM 功能闭环（2026-08-26）

- P-011 wheel 已归档为 `artifacts/torch_npu_t046_before_t047_p012.whl`，SHA256
  `beee993d4c803ed72d26284dcdc06eac97cedaf450a54398ec11285d2711d54b`。P-012 新 wheel
  SHA256 为 `61b0031cbb027548f60745dcf0a2484503a360347dec6bd3cc2f3f2bc823ebca`；1318
  条 archive entry 且无重复，source `fx_pass.py`/既有 B2 pass 与 wheel byte-equal，包含
  `_C`/`libtorch_npu.so`，不含 TorchAir、`libtensorpipe.so` 或 legacy egg-info。
- 新 wheel 已按 `--no-deps --force-reinstall` 安装。source/installed P-012 目标测试均为 6/6；
  source 隔离与 installed T-047 全量静态审计通过，int64/float64 反例 mismatch 均由 2/1
  恢复为 0，identity expand 已返回同一 GM 且自动重编译。
- NPU1 fresh installed `DvmGraphFusionPatch` 测试为 15/15，DVM backend/MLIR scheduler
  测试为 32/32，覆盖静态/动态、fp16/bf16/fp32、K=1→mul、matmul、reduce、promotion、copy、
  backward 与 NPUGraph。第一次 DVM backend 启动因显式关闭 autoload 且测试未 import
  `torch_npu`，32 条均在设备注册前失败，作为中性启动证据保留；autoload retry 才是有效结果。
- DVM backend 在没有 `torch_mlir` 时仍能使用自身 codegen 并通过 32/32；但
  `TORCHINDUCTOR_NPU_BACKEND=mlir` 的 loader 明确要求 `torch_mlir`，完整 MLIR backend 仍为
  environment-blocked。NPU1 两次有效测试结束后均无残留进程。
- reachability 反例确认 `expand_to_reshape` 当前不进入 aggregate：add→identity-expand、
  broadcast-expand→add 和 add→view 的 outer graph 都把 expand/view 留在 DVM custom op 外，
  因 `GRAPH_FUSION_SUPPORT_OP` 未列 expand/reshape/view。该子 pass 的直接合同已修复，但 aggregate
  收益必须记 `not-triggered-by-current-partitioner`，不能拿直接调用命中冒充产品触发。

### T-048：B3 DVM graph-fusion 首轮隔离性能筛选

- 只新增 audit-only worker/aggregator，不改产品源码。NPU1 空闲时使用 fresh process/cache、
  default/fullgraph、同一 source-built wheel；每个 cohort 运行 baseline/candidate 各 3 轮，顺序
  `B1,C1,C2,B2,B3,C3`，每轮 warmup 10、runs 100，并各保留一轮 10-step profiler、首次编译和
  allocated/reserved peak。
- `aggregate_basic` 比较 default Inductor 与 `DvmGraphFusionPatch`；它衡量 aggregate 方案，不把
  收益拆给所有子 pass。`k1_decompose` 两侧都启用 graph fusion，仅 baseline 在进程内跳过
  `decompose_k1_matmul_to_mul`；`sum_fp32_cast` 同理，仅 baseline 跳过
  `insert_sum_fp32_prepost_cast_prims`，从而隔离子 pass 增量。
- 每个 worker 必须记录 outer DVM custom-op gate 和子图 target before/after，且 value/dtype/shape/
  stride 在测量前后通过。P50 改善须大于 10%、P99 回退不超过 5%、allocated peak 不增加才记
  beneficial；task 减少但三门未全过只记 resource-beneficial。promotion/transpose/fold 先保留功能
  结论，未找到可稳定隔离的 paired cohort 前不伪造单 pass 性能；expand 明确不可达。

### E-143：T-048 aggregate 收益与子 pass 不可隔离证据（2026-08-26）

- `aggregate_basic` 六个 fresh worker 全部图/语义 gate 通过。default→DVM 的三轮中位 P50
  `0.279905→0.212175 ms`（改善 24.20%），P99 `0.374470→0.224960 ms`（改善
  39.93%），mean `0.367714→0.213482 ms`，首次 compile+run
  `20,319.58→2,807.76 ms`，additional allocated peak 均为 2,560 B。三门全部通过，
  `dvm_graph_fusion` 记 `supported-beneficial`。
- profiler 中 default 每步 1 个 `triton_per_fused_add_mul_sum_0`，10 步 device duration
  205.46 us；DVM 每步 3 个 task（两个 `DvmAddBroadcastMulCastReduce`、一个 `DvmCastAdd`），
  总 duration 109.02 us。task 数增加但总设备时间和端到端延迟均改善，不能按 kernel 数单项否决。
- K=1 isolate 的 before 快照在该 torch_npu subpass 之前已经是 `cast→mul→cast`；baseline 跳过
  `decompose_k1_matmul_to_mul` 后不变，原 gate 失败是审计假设错误，不是产品失败。该 pass 的
  直接 FX 合同通过，但当前 compile 增量为零，矩阵记 `not-applicable/upstream-predecomposed`。
- sum isolate candidate 的 `cast(fp32)→sum→cast(fp16)` 通过；人工跳过 pass 后生成 bare fp16
  DVM sum，并在首次 native 执行 segfault。已按图模式要求读取 transformed FX 与
  `output_code.py`，确认 outer custom op 和 `k.sum(...); k.store(..., dvm.float16)` 已生成。这个
  baseline 不可用，停止 paired 性能，sum pass 记 availability-required 的 `supported-neutral`。
- B3 八条最终为 1 beneficial、5 neutral、1 not-applicable、1 unsupported；总矩阵更新为 223
  `not-run`、2 `not-applicable`、4 `unsupported`、8 `supported-beneficial`、1
  `conditional-supported-beneficial`、9 `supported-neutral`、2
  `supported-neutral-resource-beneficial`、2 `supported-pass-disabled-performance-rejected`。

### T-049：B4 attention 入口、代表 pattern 与 scale 语义审计登记

- 先只新增 audit-only 脚本和结果，不改 PyTorch/torch_npu/Triton 产品源码。所有运行仍从
  `/home/z50063656/tmp` 启动，使用当前 P-012 source-built wheel；真实 NPU 只选择采样时空闲卡。
- 第一道 gate 审计第 31 条 `npu_fusion_attention_graph`：同一确定性 fp16 输入分别调用
  `torch.ops.npu_graph.npu_fa` 的 positional scale、keyword scale 和底层
  `torch_npu.npu_fusion_attention(scale=...)`，记录输出 value/dtype/shape 与三方最大误差。
  positional/keyword 是同一 schema 的两种等价调用方式，结果必须一致；只检查 shape 不足以
  证明 scale 合同。另记录 scale=0、默认 scale 和非 1 scale 的异常/有限性边界。
- 源码已显示 `npu_fa(*args, **kwargs)` 只在 `len(args)>8` 时倒数 `args[8]`，keyword `scale`
  原样下传；其 `IndexError` 分支在此前置条件下不可达。底层官方本地文档又明确 `scale` 是直接
  乘到 QK 的系数。动态对照前只登记为高风险，不预判应该统一为倒数还是原样。
- 第二道 gate 盘点上游 30 个 `_sfdp_pattern_*` 的注册设备、inference/training 变体、extra
  check 与 NPU replacement 路径；先选 1、5、13、18、21、28、29 七个 family smoke。必须同时
  观察 `fuse_attention` counter、变换后 SDPA 节点、generated NPU attention kernel 和数值，
  才能写“触发且可用”。未触发、fallback 与设备 kernel 缺口分开记录。
- 代表 smoke 若融合 attention 之外还生成 mask/cast/clone 等 Triton 辅助 kernel，无 shim fresh
  launcher 可能复现 E-050/E-071 的 editable PyTorch header 缺口。原始失败必须保留并先读
  transformed FX/output code；允许在新 cache 中复用 T-022 已登记的 audit-only C++20/CANN header
  wrapper，只用于区分“pass/vendor kernel 不可用”和“共享环境 launcher 合同”，不得据此宣称
  当前环境无 shim 可用或发布性能结论。
- 只有 scale 对照确认真实语义缺陷，才另建产品提案，范围优先限制到
  `npu_fusion_attention_graph.py` 与目标测试。功能闭环前不做 attention 性能，也不手写 Triton；
  现有 vendor attention 已存在时，Triton 只在 vendor path 不可用且能证明端到端收益后考虑。

### T-050：B4 pattern 1 单 pass paired 性能登记

- T-049 已证明 `_sfdp_pattern_1_half_inference` 的 matcher/总 counter 均为 1，生成代码为
  `npu_fusion_attention_v3`，Profiler 为 `aclnnFlashAttentionScore`，数值误差
  `0.0009765625`。它是七个代表 family 中首个无需辅助 Triton 就在当前 wheel 完整通过的
  无 mask 基础路径，因此先做第一个可归因 paired benchmark。
- baseline 与 candidate 使用同一 default backend、同一 `(4,8,128,64)` fp16 静态 inference
  图、同一输入和相同 fresh-process/cache。首次 isolate 只关闭 pattern 1 后，pattern 3 的
  inference variant 接管同一无 dropout 图；该次数字不计入性能。因此正式 baseline 同时把
  `pattern_name` 为 `_sfdp_pattern_1_half_inference` 或 `_sfdp_pattern_3_half_inference` 的 entry
  `extra_check` 临时改为 false；收益只归因给 1/3 等价 inference family，不伪分到单条。
  candidate 保持产品默认。worker 退出即回滚，不改 PyTorch/torch_npu/Triton 源码。
- 两侧各 3 个 fresh worker，顺序 `B1,C1,C2,B2,B3,C3`，warmup 10、runs 100；记录首次
  compile+run、mean/stdev/P50/P99、allocated/reserved peak、exact pattern/总 counter、generated
  code、10-step profiler kernel/task。数值、shape、dtype、finite 必须先通过。
- baseline 展开的 BMM/softmax 需要 Triton launcher，因此两侧统一使用 T-022 audit-only
  C++20/CANN header wrapper与 installed PyTorch headers，避免把环境差异混进 A/B；性能结论仅属于
  当前开发态合同，正式无 shim 环境仍单列。beneficial gate 为 P50 改善大于 10%、P99 不回退
  超过 5%、additional allocated peak 不增加；否则按 neutral/resource/rejected 分层。

### E-144：T-049 attention scale 与七个代表 family 功能证据（2026-08-26）

- `npu_fa` 的 positional/keyword scale 由 dispatcher 按 schema 规范化，0.5/1/2 三组输出完全
  相同。scale 0.5/2 与底层 vendor 同 scale 最大误差 `0.1217041015625`，与 vendor inverse-scale
  误差为 0；结合历史 pattern 的 `.div(inv_scale_factor)`，确认 wrapper 参数沿用 legacy divisor
  合同，不是 keyword 逃逸 bug。scale=0 两种调用均抛 `ZeroDivisionError`，暂记无效除数边界。
- pattern 1/5/13/18/21/28/29 七个 family 的目标 matcher 与 `fuse_attention` counter 均为 1，
  value/structure 合同通过，最大绝对误差依次为 0.0009765625、0.0009765625、0.002197265625、
  0.001953125、0.001953125、0.001220703125、0.0009765625。
- pattern 1/13 最终为 vendor FlashAttention；18/28 为辅助 Triton + vendor attention；5/21/29
  被 NPU dispatcher 重新展开为 BMM + Triton。matcher 命中不能冒充 vendor kernel 落地，后六条
  未跑 paired 性能前保持 `not-run` verdict，只回填功能/codegen 字段。
- pattern 5/18/21/28/29 无 shim 初跑在辅助 Triton launcher 因 editable PyTorch include view 缺
  `ATen/ATen.h` 失败；失败产物保留。T-022 audit-only wrapper 有效重跑用于归因，不等于正式环境
  原生可用。第 31 条 wrapper 只完成 scale forward，backward/dynamic/performance 仍未关闭。

### E-145：T-050 pattern 1 等价 inference family 性能闭环（2026-08-26）

- 首次只关闭 pattern 1 后由 pattern 3 接管相同图，该 smoke 不计入性能。正式 baseline 同时关闭
  两个等价 inference entry；三轮中 pattern/fusion counter 均为 0、生成 2 BMM + 2 Triton，
  candidate counter 均为 1、生成单个 `aclnnFlashAttentionScore`，A/B 隔离成立。
- 六个 fresh worker 按 `B1,C1,C2,B2,B3,C3`、warmup 10/runs 100 全部正确，最大绝对误差
  `0.0009765625`。三轮中位 P50 `0.581895→0.310165 ms`（改善 46.70%），P99
  `0.601770→0.335440 ms`（改善 44.26%），mean 改善 46.41%，首次 compile+run 改善 95.72%。
- additional allocated peak `204,477,440→25,955,328 B`（减少 87.31%），reserved peak
  `224,395,264→27,262,976 B`（减少 87.85%），device task/step 4→1。预登记三门全部通过。
- 最终只把已直接覆盖的 `_sfdp_pattern_1` fp16 静态 inference 记为 `supported-beneficial`；关闭
  pattern 3 是防止等价 matcher 接管的隔离手段，不能据此关闭其 training/dropout 覆盖。B4 首条
  最终 verdict 使总矩阵变为 222 `not-run`、9 `supported-beneficial`，其余分类不变。证据见
  `report/t049_t050_b4_attention_first_20260826.md` 与 T-050 aggregate JSON。

### T-051：B4 pattern 13 三维 BMM attention 配对性能登记

- 只新增 audit-only worker/aggregate 和结果，不改 PyTorch/torch_npu/Triton 产品源码。pattern 13
  是 inference-only 三维 BMM family；T-049 已在 `(4,8,16)` fp16 上 exact counter=1、数值通过，
  且无辅助 Triton 的 candidate 最终为 `aclnnFlashAttentionScore`。
- 性能 cohort 使用 `(32,128,64)` fp16 Q/K/V，使三维首轴对应 4×8 个 attention head，与 T-050
  的计算规模可比较但仍只归因 pattern 13。baseline 仅把
  `_sfdp_pattern_13_half_inference` entry 的 `extra_check` 临时置 false；先做短 smoke，必须满足
  baseline fuse/target counter=0 且生成 BMM+Triton、candidate=1 且生成 vendor attention。若其他
  pattern 接管，则仿照 T-050 扩大到已证实的等价 entry，并在正式测试前修订登记。
- 隔离通过后执行 baseline/candidate 各 3 个 fresh worker，顺序 `B1,C1,C2,B2,B3,C3`，warmup
  10、runs 100。两侧统一使用 T-022 audit-only launcher/header wrapper，因为 pass-off softmax
  会生成 Triton；结果只属于 development/audit-shim 合同。
- 正确性先检查 value/dtype/shape/finite，fp16 容差 `atol=rtol=0.02`；再记录 exact/总 counter、
  generated code、10-step task profile、首次 compile+run、P50/P99/mean/stdev 和 allocated/reserved
  peak。beneficial gate 仍为 P50 改善 >10%、P99 回退不超过 5%、allocated peak 不增加。

### E-146：T-051 pattern 13 资源有益/时延中性闭环（2026-08-26）

- 隔离 smoke 与正式六轮都确认 baseline 只关闭 1 个 pattern 13 entry，exact/总 fusion counter
  为 0、生成 2 BMM + 1 Triton softmax；candidate counter 为 1、生成单个
  `aclnnFlashAttentionScore`。没有等价 matcher 接管。
- 六轮 value/dtype/shape/finite 合同通过；baseline/candidate 最大绝对误差分别为
  `0.001953125/0.01318359375`，均满足 fp16 `atol=rtol=0.02`。NPU1 测试前后无进程。
- 三轮中位 P50 `0.335175→0.331850 ms`（改善 0.99%），P99
  `0.363240→0.363420 ms`（回退 0.05%），mean 改善 1.37%；未过 10% latency 主门槛。
- 首次 compile+run `34,134.41→2,944.29 ms`（改善 91.37%），additional allocated peak
  `204,472,832→25,955,328 B`（减少 87.31%），reserved peak 减少 87.85%，task/step 3→1。
  最终记 `supported-neutral-resource-beneficial`，不把资源收益冒充 latency beneficial。
- 第一次正式批次启动把 `set -e` 放在共享 `env.sh` 之前而立即退出，未创建 worker/占用 NPU；
  作为中性命令错误保留说明，不计产品失败。矩阵变为 221 `not-run`、3
  `supported-neutral-resource-beneficial`，其余分类不变。详细证据见
  `report/t051_b4_attention_pattern13_performance_20260826.md`。

### T-052：B4 float-mask re-expansion 根因与 pattern 30 对照登记

- 先只读 PyTorch replacement、torch_npu op-plugin SDPA dispatcher 和 T-049 generated code，不改
  产品源码。目标是解释 pattern 5/21/29 为何 exact matcher=1 后仍展开为 BMM + Triton。
- 源码 gate 必须区分无 mask/bool mask 与任意 additive float mask。若 vendor branch 明确只接受
  bool/None，而三个 replacement 都把 additive mask 转成 query dtype，则 math fallback 是能力域
  选择，不得误报为 matcher 或 lowering 缺失，也不能把一般 float bias 无条件转 bool。
- 增加 audit-only pattern 30 无 mask `_safe_softmax` 对照；其结构与 pattern 29 接近但没有 mask。
  在 NPU1 fresh process 要求 exact/总 counter=1、value/dtype/shape 通过，并观察最终是 vendor、
  vendor+辅助 kernel 还是 math。若无 mask 能走 vendor，则进一步固定 float mask 是主分流因素。
- 本任务只形成 capability/root-cause 证据。任意产品提案必须先证明 vendor 的 `pse` 或其他参数
  能严格表达一般 additive mask，并完成正负/广播/Inf 精度与 paired 性能；否则保持安全 math
  fallback，不以手写完整 attention 作为默认替代。

### E-147：T-052 float-mask dispatcher 根因闭环（2026-08-26）

- PyTorch pattern 5/21/29 replacement 都把 additive mask 转成 query 浮点 dtype；torch_npu
  `ScaledDotProductAttentionKernelNpuOpApi.cpp` 的 FlashAttention 与 FusedInferAttention 两条
  vendor 分支均只接受 bool mask 或 None，不满足后明确调用
  `_scaled_dot_product_attention_math`。T-049 的 BMM+Triton 是安全 capability fallback。
- 一般 float bias 不能无条件转 bool：有限加性偏置、0/1、`-inf` 和连续 position bias 的语义
  不同。当前没有证据证明 vendor `pse` 可完整承接，因此不提出产品 gate 修改或完整 Triton
  attention 替身。
- pattern 30 无 mask fresh-cache 对照 exact/总 counter=1，最大误差 `0.0009765625`，generated
  code/profiler 为单个 `npu_fusion_attention_v3`/`aclnnFlashAttentionScore`，进一步固定 mask
  dtype 是 29/30 代表输入的主分流因素。矩阵只回填 pattern 30 功能/codegen，性能未测且 verdict
  仍 `not-run`。
- pattern 30 第一次未设 fresh cache、第二次只开 debug 仍命中已有 cache，counter 证据无效；
  第三次独立 Inductor/Triton cache 才纳入结论。前两次作为中性审计方法记录保留。详细报告见
  `report/t052_b4_attention_float_mask_dispatch_20260826.md`。

### T-053：B4 pattern 5 float-mask math fallback 配对性能登记

- 只新增 audit-only worker、aggregate、结果与文档，不修改 PyTorch、torch_npu、Triton 或
  op-plugin 产品源码。目标是比较原始 pattern 5 子图与“matcher 命中、SDPA replacement 随后因
  additive float mask 被 dispatcher 安全分解为 math”的最终执行成本；本任务不尝试把一般
  float mask 转 bool，也不手写完整 attention。
- cohort 固定为 `(4,8,128,64)` fp16 Q/K/V 与 `(1,1,128,128)` fp16 additive mask，静态
  inference。baseline 仅把 `_sfdp_pattern_5_half_inference` entry 的 `extra_check` 临时置
  false，candidate 保持产品默认。先各跑一个 fresh-cache smoke，要求 baseline exact/总 fusion
  counter 为 0、candidate exact/总 counter 为 1；若其他 attention pattern 接管，必须先修订
  等价 isolate 范围，不能把未隔离数据纳入性能。
- 两侧都预期最终为 BMM + Triton math，而不是 vendor FlashAttention。worker 必须记录 generated
  code、10-step profiler task、首次 compile+run、P50/P99/mean/stdev、allocated/reserved peak，
  并先通过 value/dtype/shape/finite 与 fp16 `atol=rtol=0.02`。两侧统一使用 T-022 audit-only
  C++20/CANN header wrapper和 installed PyTorch headers，结论限于 development/audit-shim 合同。
- 隔离 smoke 通过后按 `B1,C1,C2,B2,B3,C3` 跑各 3 个 fresh worker，warmup 10、runs 100。
  beneficial gate 沿用 P50 改善 >10%、P99 回退不超过 5%、additional allocated peak 不增加；
  若 candidate 回退超过 5%，记 `supported-performance-regressed`，并把该 family 标为优先优化
  候选，但不得仅凭性能回退扩大不安全的 vendor mask gate。

### E-148：T-053 pattern 5 性能回退闭环（2026-08-26）

- 短 smoke 与正式六轮都确认 baseline 只关闭一个 pattern 5 half-inference entry，exact/总
  fusion counter 为 0；candidate exact/总 counter 为 1。两侧数值、shape、dtype、finite
  合同通过，最大绝对误差均为 `0.0029296875`，最终都没有 vendor attention。
- baseline 生成 `2 BMM + 1 Triton`，candidate 经 SDPA math re-expansion 生成
  `2 BMM + 6 Triton`。三轮中位 P50 `0.381385→0.775105 ms`（回退 103.23%），P99
  `0.409750→0.824400 ms`（回退 101.20%），首次 compile+run 回退 140.82%，task/step
  `3→8`，additional allocated peak 增加 `1,054,720 B`。三个预登记门槛全部失败。
- 最终记 `supported-performance-regressed`。优先方向是避免在 NPU float-mask capability 外做
  无收益 rewrite，或未来为严格可证明子域接 vendor；不把任意 float bias 转 bool，也不手写
  完整 attention 复制已经更快的原图 math 路径。
- 第一次 smoke 因审计脚本 final mkdir 与 debug 目录先创建冲突而未写 result；candidate 未启动。
  失败目录保留，修正只影响 audit 落盘，retry 与正式结果有效。详细证据见
  `report/t053_b4_attention_pattern5_performance_20260826.md`。

### T-054：P-013 installed wheel 功能与配对性能验收登记

- P-013 源码只按已登记范围增加 NPU exact-entry guard、loader 调用和独立无设备测试；2/2
  测试已通过。旧 P-012 wheel 已保存为
  `artifacts/torch_npu_t053_before_p013.whl`，SHA256 与登记值一致。第一次 build 命令因在
  source `env.sh` 前设置 `set -e` 而退出，未执行构建/覆盖 wheel；有效 retry 后的新 wheel
  SHA256 为 `3909fd649d777b8dfd393342da0ff2b88c5cce2ef219f0d103d063af4c2d4989`，已
  `--no-deps --force-reinstall` 安装。
- 新增 audit-only installed worker/aggregate。C 侧使用新 wheel 默认 guard；B 侧在 fresh process
  中只为 exact entry 恢复 guard 保存的原 `extra_check`，重建旧 rewrite。两侧同一新 wheel、同一
  `(4,8,128,64)` fp16 Q/K/V 与 `(1,1,128,128)` fp16 additive mask、相同 wrapper/header。
- 先各做 fresh smoke：C 必须 exact/总 counter 0、`2 BMM + 1 Triton`；B 必须 counter 1、
  `2 BMM + 6 Triton`；两侧均不得出现 vendor attention，数值/dtype/shape/finite 先通过。若
  wrapper 原 check 不可恢复或其他 pattern 接管，立即停止正式性能并回滚 P-013。
- 隔离通过后顺序 `B1,C1,C2,B2,B3,C3`，各 3 个 fresh worker、warmup 10/runs 100；记录首次
  compile、P50/P99/mean/stdev、allocated/reserved peak 和 10-step task。C 相对 B 必须 P50
  改善 >10%、P99 不回退超过 5%、allocated peak 不增加且 task 8→3，才能把 P-013 标成
  verified。测试后还需回归 pattern 1 vendor-beneficial、pattern 13 vendor-resource-beneficial
  和 pattern 21 仍可触发，证明 exact scope 没有旁路其他 family。

### E-149：P-013 pattern 5 NPU 性能门禁验证完成（2026-08-26）

- 新 wheel installed smoke 中，默认 guard exact/总 counter 为 0、生成 `2 BMM + 1 Triton`；
  测试侧恢复 wrapper 保存的原 pattern generator 后 counter 为 1、恢复 `2 BMM + 6 Triton`。
  六个正式 worker 的 shape/dtype/finite/容差全部通过，最大误差 `0.0029296875`。
- 旧 rewrite→默认 guard 的三轮中位 P50 `0.745200→0.370545 ms`（改善 50.28%），P99
  `0.767770→0.581510 ms`（改善 24.26%），mean 改善 48.79%，首次 compile+run 改善
  57.81%；task `8→3`，additional allocated peak 减少 `1,054,720 B`，reserved 不变。
  P-013 四项验收门槛全部通过。
- pattern 1/13/21 邻近 fresh regression 的 exact/总 counter 均为 1，数值通过；1/13 保持
  vendor FlashAttention，21 保持原 math fallback。最初邻近命令误把 case 1 传给不含该 choice
  的代表脚本，在 argparse 阶段退出、未编译；随后用独立 pattern 1 脚本有效通过。
- P-013 状态升级为 `verified-pass-disabled-performance-rejected`：guard 本身为
  `supported-beneficial` 修复，但被守护的 pattern 5 rewrite 仍是性能负优化，矩阵按最终产品
  行为记 `supported-pass-disabled-performance-rejected`。详细证据见
  `report/t054_b4_attention_pattern5_guard_20260826.md`。

## 提案记录

### P-013：pattern 5 NPU half-inference 性能负域门禁

- 状态：`verified-pass-disabled-performance-rejected`。目标只覆盖 T-053 已验证的
  `_sfdp_pattern_5_half_inference` + NPU default Triton backend；CPU/CUDA/XPU、fp32、training、
  pattern 6/21/29 与其他 attention family 全部保持。修改前回滚边界为当前 P-012 wheel
  SHA256 `61b0031cbb027548f60745dcf0a2484503a360347dec6bd3cc2f3f2bc823ebca`。
- 候选修改仅限 `torch_npu/_inductor/fx_passes/joint_graph.py`、backend loader 的注册调用和一个
  独立目标测试文件：在 PyTorch attention lazy pattern 生成前，为上述唯一 entry 包装 NPU-only
  `extra_check=False`，使它保持原图。包装必须幂等并保存原 check 供测试/回滚；不得修改 PyTorch
  通用源码、op-plugin SDPA dispatcher、算子 schema、C++ 或 Triton kernel。
- 先做 device-independent 单元测试：NPU + exact key 被关闭，非 NPU 和其他 key 保持原 check，
  重复 patch 不叠加。再重建 torch_npu wheel、`--no-deps` 安装，运行 pattern 5 fresh NPU
  integration，要求默认 exact/总 counter 变为 0、generated code 恢复 `2 BMM + 1 Triton`、
  数值合同通过；测试侧临时恢复保存的原 check 应再次得到 counter 1 与 `2 BMM + 6 Triton`。
- 新 wheel 下仍按 `B1,C1,C2,B2,B3,C3` 做三轮 paired；这里 B 表示恢复旧 rewrite，C 表示新
  guard。接受门槛为 guard 相对旧 rewrite 的 P50 改善 >10%、P99 不回退超过 5%、allocated
  peak 不增加，同时 task 恢复 8→3。任一调用顺序、cache 或其他 family 被改变则回滚。
- 该 guard 是性能域收缩，不把 pattern 标成“不可用”。只有 21/29 各自完成 paired 并证明同类
  回退后，才可另行登记扩大 family 范围；禁止从 pattern 5 直接外推。

### P-012：DVM sum/expand 子 pass 自包含能力门禁

- 状态：`verified-sum-safe-expand-direct-contract-aggregate-expand-unreachable`。仅修改
  `torch_npu/_inductor/dvm/fx_pass.py` 与对应 Python 测试；不改 schema、lowering、C++、DVM
  kernel 或 Triton。修改前回滚边界为 P-011 wheel SHA256
  `beee993d4c803ed72d26284dcdc06eac97cedaf450a54398ec11285d2711d54b`。
- sum pass 只允许 DVM 支持的低精度浮点输入进入 FP32 pre/post-cast；fp32 保持，整数、bool、
  float64 及不支持的显式 dtype 原图保持。正例仍覆盖 fp16/bf16 default 和显式 fp32，反例必须把
  E-141 的 int64/float64 mismatch 恢复为 0，且不能依赖 aggregate partitioner 替子 pass 兜底。
- expand pass 对缺失/非 Tensor meta 保守保持；只有 input/output shape 完全相同时改成 reshape，
  发生修改后 lint/recompile，并始终返回原 `GraphModule`。真 broadcast 不改，identity 正例执行
  value/dtype/shape/stride 等价并确认 generated code 已同步。
- 先跑目标 source 测试与 T-047 source 隔离审计，再重建 source wheel、`--no-deps` 安装并复跑
  installed 审计。NPU DVM aggregate 功能通过前不做性能；这两个修复是可用度/防逃逸修复，不宣称
  性能提升，也不需要手写 Triton。

### P-011：fusion-attention-v3 非 A5 性能拒绝门禁

- 状态：`verified-b2-disabled-a5-awaits-hardware`。只修改
  `ascend_graph_pass.py` 与 `test/_inductor/test_dynamic_shape_fx_passes.py`；不改 schema、
  lowering、C++/Triton kernel 或其他 pass。回滚边界为 P-010 wheel SHA256
  `44f2aad2465d59d6285fcd17739186a9560f90483dfa4e5de92948e848e461d8`。
- 在 `fusion_attention_v3_pass` 入口增加 `is_ascend950` capability gate：非 A5 直接保持 legacy
  op；A5 仍执行 P-010 的参数、输出用户和 fresh-meta 安全检查。测试必须分别固定 B2 no-op、
  强制 A5 safe-positive 仍升级、A5 aux/full-args negative 仍保持。
- 修改后运行完整 source FX，重建并 `--no-deps` 安装 wheel，再跑完整 installed FX；随后在
  NPU1 运行 B2 compile no-op，要求 old/v3 1/0→1/0、输出合同通过。最终 B2 verdict 记为
  `supported-pass-disabled-performance-rejected`；A5 只能记 `not-run/device-gated`，不得从
  910B2 推断 A5 性能。

### P-010：batch-embedding 与 fusion-attention-v3 保守语义域修复

- 状态：`verified-batch-resource-beneficial-attention-b2-superseded-by-p011`。只修改
  `ascend_graph_pass.py`、`ascend_custom_passes/__init__.py` 与
  `test/_inductor/test_dynamic_shape_fx_passes.py`；不改 PyTorch/Triton/C++/算子 schema，
  不新增 kernel。修改前 wheel 回滚边界为当前 T-042 SHA256
  `ea801e791373b0bd3adf9d4bfb6253ace75afa800c71b0451c9b206e4664fe5a`。
- batch-embedding 仅接受静态非负 slice dim、`step==1` 和默认 reduce dtype/options；任何
  无法证明项保持原图。cat-collapse 成功时沿用单 combined reduce；未 collapse 时每个
  select 后插入 contiguous clone，恢复各 reduce 输出的独立 storage 与 contiguous layout。
  结构测试要求 default positive 2/2→1/1 且数值/dtype/shape/alias 全同，step/dtype 反例保持
  2/2，cat-collapse 仍命中。
- attention 只在旧节点的参数可被 v3 schema 原样承接、actual-seq 为 None、未提供旧专属
  `dropout_mask/seed/offset`，并且全部用户都是索引 0–3 的 getitem 时改写；直接 tuple、索引
  4–6、超出 v3 的参数或未知 kwargs 一律保持旧 op。新节点必须用 fake op 重新生成 6-tuple
  meta，不能复制旧 7-tuple meta；meta 生成失败则回滚新节点并保持旧图。
- PRE runner 调整为 inference 时执行全体 PRE 一次、training 时只补执行 attention pass，
  不再在 inference 重复。测试用 registry spy 固定 1 次调用，并覆盖安全 output0、aux index6
  与 full legacy args 正负例。
- 修改后先跑完整 source FX 文件；再从源码重建 wheel、归档 T-042 wheel并 `--no-deps`
  安装，运行 installed FX。随后才登记 NPU：batch default/cat/step/dtype/alias，attention
  legacy/v3 首四输出数值与负例保留，fused-matmul 仅验证 B2 gate。功能通过前不做性能，
  batch non-collapse clone 若性能差可继续缩窄到 cat-collapse，不得删除 clone 换取收益。

### P-006：cat-slice-cat 与 pad-slice alias 可观察性保护

- 状态：`verified-supported-beneficial`。已只修改
  `ascend_graph_pass.py` 和目标 FX 测试，不改 PyTorch/Triton/C++，不加入 clone 或手写
  Triton；60/60 FX 测试、6/6 NPU 功能 worker 与 12/12 paired 性能 worker 通过。两条
  pass 在各自登记 safe cohort 均为 `supported-beneficial`。回滚边界是本提案新增的两个
  保守 guard。

### P-007：dtype/index/mask 可观察语义保护

- 状态：`verified-two-beneficial-one-neutral-development-audit-shim`。只修改
  `ascend_graph_pass.py` 与 `test_dynamic_shape_fx_passes.py`；不修改 PyTorch/Triton/C++，
  不新增设备 kernel。回滚边界是三处 capability 缩窄和对应测试。
- `dtype_optimal_pass`：arange int64 降级除了端点可表示，还要求所有直接用户均为明确返回
  bool 的比较；output、view、算术和未知 user 保持 int64。`.to(int64)` 不再接受 float32
  来源，只允许本来值域保证落在 int32 的 int32/bool/int16/int8，并同样要求所有直接用户
  为比较闭包。这样优化中间比较输入而不改变可观察 dtype，也不截断大 float。
- `broadcast_const_mask_compress`：只有 cond shape 与 where 输出 shape 可静态证明完全相同
  时才删除 where/full；存在广播、symbolic 无法证明或直接较小 mask 时保持原图。先不实现
  多层“删除后由下游重新广播”的激进证明。
- `fold_iota_arithmetic_pass`：保留已有常量 CSE 与带 transparent/closing closure 的 iota
  downcast；停用无范围证明的 `cmp(sub(a,b),0)→cmp(a,b)` 子改写。当前不能仅按普通有限
  样本放行，因为 IEEE Inf/NaN 与定宽整数溢出均已形成反例。
- 测试至少覆盖：arange 直出保持 int64、safe arange→comparison 仍降级；large float
  `.to(int64)` 保持；int32→int64→comparison 可降级；broadcast shape mismatch 保持、
  equal-shape 0/1 正例仍压缩；finite 普通、Inf 与 int32 overflow 的 cmp-sub 全部保持原图并
  执行等价。完成 60+ 新增 FX 全量、lintrunner、source wheel `--no-deps` 安装和 fresh NPU
  observer 后，才决定每条性能候选。
- 功能关闭结果为源码/安装态 67/67 与 9/9 audit-shim NPU worker；T-039 又完成 18/18
  paired worker。safe `dtype_optimal_pass` 与 iota downcast 的 p50 分别改善 52.06%/55.78%，
  均为 beneficial；equal-shape mask 化简只改善 0.30%，为 neutral。三者都不需要手写
  Triton 替身，且不能用这些 safe-positive 收益恢复 T-038 已证伪的不安全改写。

### P-001：`mm_plus_mm` NPU 支持

- 状态：`draft`，仅调研，禁止实现。
- 目标：评估上游 `mm + mm -> tuned_mm_plus_mm` pattern 在 NPU 的可用性与收益。
- 源码与动态证据：default backend 的 `patch_pattern_mm_plus_mm()` 对 NPU 禁用目标 pattern；experimental loader 会先恢复 Inductor baseline，正例已实际进入上游 `tuned_mm_plus_mm` 并生成 `extern_kernels._mm_plus_mm`。因此缺口是 default backend 支持与各路径性能对比，不是所有 NPU backend 都缺少实现。
- 候选方案：NPU 专用 lowering + CATLASS/AscendC/vendor 实现；Triton `tl.dot` 仅在能力和性能实测通过后纳入。
- 计划文件：待环境与方案审核后填写，当前不得修改。
- 验证：四输入静态/动态形状、转置/非连续输入、fp16/bf16/fp32、空维度、精度与 paired benchmark。
- 开放条件：专用实现相对两个独立 matmul 加 add 的稳态 p50 有明确收益，且 p99、编译时间、峰值内存无不可接受回退。

### P-002：`pad_mm` NPU 支持

- 状态：`closed-capability-available-performance-rejected`；当前产品仍由device gate排除NPU，本轮不实施。
- 目标：确认 shape padding 对 NPU GEMM 的收益和 padding/slice 图的承接缺口。
- 源码与动态证据：上游 `pad_mm.py:check_device()` 仅允许CUDA/XPU，NPU在`can_pad()`的device gate直接失败；测试侧只绕过该gate后，mm/bmm/addmm positive图均真实生成pad→GEMM→slice并正确执行，aligned negative均保持原图。三轮paired p50分别回退72.65%/65.31%/120.63%，每步task由1增至5/5/7，allocated peak多278,528/294,400/279,552 B。
- 方案结论：不把device/capability判断改为默认接受NPU，也不以手写Triton复制独立padding。当前阻力不是“padding算子缺失”，而是padding的任务和内存成本超过对齐GEMM收益。未来只有更大或特殊layout cohort先证明收益时，才评估把masked load融合进NPU专用GEMM/CATLASS/AscendC/Triton template。
- 验证：本轮已覆盖static fp16 contiguous positive与aligned negative、三family图gate/正确性、三轮warm10/runs100、task profile、首次编译和显存；未扩展dtype/dynamic/backward，因为首批性能gate已失败。

### P-003：`addmm` fusion NPU 策略

- 状态：`draft`，仅调研，禁止实现。
- 目标：比较 add+mm、NPU addmm lowering、CATLASS 和 fallback 的正确性与性能。
- 源码证据：Triton experimental backend 可禁用已注册的 add+mm -> addmm pattern。
- 候选方案：按 dtype/shape/layout 的 capability gate 选择 NPU addmm/CATLASS；没有数据前不全局打开。
- 计划文件：待环境与方案审核后填写，当前不得修改。
- 验证：bias broadcast、alpha/beta、静态/动态 shape、连续/非连续输入和 forward/backward。

### P-004：torch_npu reduction `strict_sum` 接口兼容

- 状态：`verified`；已由 T-011 按登记范围实施，最小 sum、vector/full bias addmm backward 和近邻 backend 回归通过。
- 问题：PyTorch 2.14 的 sum lowering 调用 `make_reduction(..., strict_sum=...)`；修复前的 torch_npu 覆盖函数不接受该参数，导致 addmm vector/row-bias backward 在 bias 梯度归约阶段编译失败。
- 候选修改：优先让 `torch_npu/_inductor/lowering.py:make_reduction()` 与上游签名和 `Reduction.create()` 语义保持一致；同时审查 Ascend NPU IR 内复制的旧实现，避免只修 default backend 后在 DVM/MLIR 重现。不得用测试 monkeypatch 或强制 fallback 掩盖接口漂移。
- 构建判断：这是 Python lowering 修改，开发迭代通常不需要重编 PyTorch/C++；但当前正式环境使用源码构建 torch_npu wheel，最终交付应从相同 commit 重建 wheel 并 `--no-deps` 安装验证。
- 验证：最小 `sum(dim, keepdim)` compile case；addmm vector/row/full bias forward/backward；fp16/bf16/fp32；default 与适用的 experimental/DVM/MLIR backend；检查 strict-sum 数值、generated code、无 CPU fallback，并回归现有 P0 inference/performance 哨兵。

### P-005：mm_plus_mm different-K 融合承接

- 状态：`integration-supported-beneficial-opt-in-memory-tradeoff-environment-pending`；T-023 已把 NPU-only template choice 与 extern fallback-first 接入 source-built wheel，功能与首批两个 static cohort 的集成 single-task、paired 性能、steady memory均完成。shape-A/unaligned p50 改善 `15.29%/18.04%`，但 candidate 比 baseline多 `270,336 B` peak allocated；T-024 没找到同时守住显存与 task-duration 门槛的 tile/分组。large 继续 `supported-neutral-hold`，默认关闭不变；当前 fresh host launcher仍有 PyTorch C++20/Triton C++17/torch_npu-CANN headers 合同缺口，无 shim candidate 会安全回退。剩余项只有匹配 headers 的正式无 shim 环境复验，不全局解除 size guard。
- 问题：post-grad pattern 允许两条 matmul 使用不同 K，但 `tuned_mm_plus_mm()` 要求两对输入 size 相同，否则安全退回两个 mm 加 add。NPU experimental 可编译且正确，但没有得到 fusion 的 kernel 数与带宽收益。
- 候选方案：按 T-021，在 torch_npu default backend 增加默认关闭的 NPU-only/static/different-K duplicate pattern 和 `NPUTritonTemplate` choice，始终把当前支持 different K 的 ATen extern 放在 fallback 列表中；candidate 使用独立 K1/K2 loop、mask 和 stride，不全局 monkeypatch 上游 handler。首批 capability 排除 large、dynamic、空维和未验证 dtype/layout；`128x128x128` 只作为已 profile 的候选 tile，不按单一 shape 白名单冒充通用能力。
- 验证：已关闭 K1/K2 different、对齐/非对齐、真实转置、dynamic 语义、三 dtype、forward/backward、large 三 tile device profile、paired host 稳定性与内存分解。正式接入仍需 pattern 触发/不触发、空维、数值压力、template choice 编译失败回退、Inductor autotune/cache、真实 AOTAutograd 图和匹配环境下 fresh launcher compile；p50、p99、内存与 fallback gate 必须同时满足。large 当前结论为 supported-neutral-hold，不应重新用 T-020 高方差的 11.58% 放行。

## 实施日志

- 2026-08-21，T-011：`src/torch_npu/torch_npu/_inductor/lowering.py` 的 `make_reduction()` 已增加 keyword-only `strict_sum: bool = False`，并在 dump/non-dump 两个分支传入 `Reduction.create(strict_sum=...)`。E-012 至 E-021 的 editable/build/bootstrap 边界已诊断并写入可复用 wrapper/smoke。E-022/E-023 完成新 wheel 安装、最小 sum、原始 vector-bias blocker 和近邻回归；addmm 最终 verdict 升级为 `supported-beneficial`。
- 2026-08-21，T-012/T-013：different-K 当前 fallback 的 paired baseline 为 neutral；profiler 证明每步为两个 aclnnMm 加一个 Triton add，并建立 33.93/31.13 μs candidate 单 task 预算。
- 2026-08-21，T-014：standalone Triton 原型经过 split-store、accumulation 语义两轮修正，shape-A/128³ 双累加器版本正确性最大绝对误差为 0。用户要求暂存后已停止扩展，未运行 unaligned 或性能测试；恢复入口固化在 `PAUSED_CHECKPOINT_20260821.md`。
- 2026-08-21，T-014 至 T-016 恢复：unaligned 正确性通过；candidate profiler 在两 shape 都确认 10/10 单 task且低于预算；端到端 p50 改善 15.60%/17.12%，同时记录 additional peak 多约 1.38 MiB。当前结论限定为 fp16/contiguous/static forward，进入覆盖扩展而非源码接入。
- 2026-08-21，T-017 至 T-019：bf16/fp32、真实 transposed stride、dynamic replay 和 backward 语义通过。dynamic 的新 shape/divisibility 可触发一次 Triton specialization compile；backward 应由 AOTAutograd 切分的独立图承接，audit-only wrapper 不是正式接入。
- 2026-08-21，T-020：bf16/fp32 contiguous 与 fp16 transposed 的 paired p50 改善 11.20%/15.25%/12.70%；large 隔离有效重测的 p50 中位数改善 11.58% 但因方差、长尾、单轮 p50 回退和 additional peak 增至 5.90 MB 而 hold。P-005 进入正式设计阶段，未修改功能源码。
