# mm_plus_mm different-K NPU 正式接入设计审计

## 1. 结论

T-021 的建议是：首批实现放在 torch_npu 的 default Inductor backend，使用一条 NPU-only、different-K-only 的 post-grad lowering pattern 路由到 `NPUTritonTemplate`，并把 PyTorch 现有 `aten_mm_plus_mm` extern choice 作为始终可用的安全 fallback。候选模板不应通过全局 monkeypatch 替换上游 handler，也不应直接解除 PyTorch 的 size-equality guard。

首批应由默认关闭的 rollout 开关保护，只承接 static、2D、NPU、同 dtype/device、fp16/bf16/fp32、K1/K2 静态可知且不同的图。开关关闭、符号 shape、空维、不支持 dtype/layout、template 生成/编译失败或 autotune 不可用时，保持现有图或选择 extern fallback。large 在 T-022 profiler/tile/内存闭环前不能升级为默认能力。

这一方案是纯 Python template/pattern/lowering 修改，开发本身不需重编 PyTorch C++；但当前正式环境从 site-packages 导入源码构建的 torch_npu wheel，因此实机集成验证和最终交付都必须重建 torch_npu wheel 并用 `--no-deps` 安装。不需重建 PyTorch wheel。

## 2. 当前源码数据流

### 2.1 Pattern 允许 different K

`torch/_inductor/fx_passes/post_grad.py:is_valid_mm_plus_mm():979` 检查两对 matmul 各自的 inner K 合法，并要求两个输出的 M/N 相同，但它没有要求 K1==K2。`mm_plus_mm():1016` 因此会匹配 different-K 图，然后调用 `torch._inductor.kernel.mm_plus_mm.tuned_mm_plus_mm()`。

~~~text
aten.mm(A, B) + aten.mm(C, D)
  -> is_valid_mm_plus_mm: K1 pair/K2 pair 各自合法，M/N 相同
  -> post_grad.mm_plus_mm handler
  -> tuned_mm_plus_mm(A, B, C, D)
~~~

### 2.2 Lowering 在 different K 上主动退回

`torch/_inductor/kernel/mm_plus_mm.py:tuned_mm_plus_mm():128` 分别调用两次 `mm_args()`，但随后要求 A/C 和 B/D 的完整 size 静态相等。不相等时直接返回：

~~~python
lowerings[aten.add](
    lowerings[aten.mm](mat1, mat2),
    lowerings[aten.mm](mat3, mat4),
)
~~~

这就是 T-012/T-013 观察到的两个 `aclnnMm` + Triton add。此处的 size guard 不是 pattern 可用性 gate，而是对现有实现的安全保护。

### 2.3 上游 Triton template 确实不支持 K2

`torch/_inductor/kernel/mm_plus_mm.py:mm_plus_mm_template:33` 虽然读取 K1，但 K2 仍是注释；C/D 的 contiguous 判断使用 K1，第二条 reduction loop 也是 `range(K1, ...)`。直接删除 lowering size guard 会让该 template 越界读取或少算 K2，不是合法修复。

T-014 的 standalone 修正了三点：

1. K1/K2 分别建 loop 和 tail mask；
2. 四个输入分别使用实际 stride；
3. 两个 fp32 accumulator 分别 cast 到输出 dtype 后再相加，保持当前 fallback 的舍入顺序。

### 2.4 NPU 当前只有 extern choice，没有 mm_plus_mm Triton heuristic

`torch/_inductor/kernel/mm_plus_mm.py:29` 定义了 `aten_mm_plus_mm` extern choice。`torch/_inductor/heuristics/template/aten.py:36` 把该 choice 以 device `None` 注册，所以 NPU 也能使用。对应的 C++ CompositeExplicitAutograd 实现在 `torch/csrc/inductor/inductor_ops.cpp:_mm_plus_mm_out():13`：先 `mm_out(out, a, b)`，再 `out.addmm_(c, d)`，本身不要求 K1==K2。

相反，PyTorch 的 Triton `mm_plus_mm_template` heuristic 只对 CUDA、ROCm、CPU、XPU 和 MTIA 注册；没有 device type `npu` 的注册。`V.choices.get_template_configs()` 在找不到 NPU heuristic 时使用不产生任何 config 的 `TemplateConfigHeuristics` base fallback。因此当前 same-K experimental 图使用 `extern_kernels._mm_plus_mm`，并不证明上游 Triton template 已在 NPU 上进入 algorithm choices。

### 2.5 torch_npu default backend 明确禁用该 pattern

`torch_npu/_inductor/fx_passes/post_grad.py:patch_pattern_mm_plus_mm():24` 把上游 `LoweringPatternEntry.extra_check` 包装为“非 NPU 才允许”。default loader 在 `torch_npu/_inductor/__init__.py:197` 调用该 patch，之后再安装 NPU algorithm selector、scheduler 和 wrapper。experimental loader 恢复上游 lowering/codegen 基线，所以它能观察到上游 same-K/different-K 行为，但这不是 default backend 的生产承接。

### 2.6 正式 NPU template 基类已经存在

`torch_npu/_inductor/select_algorithm.py:NPUTritonTemplate:356` 已经承接 NPU template render、benchmark request、wrapper 输出、32-bit indexing 检查与 scheduler/codegen。`patch_algorithm_selector():989` 又为 NPU 安装选择、precompile、benchmark 和失败忽略逻辑。因此正式接入应用 `NPUTritonTemplate`，不应在 lowering 中直接调用 eager Triton launcher。

## 3. 方案比较

| 方案 | 优点 | 风险 | 结论 |
|---|---|---|---|
| 直接修 PyTorch 上游 template 并解除 guard | 形式通用 | 影响所有设备；当前 K2、舍入和 NPU heuristic 都未解决 | 不作首批 NPU 接入 |
| 全局 monkeypatch `tuned_mm_plus_mm` | 修改少 | 污染 CPU/其他 backend，multi-backend restore 边界不清晰 | 拒绝 |
| 在 lowering 中直接调 standalone Triton | 容易复用原型 | 绕过 Inductor scheduler、wrapper、memory planner、cache 和 autotune | 拒绝 |
| NPU-only 重复 pattern + `NPUTritonTemplate` + extern fallback | 设备范围清晰，可使用 NPU selector/wrapper，失败可退 | 需新模板、注册和集成 UT；autotune 还要实机验证 | 选择 |

## 4. 选定设计

### 4.1 Pattern 层

在 torch_npu 注册一条与上游相同结构的 lowering pattern，但 `extra_check` 同时要求：

- rollout 开关开启；
- match 为 NPU；
- 复用 `post_grad.is_valid_mm_plus_mm(match)` 的两对 inner-K 与 M/N 合法性；
- K1/K2 都静态可知且 K1!=K2；
- 4 个 meta tensor 均为 2D、同 device/dtype，dtype 为 fp16/bf16/fp32；
- 输出非空。

上游 entry 在 default backend 上仍对 NPU 关闭，因此 same-K 行为不在首批修改范围内。新 pattern 使用独立 handler，不替换 `post_grad.mm_plus_mm` 的全局函数对象。注册必须幂等，并为 fresh process/backend switch 增加结构 UT。

### 4.2 Choice 层

新 `tuned_npu_mm_plus_mm()` 对两对输入各调用 `mm_args()`，确认输出 layout 一致，然后构造两类 choice：

1. **extern fallback first**：无条件加入 `torch._inductor.kernel.mm_plus_mm.aten_mm_plus_mm.bind(...)`。它的 C++ composite 实现支持 different K，并且第二个 matmul 用 `addmm_` 写入同一输出。即使用户的 generic `max_autotune_gemm_backends` 未包含 ATEN，该特殊融合也应保留它作为可编译的回退。
2. **NPU Triton candidate**：使用 `NPUTritonTemplate` 生成独立 K1/K2 loop、四组 stride、两个 accumulator 和 dtype-specific cast 的单 task 模板。首个 config 只复制已验证的 `128x128x128`，不在这一 patch 中伪造未验证的 autotune 空间。

`autotune_select_algorithm()` 对可编译 choice 测速；candidate 生成/编译/运行失败时记为无效，extern 仍可选。如果 NPU benchmark 无有效 timing，选择器必须退到排在第一的 extern，不能因 choices 为空使整图编译失败。

### 4.3 Rollout 与 shape 边界

新开关建议使用 torch_npu 现有 `_parse_bool_env()` 风格，例如 `TORCHINDUCTOR_NPU_ENABLE_DIFFERENT_K_MM_PLUS_MM`，默认 `False`。首批实现只处理 static different K：

- dynamic/unbacked symbol 不匹配新 pattern，保持当前两 mm + add 图；
- empty M/N/K 不匹配，保持上游空维语义；
- same-K 不匹配，不改变 default backend 现有 gate；
- large 不通过绑定测试 shape 的 whitelist 硬编码排除。在开关默认关闭的开发阶段完成 T-022，再根据 profiler/tile/内存数据设计可解释的成本 gate；没有该 gate 前不把开关改为默认开启。

### 4.4 舍入与 output dtype

候选模板保留 T-017 已验证的语义：A@B 和 C@D 分别使用 fp32 accumulator，各自 cast 到 output dtype，再以 output dtype 相加并写回。fp16/bf16/fp32 分别使用 `1e-2/3e-2/1e-4` 的既定验收容差。不能回到“两个 dot 都累加到一个 fp32 acc 后只 cast 一次”的初版语义。

## 5. 拟修文件

| 文件 | 拟修改 | 是否功能修改 |
|---|---|---|
| `torch_npu/_inductor/kernel/mm_plus_mm.py` | 新增 `NPUTritonTemplate`、candidate choice 和 `tuned_npu_mm_plus_mm()` | 是，新文件 |
| `torch_npu/_inductor/kernel/__init__.py` | 导出新注册/调用入口 | 是 |
| `torch_npu/_inductor/fx_passes/post_grad.py` | 保留上游 NPU disable，幂等注册 NPU-only different-K pattern | 是 |
| `torch_npu/_inductor/__init__.py` | default backend loader 按顺序安装新 pattern/template | 是 |
| `torch_npu/_inductor/config.py` | 新增默认关闭 rollout 开关 | 是 |
| `test/_inductor/test_mm_plus_mm.py` | 增加结构、功能、fallback、dynamic 和 AOTAutograd UT | 测试 |

首批不需修改 PyTorch 源码、Triton Ascend 源码、C++ dispatcher 或 AOTI shim。如果实践证明 `NPUTritonTemplate` 无法表达所需 kernel，应停止并回到本设计评审，不得在 wrapper/generated code 中硬插 eager launcher。

## 6. 验证合同

### 6.1 结构与路由

- rollout off：NPU different-K 保持两 mm + add，结果与 T-012 一致；
- rollout on 且 static different-K：transformed graph 进入 NPU handler，generated code 只能是 candidate 或 extern fallback；
- same-K、CPU 和非 NPU 不进入新 handler；
- 强制 candidate 生成/编译失败时，extern fallback 仍正确执行；
- loader/backend 重复调用不重复注册，不改变其他 pattern 数量。

任一 compile/correctness 失败都按图模式分流检查本轮 `output_code.py`；成功用例不展开该产物。

### 6.2 功能 cohort

- fp16 shape-A/unaligned contiguous；
- bf16/fp32 shape-A contiguous；
- fp16 unaligned true-transposed stride；
- K1/K2 各自尾块，M/N 尾块；
- empty M/N/K、mixed dtype、M/N 不相等等负例；
- 数值压力：大/小幅值、正负抵消、非有限输入语义、更大 K；
- `dynamic=True` first/replay 确认新 handler 不触发，当前 fallback 语义不变；
- AOTAutograd output 和 A/B/C/D 四梯度，确认 forward 候选与独立 backward graph。

### 6.3 性能与内存

使用正式 compiled candidate 对比 rollout-off current graph 和强制 extern fallback，每配置 warmup 10、runs 100、3 轮交错，报告 mean±stdev、p50、p99、compile first、allocated/reserved peak 和 NPU profiler task duration。

首批必须重测 T-016/T-020 的中小 cohort，并为 large 增加：

1. candidate 单 task profiler；
2. 至少 64³、128³ 和一个中间 tile；
3. pure `torch.empty`/extern/template 的 allocated 与 reserved 分解；
4. 无外部 NPU 进程且 CPU 调度稳定的重复采样。

## 7. 内存问题的当前判断

standalone candidate additional peak 在 shape-A fp16/bf16 约 1.70 MB，fp32 约 3.39 MB，large fp16 约 5.90 MB。它随 dtype 和输出规模增长，所以不是简单的固定 runtime 常量。但 T-014 launcher 绕过 Inductor memory planner，当前证据不足以把峰值全部归因于 kernel accumulator 或 Triton runtime。

可能组成包括：output 的 allocator size class/padding、Triton launch/runtime 通过 PyTorch allocator 申请的缓冲、以及 standalone 输出没有参与 scheduler reuse。正式 `NPUTritonTemplate` 可能改变其中一部分，也可能不改变。因此 T-022 必须分解测量，在数据前不写“固定 workspace”或“只是 output”结论。

T-022 后续更新：在 paired 前显式删除首次 baseline/candidate output 后，三个 large tile 的 pure output/baseline/candidate additional allocated peak 稳定为 `655,872/1,311,744/1,967,104 B`。candidate 比 baseline 多 `655,360 B`，恰为一个逻辑 fp16 output；T-020 的 `5.90 MB` 包含长生命周期首次输出造成的放大，不能继续作为 steady workspace。reserved delta 为 0 只代表 allocator 已预留 `73,400,320 B`，仍不能写成“无 workspace”。

## 8. 回滚边界与构建

回滚的最小单元是：默认关闭 rollout 开关，新 NPU pattern 不匹配，图恢复现有两 mm + add。代码回滚时只撤销上表 5 个 torch_npu Python 文件与新 UT，不撤销 T-011 `strict_sum` 修复，不触碰 Triton Ascend 已有 3 个外部修改或 torch_npu 未跟踪代码生成文件。

修改为纯 Python，所以：

- 不需重编 PyTorch wheel；
- 不需修改/重编 Triton Ascend；
- 需从同一 torch_npu commit 重建 wheel、记录 SHA256、使用 `pip --no-deps` 安装，再从 `/home/z50063656/tmp` 验证导入路径与用例。

T-022 还发现当前 PyTorch editable 源码树缺少 `torch/include`，wheel headers 要求 C++20，而安装的 Triton Ascend launcher 固定 C++17；installed torch_npu headers 又引用 CANN 9.0.1 缺失的 conditional graph 类型。审计 compiler shim 只能用于测量 device kernel。正式 integration 的 fresh launcher 验证必须先对齐 PyTorch/Triton/torch_npu/CANN header 合同；这不改变“纯 Python pass 修改无需重编 PyTorch”的代码边界判断。

## 9. 下一闸门

T-021 完成的是源码设计，不是实现授权。T-022 已完成 large profiler/tile/内存分解：三个 tile 功能和 device task 都通过，但 paired p50 改善只有 6.64%–7.55%，large 保持 supported-neutral-hold。下一步应在 `change_control.md` 新建独立 implementation 任务，逐文件列出首批 capability、默认关闭开关、patch、回滚、最小 UT 和 fresh launcher 环境复验；首批范围排除 large 与 dynamic。
