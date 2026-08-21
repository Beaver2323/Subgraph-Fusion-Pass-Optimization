# PyTorch Feature 设计与实现分析

> 目标模块：TorchInductor pass 在 Ascend NPU 上的可用性、性能评估与替代实现。
>
> 面向读者：了解 Python 和基本 PyTorch 用法，但尚未系统阅读 TorchDynamo、AOTAutograd、Inductor 或 torch_npu 源码的开发者。
>
> 当前源码基线：PyTorch <code>release/2.14@8e86e0a</code>、torch_npu <code>master@83cc452</code>、Triton Ascend <code>release/3.2.2@8bd9f38</code>。当前运行基线、wheel 哈希和设备信息以 [current_status_and_background.md](current_status_and_background.md) 为准。
>
> 项目于 2026-08-21 按用户要求暂停。新接手或恢复工作时先阅读
> [暂停检查点](PAUSED_CHECKPOINT_20260821.md)，不要直接启动性能测试。

## 模块设计目标与背景

### 1. 先把任务翻译成工程问题

本任务不是统计源码中有多少个名为 pass 的函数，也不是把所有计算改写成 Triton。最终交付是一张可审计的 NPU 评估矩阵：每个 pass 都要有适用性、最小触发图、实际匹配证据、正确性、generated code/fallback、性能和最终处理结论。完整任务边界见 [task_scope_and_code_map.md](task_scope_and_code_map.md)。

对每条记录需要回答：

1. 这个 pass 是否应该作用于 NPU 图？
2. 正例是否真正触发，负例是否避免误触发？
3. 图变换后数值、dtype、shape、stride、alias 和梯度是否正确？
4. 替换后的节点能否被 NPU lowering、scheduler 和 codegen 承接？
5. 最终执行的是 NPU kernel，还是 graph break、CPU fallback 或低效 extern 路径？
6. pass-on 相对同 backend 的 pass-off 是否有稳定收益？
7. 若不支持或回退，应修改 gate、lowering、kernel 还是编译器？

这七个问题对应评估矩阵中的动态字段。矩阵说明见 [pass_evaluation_matrix.md](report/pass_src_20260820/pass_evaluation_matrix.md)，机器可填写版本见 [pass_evaluation_matrix.csv](report/pass_src_20260820/pass_evaluation_matrix.csv)。

### 2. 必须先掌握的术语

| 术语 | 本任务中的含义 | 源码落点 |
|---|---|---|
| FX GraphModule | Dynamo/AOTAutograd 交给编译器的 Python 图结构，节点携带 target、参数和 meta 信息 | <code>torch/fx</code>；Inductor 入口接收 <code>GraphModule</code> |
| pass | 读取并可能修改 FX 图或 Inductor IR 的一次变换 | <code>torch/_inductor/fx_passes/pre_grad.py::pre_grad_passes():336</code> 等 |
| pattern | 对局部 FX 子图的结构描述；匹配成功后执行 handler 或 replacement | <code>torch/_inductor/pattern_matcher.py::PatternMatcherPass.apply():2605</code> |
| extra_check / gate | pattern 结构匹配后，对 device、dtype、shape、layout 等再做能力检查 | <code>torch/_inductor/pattern_matcher.py::PatternMatcherPass.apply():2679</code> |
| lowering | 将 Aten/FX 语义节点变成 Inductor IR、extern choice 或 fallback | <code>torch/_inductor/graph.py::GraphLowering.call_function():1405</code> |
| scheduler | 对 lowering 后的 IR 节点做依赖、融合、分区和内核调度 | <code>torch/_inductor/scheduler.py::Scheduler.codegen():10012</code> |
| codegen | 生成 NPU kernel 与调用这些 kernel 的 Python/C++ wrapper | <code>torch/_inductor/graph.py::GraphLowering.codegen():2994</code> |
| graph break | Dynamo 无法继续捕获整图，将 Python 函数拆成多个编译区或退回 eager | <code>torch/__init__.py::compile():3045</code> 的 <code>fullgraph</code> 契约 |
| fallback | 图仍可编译，但某个 op 没有专用 lowering，转为 Aten/extern 等保底路径 | <code>torch/_inductor/graph.py::GraphLowering.call_function():1405</code> |
| backend | NPU 代码生成和运行路径的组合，不只是一个性能开关 | <code>torch_npu/_inductor/__init__.py::_BACKEND_LOADERS:323</code> |

一个模型能成功执行，不代表目标 pass 可用。例如原始 <code>aten.mm</code> 能被 NPU 承接，但 <code>pad_mm</code> 可能从未匹配；因此必须同时观察变换图和 generated code。这个区别已经在 [P0 功能报告](report/p0_gate_first_run_20260820.md) 中得到实机验证。

### 3. 为什么 pass、lowering 和 kernel 必须分开判断

以 <code>mm + mm</code> 为例：

- pattern 负责识别两个矩阵乘再相加，入口是 <code>torch/_inductor/fx_passes/post_grad.py::mm_plus_mm():1016</code>。
- handler 调用 <code>torch/_inductor/kernel/mm_plus_mm.py::tuned_mm_plus_mm():128</code>，让算法选择器在 extern/template choice 中选择实现。
- torch_npu default backend 又通过 <code>torch_npu/_inductor/fx_passes/post_grad.py::patch_pattern_mm_plus_mm():24</code> 对 NPU 禁用这条 pattern。

因此“不支持”至少可能有三种完全不同的原因：

1. pattern 根本没有注册或没有匹配；
2. pattern 匹配，但 capability gate 拒绝 NPU；
3. 图已改写，但 lowering、scheduler 或 kernel 无法承接。

三者的修改位置不同。没有定位层级就直接写 Triton，通常会修错地方。

### 4. 当前项目已经做到哪里

当前事实以 [当前状态文档](current_status_and_background.md) 和两份 P0 报告为准：

- 已静态整理 251 条概念级记录；它们不是 251 个都能独立运行的函数。
- 记录中有 164 个 direct case、41 个 observer stage、25 个 registry container 和 21 个 manual review。
- P0 的 10 个正负用例分别在 default 和 triton_experimental backend 运行，20/20 编译与 eager 对齐。
- <code>pad_mm</code>、<code>pad_bmm</code>、<code>pad_addmm</code> 仍因device gate保持产品<code>unsupported</code>；T-025测试侧绕过后功能可承接，但T-026三类p50回退72.65%/65.31%/120.63%，replacement已按性能否决。
- addmm fusion/default 已覆盖三 dtype、代表 shape、真实转置、dynamic、bias guard 和 backward；8/8 配置的 p50 收益超过 10%。T-011 修复 `strict_sum` reduction 接口后，最终 verdict 为 <code>supported-beneficial</code>。
- mm_plus_mm/triton_experimental 的 same-K 代表网格 8/8 功能正确，6/8 p50 收益超过 10%，transposed/dynamic 为 neutral。
- mm_plus_mm different-K 的 T-014 至 T-024 已覆盖三 dtype、真实 transposed stride、dynamic replay、backward、正式接入、集成 paired、memory root cause和 workspace 替代搜索。T-023 的 default-off candidate在 shape-A/unaligned p50 改善 15.29%/18.04%，但比 baseline多 270,336 B peak allocated；T-024 没找到同时守住显存和 task-duration gate的配置。正式 verdict 为 <code>conditional-supported-beneficial</code>，large/dynamic/empty/arbitrary-stride/same-K继续 fallback。这说明“算子可用”和“可以默认启用”是两个不同结论。
- 当前矩阵总计 246 条 <code>not-run</code>、3 条 <code>unsupported</code>、1 条 <code>supported-beneficial</code>、1 条 <code>conditional-supported-beneficial</code>。

### 5. 从头阅读现有文档的路线

按下面顺序阅读，不要先从 CSV 或 debug 目录开始：

1. [task_scope_and_code_map.md](task_scope_and_code_map.md)：先理解目标、编译链和源码分工。
2. [current_status_and_background.md](current_status_and_background.md)：掌握当前环境、已有结论和性能方法。
3. [p0_case_design.md](p0_case_design.md)：学习如何为一个 pass 设计正例、负例和 gate。
4. [p0_gate_first_run_20260820.md](report/p0_gate_first_run_20260820.md)：学习如何区分“编译成功”和“目标 pass 触发”。
5. [p0_ab_first_shape_20260820.md](report/p0_ab_first_shape_20260820.md)：学习单 pass paired A/B。
6. [pass_evaluation_matrix.md](report/pass_src_20260820/pass_evaluation_matrix.md)：理解矩阵字段和批次。
7. [T-023 集成报告](report/t023_mmplus_different_k_integration_20260821.md) 与 [T-024 workspace 审计](report/t024_mmplus_different_k_workspace_20260821.md)：学习如何在性能收益和显存代价冲突时形成条件性结论。
8. [p1_batch_design.md](p1_batch_design.md)：进入下一批 NPU custom、DVM/MLIR 和 attention。
9. [change_control.md](change_control.md)：任何功能源码修改前，先登记证据、修改点、验证和回退。

根目录旧 <code>report/pass_inventory.md</code> 属于历史诊断；当前静态基线是 <code>report/pass_src_20260820/</code>，动态环境由 <code>/home/z50063656/Benchmark/env.sh</code> 启动 Conda <code>benchmark-py311</code>。T-022/T-023 还要求区分 runtime 已可用与 fresh Triton host launcher 编译合同是否完整。

## 整体设计架构

### 核心组件说明

| 组件 | 职责 | 文件路径 |
|---|---|---|
| torch.compile API | 解析 fullgraph、dynamic、backend、mode、options，构造 Inductor wrapper | <code>torch/__init__.py::compile():3045</code> |
| Inductor wrapper | 保存 compile 配置，把 Dynamo 捕获的 GraphModule 交给 compile_fx | <code>torch/__init__.py::_TorchCompileInductorWrapper:2818</code> |
| TorchDynamo | 捕获 Python frame，生成 FX GraphModule，并调用 backend callable | <code>torch/_dynamo/eval_frame.py::_optimize():1791</code> |
| compile_fx | 组织 pre-grad、AOTAutograd、joint graph、post-grad 和实际编译 | <code>torch/_inductor/compile_fx.py::compile_fx():2901</code> |
| pre-grad pass | 在未完全 functionalize 的图上规范化和融合，必须谨慎处理 alias/mutation | <code>torch/_inductor/fx_passes/pre_grad.py::pre_grad_passes():336</code> |
| joint graph pass | 在联合前后向图上执行 pattern、常量折叠和随机算子处理 | <code>torch/_inductor/fx_passes/joint_graph.py::joint_graph_passes():699</code> |
| post-grad pass | 在 normalized/functionalized 图上执行主要 pattern 与 custom pass | <code>torch/_inductor/fx_passes/post_grad.py::post_grad_passes():177</code> |
| pattern matcher | 保存 pattern registry，匹配节点、检查 extra_check、执行 replacement 并计数 | <code>torch/_inductor/pattern_matcher.py::PatternMatcherPass:2583</code> |
| GraphLowering | 将 FX 节点转成 Inductor IR；没有 lowering 时决定 fallback 或报错 | <code>torch/_inductor/graph.py::GraphLowering:386</code> |
| backend codegen registry | 为设备注册 Scheduling 和 WrapperCodegen | <code>torch/_inductor/codegen/common.py::register_backend_for_device():431</code> |
| torch_npu backend loader | 按 default/ascendc/mlir/dvm/triton_experimental 装载 patch、lowering 和 codegen | <code>torch_npu/_inductor/__init__.py::_load_backend():331</code> |
| NPU custom pass registry | 以 PassType 和 FxPassLevel 保存并执行 NPU pass | <code>torch_npu/_inductor/fx_passes/ascend_custom_passes/register_custom_pass.py::register_custom_pass():28</code> |
| NPU GEMM lowering | 为 aten.mm/addmm 注册 ATen、CATLASS、外部 choice 或 fallback | <code>torch_npu/_inductor/kernel/mm.py::_register_npu_inductor_mm():56</code>、<code>_register_npu_inductor_addmm():126</code> |
| NPU scheduler/wrapper | 承接 NPU IR，生成内核调度和运行 wrapper | <code>torch_npu/_inductor/codegen/npu_combined_scheduling.py::NPUCombinedScheduling:25</code>、<code>codegen/wrapper.py::NPUPythonWrapperCodeGen:199</code> |
| 审计工具 | 静态枚举、fresh-worker 动态探针、证据归档和矩阵回填 | <code>inductor_pass_npu_audit/audit_passes.py</code>、<code>run_p0_gate_probe.py</code> |

### 整体执行流程

~~~mermaid
flowchart TD
    A[Python 模型或函数] --> B[torch.compile]
    B --> C[TorchDynamo 捕获 FX 图]
    C --> D[Inductor compile_fx]
    D --> E[pre_grad passes]
    E --> F[AOTAutograd / functionalization]
    F --> G[joint_graph passes]
    G --> H[前向/反向图分区]
    H --> I[post_grad passes]
    I --> J[GraphLowering: FX 到 Inductor IR]
    J --> K[Scheduler / Fusion]
    K --> L[NPU backend codegen + wrapper]
    L --> M[NPU kernel / vendor extern / fallback]

    N[torch_npu backend loader] --> E
    N --> G
    N --> I
    N --> J
    N --> K
    N --> L

    O[observer + counters + debug artifacts] -.证据.-> E
    O -.证据.-> G
    O -.证据.-> I
    O -.证据.-> J
    O -.证据.-> L
~~~

这样的拆分允许通用 PyTorch pass 与设备后端解耦：通用语义优化留在 PyTorch，NPU 能力、lowering 和 codegen 放在 torch_npu。源码对此提供了明确扩展接口：<code>torch/_inductor/codegen/common.py::register_backend_for_device():431</code> 注册设备调度和 wrapper；<code>torch/_inductor/config.py</code> 的 custom pass 槽位允许后端注入图变换。

### pass 阶段为什么不能混为一谈

| 阶段 | 图的特点 | 编写 pass 的主要风险 |
|---|---|---|
| pre_grad | 尚未完全 functionalize/normalize | alias、mutation、参数 schema；源码在 <code>pre_grad_passes()</code> docstring 中明确警告 |
| joint_graph | 前后向联合图 | 前后向语义、随机数、保存张量、pattern 训练变体 |
| post_grad | normalized/functionalized，前向和反向分别调用 | lowering 可承接性、stride/meta、设备 gate |
| lowering/IR | 不再只是 FX 图 | layout、extern choice、fallback、动态符号 |
| scheduler/codegen | 内核与 wrapper 生成 | fusion、stream、内存依赖、编译器能力 |

因此矩阵中的 251 条记录被分成 direct case、observer stage、registry container 和 manual review。registry 本身通常不独立测性能，而由其中的 entry 覆盖。

### 评估数据流

~~~mermaid
flowchart LR
    S[源码扫描] --> I[pass_inventory.json]
    I --> M[251 条评估矩阵]
    M --> C[最小正例和负例]
    C --> T[目标 pass 触发证据]
    T --> R[正确性与 fallback/codegen]
    R --> P[pass-on / pass-off 性能]
    P --> V[最终 verdict]
    V --> X{需要替代?}
    X -- 否 --> D[保留或标记不适用]
    X -- 是 --> Q[变更提案]
    Q --> K[vendor / CATLASS / AscendC / Triton]
~~~

## 入口分析

### 入口一：torch.compile

文件：<code>src/pytorch/torch/__init__.py</code>

入口函数：<code>torch.compile():3045</code>

职责：

- 接收 model、fullgraph、dynamic、backend、mode 和 options。
- 当 backend 为 <code>inductor</code> 时构造 <code>_TorchCompileInductorWrapper</code>。
- 最终调用 <code>torch._dynamo.optimize(...)(model)</code>，源码位置为 <code>torch/__init__.py::compile():3280</code>。

本任务的最小使用方式：

~~~python
import torch
import torch_npu

def fn(x, y):
    return x @ y

x = torch.randn(128, 256, device="npu", dtype=torch.float16)
y = torch.randn(256, 320, device="npu", dtype=torch.float16)

compiled = torch.compile(
    fn,
    backend="inductor",
    fullgraph=True,
    options={"npu_backend": "default"},
)
actual = compiled(x, y)
expected = fn(x, y)
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
~~~

参数在本任务中的意义：

| 参数 | 建议 | 原因 |
|---|---|---|
| backend | 固定为 <code>inductor</code> | 本任务研究的是 Inductor pass |
| fullgraph | 功能门禁阶段设为 True | graph break 立即报错，避免局部 eager 掩盖问题 |
| dynamic | 分静态 False 与动态 True 测试 | 动态 shape 可能改变 gate、specialization 和 lowering |
| mode | GEMM pattern 需要时使用 max-autotune | <code>is_valid_mm_plus_mm()</code> 要求 max_autotune 或 max_autotune_gemm |
| options.npu_backend | 每个 fresh process 只选一个 | backend loader 会修改全局 Inductor 注册状态 |
| options.trace.enabled | 需要 generated code 时开启 | 生成 torch_compile_debug 证据 |

### 入口二：NPU backend 选择

torch_npu 将 <code>npu_backend</code> 注入 Inductor config，并按以下优先级解析：

1. 当前 compile options；
2. <code>torch._inductor.config.npu_backend</code>；
3. 环境变量 <code>TORCHINDUCTOR_NPU_BACKEND</code>；
4. 默认 <code>default</code>。

源码证据：<code>torch_npu/utils/_dynamo.py::_resolve_npu_backend():140</code>。每次 compile 调用进入 <code>_NpuBackendScope</code> 临时设置环境并调用注册，退出后恢复，见 <code>torch_npu/utils/_dynamo.py::_NpuBackendScope:153</code>。

可选 backend 来自 <code>torch_npu/_inductor/__init__.py::_BACKEND_LOADERS:323</code>：

- <code>default</code>：当前 NPU Triton/ATen/CATLASS 等综合路径；
- <code>triton_experimental</code>：物理隔离的 experimental Triton codegen 与 heuristics；
- <code>ascendc</code>：AscendC 承接路径；
- <code>mlir</code>：MLIR codegen；
- <code>dvm</code>：DVM graph fusion 与 MLIR 承接。

backend 不是普通配置开关。loader 会注册 lowering、替换函数和 codegen，有全局状态，所以正式测试坚持一个 case/backend 一个 fresh process。

### 入口三：静态清单

工具：<code>inductor_pass_npu_audit/audit_passes.py</code>

用途：不导入 torch，通过 AST/文本扫描枚举 pipeline、pattern、NPU custom pass、DVM/MLIR 变换和 extension hook。

运行入口：

~~~bash
cd /home/z50063656/tmp
python /home/z50063656/Pass/inductor_pass_npu_audit/audit_passes.py +  --pytorch-root /home/z50063656/Pass/src/pytorch +  --torch-npu-root /home/z50063656/Pass/src/torch_npu +  --output /home/z50063656/Pass/inductor_pass_npu_audit/report/pass_src_20260820
~~~

静态标签只负责路由，不能证明 NPU 可用。一个函数出现在源码中，最多说明“存在候选入口”。

### 入口四：P0 动态探针

工具：<code>inductor_pass_npu_audit/run_p0_gate_probe.py</code>

关键实现：

- <code>_instrument():229</code> 包装 <code>GraphTransformObserver</code> 和 <code>PatternMatcherPass.apply</code>，记录阶段调用、匹配数和 active config。
- <code>worker():424</code> 在独立 worker 内导入 torch、构造 case、执行 eager、compile、正确性和内存采集。
- <code>orchestrate():506</code> 强制从 <code>/home/z50063656/tmp</code> 启动，并为每个 case/backend 生成子进程。
- <code>_disable_target_pass():386</code> 仅在测试进程内关闭目标 pattern，用作单 pass A/B baseline。

运行前必须：

~~~bash
cd /home/z50063656/tmp
source /home/z50063656/Benchmark/env.sh
~~~

不得在 torch_npu 源码目录中导入 torch，避免源码树级联导入污染正在验证的 wheel。

## 完整调用链分析

### 阶段 0：torch_npu 懒加载 NPU 编译支持

入口函数：

- <code>torch_npu/utils/_dynamo.py::_lazy_dynamo_setup():730</code>
- <code>torch_npu/utils/_dynamo.py::_setup_inductor_for_compile():751</code>
- <code>torch_npu/utils/_dynamo.py::_InductorNpuRegistry.register_inductor_npu():98</code>

输入：torch.compile 的 options 和当前环境/config。

状态变化：

1. 注册 NPU Dynamo device interface 与 trace rule。
2. 解析当前 NPU backend。
3. 首次导入 <code>torch_npu._inductor</code>，或 backend 改变时重新执行 <code>_load_backend()</code>。
4. 把已加载 backend 记录到 <code>_InductorNpuRegistry._loaded_backend</code>。

输出：PyTorch Inductor 的全局 registry 中出现 NPU device interface、lowering、scheduler、wrapper 和 custom pass。

修改建议：backend 选择逻辑改 <code>torch_npu/utils/_dynamo.py</code>；backend 内容改 <code>torch_npu/_inductor/__init__.py</code>。不要在业务模型里永久 monkeypatch 全局 registry。

### 阶段 1：torch.compile 构造 backend wrapper

函数：

- <code>torch/__init__.py::compile():3045</code>
- <code>torch/__init__.py::_TorchCompileInductorWrapper.__init__():2821</code>
- <code>torch/__init__.py::_TorchCompileInductorWrapper.apply_options():2868</code>

输入：Python callable 与 compile 参数。

状态变化：

- mode 和 options 被标准化到 wrapper 的 <code>self.config</code>。
- torch_npu 对 wrapper 做受控 patch，在 <code>torch_npu/utils/_dynamo.py::patch_inductor_wrapper():194</code> 中注入 <code>npu_backend</code>、shape handling 配置和 backend scope。

输出：一个 callable backend wrapper。

为什么调用到这里：<code>torch.compile</code> 在 backend 为 <code>inductor</code> 时显式构造该 wrapper，然后交给 Dynamo。

### 阶段 2：Dynamo 捕获 FX 图

函数：

- <code>torch/__init__.py::compile():3280</code>
- <code>torch/_dynamo/eval_frame.py::optimize():1775</code>
- <code>torch/_dynamo/eval_frame.py::_optimize():1791</code>

输入：原 Python 函数和 backend wrapper。

状态变化：Dynamo 对 Python frame 安装捕获上下文，生成 FX GraphModule、guards 和 example inputs。<code>fullgraph=True</code> 对应 <code>nopython=True</code>，捕获失败时不允许静默拆成多个图。

输出：首次调用时触发编译、后续满足 guards 时复用的 callable。

修改建议：本任务通常不改 Dynamo。只有明确证明捕获规则或 guard 阻止了合法 NPU 图时，才进入 <code>torch/_dynamo</code>；否则优先在 pass 或 backend 层解决。

### 阶段 3：Inductor compile_fx 组织 pre-grad 与 AOTAutograd

函数：

- <code>torch/__init__.py::_TorchCompileInductorWrapper.__call__():2895</code>
- <code>torch/_inductor/compile_fx.py::compile_fx():2901</code>
- <code>torch/_inductor/compile_fx.py::run_pre_grad_passes():2836</code>
- <code>torch/_inductor/compile_fx.py::_recursive_pre_grad_passes():564</code>

输入：Dynamo 产生的 GraphModule、example inputs、config patches。

状态变化：

1. compile_fx 应用配置并取得 decomposition table。
2. pre-grad 阶段递归处理子图，再调用 <code>pre_grad_passes()</code>。
3. compile_fx 组织 AOTAutograd；其 docstring 明确说明该函数负责调用 AOTAutograd，并在回调中进入 inner compile。

输出：functionalization/partition 前后的图和后续编译回调。

pre-grad 的源码警告很重要：<code>torch/_inductor/fx_passes/pre_grad.py::pre_grad_passes():336</code> 明确指出此时图尚未 functionalize/normalize。任何删除 view、clone 或 mutation 的优化，都要先证明 alias 语义安全。

### 阶段 4：joint graph 和 post-grad pass

joint graph：

- 递归入口：<code>torch/_inductor/compile_fx.py::_recursive_joint_graph_passes():588</code>
- 主执行：<code>torch/_inductor/fx_passes/joint_graph.py::joint_graph_passes():699</code>

joint graph 先执行 custom pre pass、canonicalization、constant folding、pattern registry、随机数处理和 custom post pass。输入设备会传给 lazy pattern 初始化，因此生成式 attention pattern 的设备上下文不能忽略。

post-grad：

- 递归入口：<code>torch/_inductor/compile_fx.py::_recursive_post_grad_passes():636</code>
- 主执行：<code>torch/_inductor/fx_passes/post_grad.py::post_grad_passes():177</code>

post-grad 的图已经 normalized/functionalized；函数会在前向、反向图分别调用。执行顺序包括：

1. DCE、locality 等通用变换；
2. <code>config.post_grad_custom_pre_pass</code>；
3. 通用 pattern matcher registry；
4. <code>config.post_grad_custom_post_pass</code>；
5. backend-specific custom pass。

torch_npu default backend 通过 <code>torch_npu/_inductor/fx_passes/graph_match_pass.py::pre_grad_custom_pass_fuc():11</code> 和 <code>post_grad_custom_pass_fuc():15</code> 把 NPU custom pass 注入这些槽位。

### 阶段 5：pattern 是怎样匹配和替换的

核心数据结构：

- <code>PatternMatcherPass.patterns</code>：以 <code>(node.op, target)</code> 为键保存 PatternEntry。
- <code>LoweringPatternEntry</code>：匹配后插入带 <code>_inductor_lowering_function</code> 的 handler 节点，见 <code>pattern_matcher.py::LoweringPatternEntry.apply():1377</code>。
- <code>GraphPatternEntry</code>：handler 直接修改 FX graph，见 <code>pattern_matcher.py::GraphPatternEntry.apply():1398</code>。
- <code>ReplacementPatternEntry</code>：用已 trace 的 replacement graph 替换匹配子图。

实际匹配函数：<code>torch/_inductor/pattern_matcher.py::PatternMatcherPass.apply():2605</code>。

对每个候选节点，它依次：

1. 根据 node op/target 取候选 entry；
2. 调用 <code>entry.pattern.match(node)</code>；
3. 检查 mutation region 和 stream/mempool 边界；
4. 调用 <code>entry.extra_check(match)</code>；
5. 调用 <code>entry.apply(...)</code> 修改图；
6. 更新 <code>counters[backend][pattern_matcher_count]</code> 和节点数。

所以一个可靠的“触发成功”至少需要 pattern 计数和变换后图互相印证。只有一个全局 pattern count，而没有目标节点变化，不能归因给指定 pass。

### 阶段 6：GraphLowering 把 FX 变成 Inductor IR

构造入口：<code>torch/_inductor/compile_fx.py::_compile_fx_inner()</code> 在 <code>1625</code> 附近创建 <code>GraphLowering</code>。

执行：

- <code>torch/_inductor/graph.py::GraphLowering.run_node():1928</code> 逐个解释 FX node。
- <code>torch/_inductor/graph.py::GraphLowering.call_function():1405</code> 从全局 lowerings 查找目标。
- pattern 产生的 passthrough lowering handler 带有 <code>_inductor_lowering_function</code>，可直接调用。
- 若目标没有 lowering，代码根据 fallback allowlist、decomposition 和 implicit fallback 配置选择保底或抛出缺失算子错误。

输出：Inductor IR buffers、operations、extern nodes、layouts 和依赖。

修改建议：

- Aten op 已存在但 NPU choice 不合理：改 torch_npu lowering/kernel choice。
- 通用 op 的设备无关 lowering 缺失：优先讨论 PyTorch 上游。
- 只是图变换 gate 阻止：不要先改 lowering。

### 阶段 7：NPU scheduler、codegen 和 wrapper

<code>torch_npu/_inductor/__init__.py::_load_triton_backend():71</code> 调用：

<code>register_backend_for_device("npu", NPUCombinedScheduling, NPUPythonWrapperCodeGen, CppWrapperNpu)</code>

对应源码：

- <code>torch_npu/_inductor/codegen/npu_combined_scheduling.py::NPUCombinedScheduling:25</code>
- <code>torch_npu/_inductor/codegen/wrapper.py::NPUPythonWrapperCodeGen:199</code>
- <code>torch_npu/_inductor/codegen/cpp_wrapper_npu.py::CppWrapperNpu:190</code>

<code>torch/_inductor/graph.py::GraphLowering.codegen():2994</code> 创建 Scheduler、调用 <code>Scheduler.codegen()</code>，再让 wrapper 生成最终代码。<code>compile_to_module():3053</code> 将 wrapper 编译/加载为可调用模块，并记录 output code 路径。

最终输出可能是：

- NPU Triton kernel；
- ATen/vendor extern kernel；
- CATLASS/AscendC/MLIR/DVM 生成路径；
- fallback。

所以“没有看到 Triton kernel”不等于失败。对 GEMM/attention，vendor extern 可能就是正确且更快的实现。

### 阶段 8：审计证据如何回填矩阵

<code>run_p0_gate_probe.py::worker():424</code> 的顺序是：

1. 记录环境与设备可见性；
2. 构造 eager reference；
3. 安装 observer/pattern instrumentation；
4. <code>torch.compile(..., fullgraph=True)</code>；
5. 首次编译并同步 NPU；
6. eager/compiled 数值比较；
7. 保存 counter delta、observer、active config、峰值内存和 debug root；
8. benchmark 模式下采集 warmup 后的 mean/stdev/p50/p99。

注意：worker 将状态写成 <code>compile-correct</code>，源码注释明确说明这不代表目标 pass 已触发。目标触发由 transformed FX graph、Inductor IR 和 <code>output_code.py</code> 再判定。

### P0 源码案例一：mm_plus_mm

调用链：

~~~text
post_grad pass_patterns
  -> post_grad.py::is_valid_mm_plus_mm()
  -> post_grad.py::mm_plus_mm()
  -> kernel/mm_plus_mm.py::tuned_mm_plus_mm()
  -> algorithm choices / extern or Triton template
~~~

关键 gate：<code>is_valid_mm_plus_mm():979</code> 要求 max autotune、四个 meta value、两个合法 matmul 和相同输出 M/N。

NPU 分化：

- default backend 调用 <code>torch_npu/_inductor/fx_passes/post_grad.py::patch_pattern_mm_plus_mm():24</code>，只对 NPU 使 extra_check 返回 false。
- triton_experimental 在加载前恢复 Inductor baseline，并能进入上游 <code>tuned_mm_plus_mm</code>。
- T-023 没有解除共享 gate，而是在 default backend 另行注册 NPU-only different-K handler；它只在环境开关启用、static positive 2D、K1!=K2、同 dtype/device、受支持 stride 和 output 上限内追加 template choice，extern 始终保留。

实测结论见 [P0 功能报告](report/p0_gate_first_run_20260820.md#mm_plus_mm)、[性能 A/B](report/p0_ab_first_shape_20260820.md#mm_plus_mm--triton_experimental) 与 [T-023 集成报告](report/t023_mmplus_different_k_integration_20260821.md)。

### P0 源码案例二：pad_mm / pad_bmm / pad_addmm

注册链：

~~~text
pad_mm.py pattern/replacement family
  -> gen_register_replacement()
  -> should_pad_mm / should_pad_bmm / should_pad_addmm
  -> should_pad()
  -> can_pad()
  -> check_device()
~~~

根因位于 <code>torch/_inductor/fx_passes/pad_mm.py::check_device():80</code>：

~~~python
return (a.is_cuda and b.is_cuda) or (a.is_xpu and b.is_xpu)
~~~

NPU 在 <code>can_pad():97</code> 就被拒绝，尚未进入 <code>force_shape_pad</code> 和性能启发式。因此把 config 设为 true 仍不触发，不是“启发式认为不划算”。

experimental backend 还通过 <code>torch_npu/_inductor/triton_experimental/overrides.py::_disable_pad_mm_pass():74</code> 把 <code>shape_padding</code> 设为 false。

真正的支持评估必须比较：

~~~text
原始 mm/bmm/addmm
vs
constant_pad_nd -> mm/bmm/addmm -> slice
~~~

若 padding、额外读写和 slice 抵消了 GEMM 对齐收益，正确结论可能是保持关闭，而不是实现一个 Triton 替身。

T-025/T-026 已实际完成这组比较。positive 图和 aligned negative 的功能gate全部通过，但mm/bmm/addmm每步设备task分别从1增至5/5/7，三轮p50均明显回退，峰值allocated还多约272–288 KiB。因此当前device gate不仅是未适配边界，也避免了代表shape上的真实性能回退。详见 [pad family报告](report/t025_t026_pad_family_20260821.md)。

### P0 源码案例三：addmm fusion

上游 gate：<code>torch/_inductor/fx_passes/post_grad.py::should_prefer_unfused_addmm()</code> 位于 1850 行附近，检查 device、dtype、bias 形状和用户结构。

NPU lowering：<code>torch_npu/_inductor/kernel/mm.py::_register_npu_inductor_addmm():126</code> 注册 <code>aten.addmm</code> lowering，根据静态性、非零 shape、连续性和 CATLASS 能力选择实现；条件不满足时使用 fallback handler。

experimental 控制：<code>torch_npu/_inductor/triton_experimental/fx_passes.py::_disable_addmm_fusion_pass():534</code> 将已注册 addmm pattern 的 extra_check 置 false。

因此本候选不是“缺少 addmm 算子”，而是要验证 fusion gate 与现有 NPU lowering 的组合是否在不同输入上正确且有收益。

### 完整调用链总结

~~~mermaid
sequenceDiagram
    participant U as 用户函数
    participant TC as torch.compile
    participant D as TorchDynamo
    participant NW as NPU backend scope
    participant FX as compile_fx/AOTAutograd
    participant P as FX passes
    participant PM as PatternMatcher
    participant GL as GraphLowering
    participant S as Scheduler
    participant CG as NPU Codegen
    participant R as NPU Runtime

    U->>TC: model, fullgraph, options.npu_backend
    TC->>NW: 初始化 Inductor wrapper
    NW->>NW: 解析并加载 backend
    TC->>D: optimize(backend wrapper)
    D->>FX: GraphModule + example inputs
    FX->>P: pre_grad
    FX->>P: joint_graph
    FX->>P: post_grad
    P->>PM: apply registered entries
    PM-->>P: transformed GraphModule + counters
    FX->>GL: GraphModule
    GL->>GL: lookup lowerings / fallback
    GL->>S: Inductor IR operations
    S->>CG: fused/scheduled nodes
    CG-->>FX: compiled module + output code
    FX-->>D: callable
    D-->>U: cached compiled result
    U->>R: 执行 NPU kernel/extern
~~~

## 扩展点分析

### 可扩展点

| 扩展目标 | 修改位置 | 推荐方式 |
|---|---|---|
| 新增跨设备都成立的通用 FX pattern | <code>src/pytorch/torch/_inductor/fx_passes/</code> | 使用现有 registry 与 extra_check，补 CPU/CUDA/XPU/NPU 语义审查 |
| 只为 NPU 放宽或收紧上游 pattern | <code>src/torch_npu/torch_npu/_inductor/fx_passes/</code> | 用 NPU capability predicate，不直接污染通用语义 |
| 新增 NPU custom pass | <code>ascend_custom_passes/</code> | 使用 <code>@register_custom_pass(PassType, FxPassLevel)</code> |
| 控制 NPU custom pass | <code>register_custom_pass.py::_get_shut_down_pass_set():21</code> | 记录 <code>SHUT_DOWN_FX_PASS_LIST</code>，测试 pass-on/pass-off |
| 修改 mm/addmm choice | <code>torch_npu/_inductor/kernel/mm.py</code> | 优先复用 ATen/CATLASS/extern choice 和 algorithm selector |
| 新增 Aten 到 NPU lowering | <code>torch_npu/_inductor/lowering.py</code> 或 <code>kernel/</code> | 注册 lowering，明确 fallback 和 dtype/layout capability |
| 修改 NPU fusion/scheduling | <code>torch_npu/_inductor/scheduler.py</code>、<code>codegen/</code> | 保持依赖、stream、memory 和 wrapper 契约 |
| 新增 DVM/MLIR 变换 | <code>torch_npu/_inductor/dvm/</code>、<code>ascend_npu_ir/</code> | 先分结构层与后端层验证 |
| 手写 Triton kernel | 优先放 torch_npu kernel/实验接入层 | 先证明现有 vendor/CATLASS/AscendC 不足 |
| 修改 Triton Ascend 编译器 | <code>src/triton-ascend/</code> | 仅在 Triton 语言或 Ascend backend 本身存在缺陷时 |
| 扩展审计 harness | <code>inductor_pass_npu_audit/</code> | 主进程不导入 torch，每个 backend fresh worker |

### 修改已有行为

#### 修改一个 pattern

先定位：

1. registry 中的 pattern/handler；
2. extra_check；
3. pass number 与执行阶段；
4. replacement 输出节点；
5. 下游 lowering。

修改后至少增加：

- 一个真正匹配的正例；
- 一个形状相似但不应匹配的负例；
- dtype/shape/stride/alias 或 backward 边界；
- 变换前后图和 generated code；
- pass-on/pass-off paired benchmark。

#### 新增一个 NPU custom pass

注册入口是 <code>torch_npu/_inductor/fx_passes/ascend_custom_passes/register_custom_pass.py::register_custom_pass():28</code>。执行入口是：

- <code>run_register_pre_custom_passes():13</code>；
- <code>run_register_post_custom_passes():28</code>。

源码显示多数 PRE pass 只在 inference 执行；<code>fusion_attention_v3_pass</code> 又在后续按名称额外执行。是否真的重复扫描必须用 observer 实测，不能只由循环结构直接推断性能影响。

#### 新增或修改 lowering

先检查 op 是否已经注册。以 addmm 为例，现有 <code>_register_npu_inductor_addmm():126</code> 已有 CATLASS 能力检查和 fallback；新增 Triton 前必须与这些 choice 同输入比较。

lowering 需要说明：

- 支持的 dtype/device/layout；
- 静态与动态 shape；
- 空维、非连续和广播；
- 不支持时回退到哪里；
- autotune 没有合法 choice 时的行为。

### 常见扩展模式

#### 模式一：只改 capability gate

适用于“已有 replacement 和 lowering，但 NPU 被保守 gate 拦住”。先用测试侧临时开关证明功能与收益，再把 gate 设计成后端 capability；不要先全局放开。

#### 模式二：FX 重写到 vendor op

适用于 attention、embedding 或稳定复合算子。优先复用 NPU vendor op，因为它通常已经处理 tiling、workspace 和硬件特性。<code>fusion_attention_v3_pass</code> 位于 <code>ascend_custom_passes/ascend_graph_pass.py::fusion_attention_v3_pass():916</code>。

#### 模式三：新增 Inductor template/choice

适用于 GEMM 等需要 autotune 的计算。在现有 choice 列表中加入候选，由 <code>autotune_select_algorithm</code> 选择，而不是替换掉安全 fallback。

#### 模式四：手写 Triton

适合稳定的 elementwise/reduction 融合或缺失 epilogue。GEMM、attention、collective 默认优先 vendor/CATLASS/AscendC；Triton 只有在能力与 profile 都证明合适时进入。

### 容易踩坑的地方

1. **把 compile-correct 当成 pass 可用**：必须看 transformed graph、IR 或 output code。
2. **只看 pattern matcher 总计数**：同一阶段可能有其他 pattern 匹配，必须关联目标节点。
3. **跨 backend 复用同一进程**：torch_npu loader 会改变全局 registry，结果可能串扰。
4. **缓存绕过 pass**：正式触发检查使用 <code>TORCHINDUCTOR_FORCE_DISABLE_CACHES=1</code>。
5. **在源码树目录运行测试**：会造成 torch_npu 源码级联导入；所有测试从 <code>/home/z50063656/tmp</code> 发起。
6. **混用源码 Python 与不匹配 wheel**：必须记录 <code>torch.__file__</code>、<code>torch_npu.__file__</code>、native extension 和 git version。
7. **用 eager 对 compiled 归因单个 pass**：单 pass 性能必须同 backend、pass-on/pass-off。
8. **把 skip 当 pass**：NPU 不可见、OOM 或环境失败必须单独分类。
9. **忽略 alias/mutation**：尤其是 pre-grad 的 view/clone/expand/cat 折叠。
10. **忽略动态 shape**：静态正例成功不能证明符号 shape 不会误 specialize 或错误改写。
11. **先写 Triton再找收益**：替代方案必须来自已定位缺口和 profile。
12. **清理未知工作树修改**：当前 Triton 有既有兼容修改，torch_npu 有构建生成物；不得擅自回退。

### 本任务下一步的实际执行计划

#### 已完成：P0 addmm 覆盖与通用 blocker

上述 dtype、shape、layout、dynamic、bias/shape guard、forward/backward 和三轮交错 paired A/B 已完成。T-011 还展示了一个重要工程方法：pass 本身已经正确改图，但训练路径可被下游公共 reduction lowering 阻断；应修复真实跨切面接口，而不是修改 addmm pattern 或手写 Triton 绕开。源码改动仅在 `torch_npu/_inductor/lowering.py`，新 wheel 以 `--no-deps` 安装，原 blocker 与近邻回归通过。

#### 已完成：mm_plus_mm different-K 从定位到条件性集成

different-K 能匹配 post-grad pattern，但原 lowering 会安全退回两个 mm 加 add。T-012/T-013 先证明 fallback 本身 neutral，并拆出两个 aclnnMm、一个 Triton add和 launch gap。T-014 至 T-020 完成独立 K1/K2 Triton 微原型、单 task、三 dtype/真实转置/dynamic/backward 和扩展性能。T-021/T-022 完成正式设计与 large hold。T-023 随后把 NPU-only、default-off、ATen extern fallback-first template接入源码构建 wheel：首批功能通过，shape-A/unaligned集成 p50 改善15.29%/18.04%；allocator trace证明每 block 64 KiB、grid 6 的 Ascend Triton workspace使 candidate比 baseline多270,336 B peak allocated。T-024 又筛选七种 option/tile/grouped-program配置，没有同时通过显存和 10 μs task gate。因此最终是条件性有益，不默认开启。

#### 第一步：在匹配 headers 的环境关闭 T-023 无 shim 复验

产品源码和修复版 wheel 已完成，不要重复实现。当前 editable PyTorch 源码树缺少 launcher include view、PyTorch headers要求C++20、Triton launcher固定C++17、torch_npu headers与CANN 9.0.1 conditional graph类型不齐；审计垫片只能提供 device证据，不能作为产品环境。环境支线只需在版本/header合同匹配的独立环境安装同 wheel，重跑 rollout-on fresh compile、两 cohort correctness和single-task smoke。默认关闭、131072 output上限和fallback保持不变。

#### 已完成：pad family capability 与性能判定

测试侧受控capability绕过没有修改产品源码：三family正例真实变换且正确，aligned负例不误触发；但三轮配对性能和显存均失败。因此P-002已关闭为`capability-available-performance-rejected`，保持gate且不手写Triton padding。

#### 当前主线：执行 P1 的 66 条记录

T-027 已先把7个B2 custom pass的device-independent结构UT跑到32/32；T-028又确认首个`fold_reduce` positive在真实default-backend NPU compile中把sum节点从1消到0且数值正确。注意它仍只有positive功能证据，不能跳过negative和单pass paired性能直接写`supported`。

按 [p1_batch_design.md](p1_batch_design.md) 顺序：

1. B2 的冗余/恒等消除结构 UT；
2. layout/copy pass 的 NPU generated code；
3. dtype/index/mask 的极值和 dynamic shape；
4. 三个复合融合的 vendor op 与单 pass 性能；
5. DVM/MLIR 的结构层和后端层；
6. attention 先跑 1、5、13、18、21、28、29 七个代表 family，再扩到 30 个。

#### 后续：只对真实缺口实施替代

优先级：

~~~text
修正 gate
  -> 复用已有 NPU lowering/vendor op
  -> CATLASS 或 AscendC
  -> NPU 专用 Inductor template
  -> 手写 Triton
  -> 修改 Triton Ascend 编译器
~~~

这不是绝对的性能排序，而是维护成本与定位边界。最终选择仍由同机 paired benchmark 决定。

## 总结

### 1. 核心工作机制

Inductor pass 在 FX 图上做变换；pattern matcher 负责结构匹配与 capability gate；GraphLowering 把变换结果转为 IR；scheduler/codegen 决定内核和 wrapper；torch_npu 通过 backend loader、custom pass、lowering 和 codegen registry 接入 NPU。

### 2. 核心调用路径

~~~text
torch.compile
  -> _TorchCompileInductorWrapper
  -> torch._dynamo.optimize
  -> compile_fx / AOTAutograd
  -> pre_grad
  -> joint_graph
  -> post_grad
  -> PatternMatcherPass
  -> GraphLowering
  -> Scheduler
  -> NPU codegen / extern / fallback
~~~

### 3. 核心数据结构

- FX <code>GraphModule</code>：pass 的主要输入输出。
- <code>PatternMatcherPass.patterns</code>：pattern registry。
- <code>PatternEntry</code> 子类：定义匹配后的处理方式。
- node <code>meta["val"]</code>：device、dtype、shape、stride 等 gate 信息。
- Inductor <code>lowerings</code>：Aten target 到 lowering handler 的映射。
- backend <code>device_codegens</code>：设备到 Scheduling/Wrapper 的映射。
- 评估矩阵：静态记录、动态证据和最终 verdict 的项目数据模型。

### 4. 推荐源码阅读顺序

1. <code>torch/__init__.py::compile()</code> 与 <code>_TorchCompileInductorWrapper</code>。
2. <code>torch/_inductor/compile_fx.py</code> 的递归 pass 入口和 <code>compile_fx_inner</code>。
3. <code>pre_grad.py</code>、<code>joint_graph.py</code>、<code>post_grad.py</code> 三个主函数。
4. <code>pattern_matcher.py::PatternMatcherPass.apply()</code> 和三个 PatternEntry 子类。
5. <code>graph.py::GraphLowering.call_function()</code>、<code>run_node()</code>、<code>codegen()</code>。
6. <code>torch_npu/utils/_dynamo.py</code> 的 backend scope。
7. <code>torch_npu/_inductor/__init__.py</code> 的 backend loader。
8. 选择一个真实候选，沿 pattern→gate→lowering→codegen→报告完整走一遍。

### 5. 建议的三个入门练习

1. **读图练习**：运行 addmm 正例，比较 eager FX、post-grad transformed graph 和 output code，指出融合发生在哪一层。
2. **gate 练习**：从 <code>pad_mm.py::check_device()</code> 向上追到 pattern 注册，向下追到 replacement，解释为什么 <code>force_shape_pad=True</code> 仍无效。
3. **性能练习**：阅读 addmm 三轮 A/B JSON，自己计算 p50 中位数和收益比例，并解释为什么 eager-vs-compiled 不能作为单 pass baseline。

这三个练习对应的项目证据已经形成，适合作为新接手者的复盘入口。复盘后阅读 [暂停检查点](PAUSED_CHECKPOINT_20260821.md)、T-012/T-013、[T-014–T-016 报告](report/t014_t016_mmplus_different_k_candidate_20260821.md)、[T-017–T-019 覆盖报告](report/t017_t019_mmplus_different_k_coverage_20260821.md)、[T-020 扩展性能报告](report/t020_mmplus_different_k_extended_benchmark_20260821.md)、[T-021 正式接入设计](report/t021_mmplus_different_k_integration_design_20260821.md)、[T-022 large 分解报告](report/t022_mmplus_different_k_large_profile_20260821.md)、[T-023 集成报告](report/t023_mmplus_different_k_integration_20260821.md)和[T-024 workspace 审计](report/t024_mmplus_different_k_workspace_20260821.md)。当前主线是 pad family；T-023 只剩匹配环境的无 shim复验，不要重新执行已经闭环的 P0 addmm 或 T-014 至 T-024 基础矩阵。
