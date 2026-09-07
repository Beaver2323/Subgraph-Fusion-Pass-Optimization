# NPU 功能验证、诊断与最小修复实操指南

> 更新时间：2026-09-07 11:20 CST（UTC+08:00）
>
> 适用范围：本仓库 acceptance unit 在 Ascend NPU 上的功能验证、差异分类、最小修复和回归
>
> 固定后端：`triton_experimental`；除非单个任务由用户明确指定其他后端

这份文档补齐此前散落在各 case 复现报告中的过程。它回答五个问题：社区原生入口是否真的执行、
目标优化是否命中并生效、NPU 数值/梯度是否正确、差异属于产品边界还是回归、修复后如何证明没有
破坏相邻合同。它不是“所有失败都要修”的指南：明确产品关闭项只记录，不绕过 gate 制造 ON 路径。

## 1. 一条完整工作线

```text
GPU reference 已冻结
  → NPU 原生社区入口
  → 判断 test body 是否真实执行
  → 必要时做测试入口最小适配
  → 验证 backend、target counter、FX、codegen、正确性
  → 定位首个分歧层
  → 分类：一致 / 产品预期分歧 / capability pending / NPU regression
  → 仅对 regression 或经评审的 capability 缺口做最小修复
  → 原 case + 全 variants + 邻近用例回归
  → 写 NPU result 与 GPU/NPU comparison
  → 功能门禁通过后才进入同 backend OFF/ON 性能
```

“进入测试函数”“pattern 匹配”“FX 被改写”“lowering 成功”“模板最终获选”是不同层级，不能只凭
数值 PASS 或一个 counter 把它们合并成“优化生效”。

## 2. 固定环境与启动方式

所有测试必须从 `/home/z50063656/tmp` 发起，不能在 PyTorch/torch_npu 源码目录中 import `torch`。
后端选择具有进程级生命周期，必须在 Python 导入前设置；OFF/ON 和跨 backend 必须使用新进程。

```bash
cd /home/z50063656/tmp

source /home/z50063656/Pass/activate_pass.sh

export ASCEND_RT_VISIBLE_DEVICES=0
export SET_NPU_DEVICE=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1
```

运行前用 `npu-smi info` 确认物理卡和进程。功能测试允许按项目约定共享时，应记录共享状态；性能测试
必须遵循对应 performance plan 的设备隔离要求。

每轮使用独立 artifact/cache 目录：

```bash
export RUN_ID
RUN_ID="npu-$(date '+%Y%m%dT%H%M%S%z')"

export RUN_ROOT="/home/z50063656/tmp/pass-npu-results/${RUN_ID}"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="${RUN_ROOT}/torch-compile-debug"
export TORCH_TRACE="${RUN_ROOT}/torch-trace"
export TORCHINDUCTOR_CACHE_DIR="${RUN_ROOT}/inductor-cache"
export TRITON_CACHE_DIR="${RUN_ROOT}/triton-cache"

mkdir -p "${RUN_ROOT}"
```

## 3. 第一步：先跑社区原生入口

原生入口的作用是确认上游测试能否在当前环境自然进入 test body。以 T-076 mm-plus-mm 为例：

```bash
python /home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py \
  TestPatternMatcher.test_mm_plus_mm \
  >"${RUN_ROOT}/native.stdout.log" \
  2>"${RUN_ROOT}/native.stderr.log"

echo "$?" >"${RUN_ROOT}/native.return_code"
```

不能把退出码 0 自动记成 PASS。必须同时检查测试统计：

- `Ran N tests` 且 `N>0`：测试体真实执行；
- 退出 0 但没有测试统计或 `Ran 0 tests`：记 `NO_TESTS`；
- `skipped`：保留 skip 原因，不能改写成 PASS；
- 导入/设备/依赖失败：记 `ENV_BLOCKED`，没有产品行为结论。

T-076/T-077 多数入口受上游 `HAS_GPU`、`GPU_TYPE` 或动态测试类生成条件阻断。这个结果只说明
测试路由未覆盖 PrivateUse1/NPU，不说明 pattern 在 NPU 上失败。

## 4. 第二步：原生阻断后才做测试入口最小适配

最小适配的目标是让“同一个社区合同”进入 NPU，而不是重写一个更容易通过的新测试。允许的变化：

- 设备从 CUDA/GPU_TYPE 改为 `npu`；
- 显式传入 `options={"npu_backend": "triton_experimental"}`；
- 绕过只负责生成 GPU 测试类的 harness；
- 捕获 target counter、FX、generated code、数值、梯度和环境；
- GPU-only GEMM choice 在产品 gate 关闭后无合法路径时，经记录后补 ATEN fallback choice。

禁止的变化：

- 改 shape、dtype、stride、dynamic、forward/backward 合同后仍称原例；
- 删除负例或放宽 pattern guard；
- 临时关闭 NPU 明确产品 disable 来制造命中；
- 使用 default、DVM、MLIR 的结果替代 `triton_experimental`；
- CPU fallback 后只看输出正确就称 NPU 图模式 PASS。

已有适配器以 case 为单位放在 `issues/<case-id>/npu_adapter.py`，并强制配套
`issues/<case-id>/适配报告.md`。统一字段、代码框、调用链和边界要求见
[NPU 最小适配报告规范](ADAPTER_REPORT_STANDARD.md)。例如：

```bash
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/\
issues/REF-mm-plus-mm-native/npu_adapter.py \
  --pytorch-root /home/z50063656/Pass/src/pytorch \
  --artifact-dir "${RUN_ROOT}/adapter-artifacts"
```

适配器不是产品修复。它只修复“测试怎么进入 NPU”的问题，必须在结果中列出
`adapter_deviation` 和 `product_gate_bypassed=false`。

## 5. 第三步：证明功能、命中和生效

每个正例至少收集以下五类证据；负例则证明 target counter 为零或目标结构没有出现：

| 证据 | 回答的问题 | 常用来源 |
| --- | --- | --- |
| backend 验真 | 是否真在 `triton_experimental` | `_InductorNpuRegistry._loaded_backend` |
| target counter/marker | matcher/handler 是否触发 | `torch._dynamo.utils.counters` |
| transformed FX | replacement/decomposition 是否改变图 | `fx_graph_transformed.py` |
| generated code/IR | lowering、fallback、模板走了哪条路径 | `output_code.py`、IR debug |
| correctness | 数值、dtype、stride、梯度或统计合同是否成立 | 社区断言及 adapter result |

后端验真应在 backend 注册后执行：

```python
# 适配器内；环境变量必须在 import torch/torch_npu 前由进程设置
from torch_npu.utils._dynamo import _InductorNpuRegistry

assert _InductorNpuRegistry._loaded_backend == "triton_experimental"
```

### 5.1 如何区分各层级

- `PATTERN_MATCHED`：counter/handler 触发；handler 仍可能因额外条件返回原图。
- `REWRITE_APPLIED`：transformed FX 已出现 replacement，或 GEMM 被分解为
  `unsqueeze/mul/sum`；发生在 lowering/autotune 之前。
- `LOWERING_SELECTED`：FX op 已映射到 NPU Inductor IR 或 extern 路径。
- `TEMPLATE_CANDIDATE`：候选进入 autotune，不代表最终获选。
- `TEMPLATE_SELECTED`：生成代码实际调用目标模板。

T-077 B2B 是典型例子：最小 capability 探针能让融合候选进入 autotune，但当前 NPU 上最终仍选择
unoptimized fallback，因此不能写成“B2B 融合已生效”。decompose-BMM/MM/addmm 则在 FX 阶段已经
完成分解，但实测性能回退，所以最终保留 device guard。

## 6. 第四步：沿编译栈定位首个分歧

按下列顺序定位，找到第一个与 GPU contract 不一致的层后再决定修复位置：

1. 测试发现/实例化：是否 `NO_TESTS`；
2. backend 注册：实际 backend 是否正确；
3. product/config gate：是否存在明确 NPU disable；
4. pattern extra check：shape、dtype、device、用户关系等 guard；
5. replacement/handler：是否真正改变 FX；
6. lowering：ATen op 是否进入正确 NPU IR/extern handler；
7. codegen/template/autotune：候选是否能生成并最终获选；
8. runtime：kernel 启动、同步、内存与设备异常；
9. correctness：forward、两路梯度、dtype、stride、统计分布；
10. performance：只在前九层通过后分析。

分类规则：

- `BEHAVIOR_UNCHANGED`：GPU/NPU 的正例和负例合同一致；
- `EXPECTED_PRODUCT_DIVERGENCE`：有明确 NPU 产品 gate/device policy，原图 fallback 正确；
- `CAPABILITY_PENDING`：只有 generic cuda/xpu guard，没有明确 NPU 产品 disable；需单独评审探针；
- `NPU_REGRESSION`：没有合法产品边界，NPU 在同一合同上出现错误或异常。

## 7. 第五步：只做最小、分层修复

修复位置必须与首个分歧层一致：

- 只是不生成 NPU 测试类：修 tracker adapter，不改 PyTorch pass；
- NPU helper API 不兼容：修 `triton_experimental` helper；
- NPU lowering 错误：修 NPU override/lowering，不改上游 pattern guard；
- generic device guard 且探针正确、有收益：再评审是否扩展产品支持；
- 明确产品 disable：不修、不测 ON，只记录免测证据。

T-077 decompose-MM 的真实回归展示了这种分层。目标 post-grad pass 对 M=2048 负例正确地没有分解，
错误发生在之后的 small-mm pointwise lowering：前向接近正确，但左梯度 4085/4096 元素不一致。
最小候选仅在 NPU 禁用该错误 heuristic，让它回到可靠 extern mm：

```python
# torch_npu/_inductor/triton_experimental/overrides.py
def npu_safe_small_mm_pointwise(m, k, n, layout):
    if layout.device.type == "npu":
        return False
    return original_small_mm_pointwise(m, k, n, layout)
```

补丁副本位于 `issues/REF-decompose-mm-native/backend_fix_dfbcc25.patch`。它没有改变 decompose-MM
的 M/K/N 条件，也没有为了性能强行启用该 pass。

## 8. 第六步：修复验证必须覆盖什么

至少包含：

1. 原失败输入修复前失败、修复后通过；
2. 同 acceptance unit 的所有正例、负例和 regression variants；
3. forward、需要时的左右梯度；
4. target counter/FX/codegen 路径没有被意外改变；
5. NPU-only 修复不改变其他设备逻辑；
6. fresh process 和 fresh cache；
7. 安装态或候选态来源、commit/patch SHA 和环境时间戳。

decompose-MM 的修复验证命令示例：

```bash
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/\
issues/REF-decompose-mm-native/npu_adapter.py \
  --artifact-dir "${RUN_ROOT}/decompose-mm-fixed" \
  --apply-candidate-small-mm-guard
```

该适配器执行 fp32/mixed 的正例、K 阈值负例、M 阈值负例共六个 variants，并分别核对 forward、
left-grad 和 right-grad。修复只通过原失败点而没有跑负例/邻近合同，不能标为 `verified`。

## 9. 第七步：结构化结果与 comparison

完成动态执行后，应更新：

- `results/current/<acceptance-unit>/npu_result.json`：真实 NPU 环境、输入、counter、FX/codegen、
  correctness、adapter 与修复信息；
- `results/current/<acceptance-unit>/comparison_result.json`：逐 variant 的 GPU/NPU 行为、首个分歧与
  verdict；
- `issues/<case-id>/复现报告.md`：可复现命令、原始阻断和动态证据；
- `issues/<case-id>/适配报告.md`：最小偏差、代码框、必要调用链、社区合同保持性与适配边界；
- 若修源码，再补 `根因分析.md`、`修复验证报告.md` 和 `代码合入描述.md`；
- task 级 report/guide：汇总每个 pattern 的代码意图、GPU/NPU 对照与性能处置。

零设备校验从规定目录运行：

```bash
cd /home/z50063656/tmp

python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/\
scripts/validate_comparison_data.py

python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/\
scripts/generate_current_acceptance_matrix.py \
  --check
```

### 9.1 修复报告的强制代码与调用栈合同

只要修改了产品源码，`根因分析.md` 和 `修复验证报告.md` 不能只写“某 guard/helper 有问题”。每份报告
至少必须包含：

1. 触发输入与原测试核心代码框，标出仓库相对路径和起始行；
2. 原实现中决定行为的 pattern/guard/lowering 代码框；
3. 从测试入口到失败断言或异常点的最短必要调用栈；
4. 首个分歧层及其动态证据，区分实际 traceback 与源码重建调用链；
5. 完整修复代码框、文件位置和为何是最小修复；
6. 修复前后 FX/IR/generated-code/runtime 路径；
7. 原失败、全部 variants、邻近用例和 backend 专属 UT 的结果表；
8. 可直接执行的复验命令，以及 artifact 路径、commit 和 SHA256。

对数值错误，如果没有异常 traceback，必须写“无 exception traceback”，再用源码注册关系、FX、IR、
generated code 和断言位置重建必要调用链；禁止把推断伪装成采集到的堆栈。完整范例：

- `issues/REF-addmm-contract-native/根因分析.md` 与 `修复验证报告.md`：产品 gate/capability 候选；
- `issues/REF-decompose-mm-native/根因分析.md`；
- `issues/REF-decompose-mm-native/修复验证报告.md`。

## 10. 第八步：何时允许性能测试

性能只有在以下条件同时满足时解锁：

- `triton_experimental` backend 已验真；
- OFF/ON 两臂都能正确运行；
- ON 的 target counter/FX 证明目标优化真实生效；
- OFF 只关闭目标 pass，不关闭整个 pipeline；
- 正负例和修复回归通过；
- 没有明确 NPU 产品 disable。

采用 fresh-process 交错顺序，例如 `OFF1→ON1→ON2→OFF2→OFF3→ON3`，记录 host p50/p99、NPU Event
p50/p99、峰值内存、task/kernel 数和每轮原始哈希。社区有 benchmark 时优先复用其 shape、输入合同和
方法；没有时才从社区功能正例派生，并明确标注来源。

已完成任务可使用：

```bash
cd /home/z50063656/tmp

bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/\
scripts/run_npu_performance_task.sh \
  --task T-077 \
  --unit gumbel \
  --npu 0
```

T-079 已完成上述全流程，可使用 `scripts/run_t079_npu_all.sh` 与
`scripts/run_t079_performance.sh`；`bmm→mm` 的性能回退另由
`scripts/verify_t079_bmm_gate.sh` 验证最终产品门禁。T-080 也已完成：
`scripts/run_t080_npu_all.sh NPU_ID function` 一键复验 Scatter 最终门禁、Softmax 产品 fallback 和
Constructor 功能；`performance` 模式只校验已冻结性能处置，不会绕过 Scatter/Softmax 产品关闭制造 ON 路径。
完整代码、调用链、适配、修复和 GPU/NPU 对照见 `docs/T080_RESULT_AND_LEARNING_GUIDE.md`。

## 11. 从哪里学习已有真实案例

| 主题 | 推荐入口 |
| --- | --- |
| 原生 0 tests 与最小入口适配 | `issues/REF-mm-plus-mm-native/复现报告.md` |
| 明确产品 gate，不绕过关闭项 | `issues/REF-pad-mm-dynamic-m-native/复现报告.md` |
| pattern 命中、lowering 回退的区别 | `report/t076_pattern_gpu_npu_guide_20260902.md` |
| decomposition 与 autotune/性能的区别 | `report/t077_pattern_gpu_npu_guide_20260902.md` |
| lowering correctness 回归与最小修复 | `issues/REF-decompose-mm-native/根因分析.md`、`修复验证报告.md` |
| helper API 根因、修复和邻近验证 | `issues/REF-partial-reuse-positive-native/根因分析.md` |
| 完整的复现/根因/验证/合入四件套 | `issues/REF-addcdiv-fma-codegen-native/` |
| GPU rewrite 在 NPU 功能正确但性能回退的产品门禁 | `issues/REF-bmm-to-mm-native/` |

阅读任何历史结论前，都要核对 source revision、输入合同、backend、gate 生命周期和测量方法。只要
其中一项不同，就只能作为非计数历史证据，不能迁移成当前 verdict。
