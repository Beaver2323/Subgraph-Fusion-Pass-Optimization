# T-080 功能与性能测例讲解

> 更新时间：2026-09-07 20:41 CST（UTC+08:00）
> 状态：本页保留冻结时的测例设计；GPU/NPU、性能与产品处置均已完成。
> NPU 固定后端：`triton_experimental`；正式结论见
> [`T080_RESULT_AND_LEARNING_GUIDE.md`](T080_RESULT_AND_LEARNING_GUIDE.md)。

NPU 的逐步操作、证据判定、最小适配和修复门禁统一见
[`NPU_FUNCTION_REPAIR_WORKFLOW.md`](NPU_FUNCTION_REPAIR_WORKFLOW.md)。本页解释 T-080 各单元合同，
逐单元执行合同见 [`T079_T080_NPU_FUNCTION_REPAIR_PLAN.md`](T079_T080_NPU_FUNCTION_REPAIR_PLAN.md)；
正式 worker、结果与门禁在结果指南中导航。

下文按单元讲解功能测例、性能测例及其证据边界。

## 1. 常量 full 上的 scatter 改为 pointwise（`AU-joint-graph-scatter-upon-const-tensor`）

代码位置：`torch/_inductor/fx_passes/joint_graph.py:1150-1235`。

```python
# torch/_inductor/fx_passes/joint_graph.py:1150-1235
def scatter_upon_const_tensor_extra_check(match):
    return selector_ft.size(dim) == 1

def scatter_upon_const_tensor(match, shape, background_val, dtype, dim, selector, val):
    metrics.num_matches_for_scatter_upon_const_tensor += 1
    # broadcast index 后用 where 选择 val 或 background_val
```

功能意图：原图先物化完整常量张量，再做稀疏 scalar scatter；改写后直接按 selector 用 pointwise
`where` 生成结果，避免大张量 mutation。8 个社区 cases 覆盖：3D 最后维、非最后维、负 dim 正例；
index 太短、selector 太密、base 非常量负例；FP16/BF16 dtype；以及 CrossEntropy backward 端到端。

GPU 实测 8/8 通过。正例在 joint-graph 后的 FX 中形成 `iota/eq/where`，负例保留 scatter/copy；
低精度 dtype 与 CrossEntropy backward 原生断言均通过。由于 joint-graph 改写早于当前 debug 捕获点，
部分正例 before/after 同形，命中应结合社区 target metric 与访存断言判定。后续 NPU 候选功能已
验证，但完整社区图性能回退并增加显存，产品最终默认关闭，不能只凭候选命中宣称默认生效。

性能测例直接来自社区 `test_cross_entropy_loss` 的 `DO_PERF_TEST=1` 分支：
`B=32,T=1024,D=768,V=50257`，执行 linear→cross_entropy→backward。社区设计同时测设备时间和
峰值内存，A100 注释数据只能作来源说明，不能迁移为 NPU 结论。NPU OFF/ON 使用
`optimize_scatter_upon_const_tensor=false/true`；若全量 shape 资源不足，应记 blocked，不能静默缩小
后仍称“社区 benchmark”。

## 2. prepare_softmax 改为 online softmax primitive（`AU-post-grad-prepare-softmax`）

代码位置：`torch/_inductor/fx_passes/post_grad.py:490-520`。

```python
# torch/_inductor/fx_passes/post_grad.py:490-514
def prepare_softmax_pattern(x, dim):
    xmax = x.amax(dim=dim, keepdim=True)
    xsub = x - xmax
    xexp = xsub.exp()
    return xmax, xexp.sum(dim=dim, keepdim=True), xsub, xexp

def prepare_softmax_replacement(x, dim):
    xmax, xsum = prepare_softmax_online(x, dim)
    ...
```

功能意图：识别 max/sub/exp/sum 的 softmax 前处理，让 max 与 sum 通过 online primitive 合并遍历。
三个功能入口分别验证 fast-math codegen、strict signed-zero 回归和社区 perf 合同。signed-zero case
不是“性能负例”，而是防止优化改变 IEEE 符号零语义的 correctness 门禁。

GPU 实测 3/3 通过，FX 明确显示 `amax/sub/exp/sum` 被替换为
`prims.prepare_softmax_online.default`。NPU 测试态 generic guard 探针也形成该 prim，但产品
`FALLBACK_LIST` 明确关闭其 lowering、generated code 无 online kernel，最终为受控差异和性能免测。

社区已经提供 BF16 `N=32768,V=50304` 的 `DO_PERF_TEST` 全量 shape，以及默认
`[1024,2048]`/`[128,128]` 诊断 shape。社区部分代码比较 eager/compiled，但 tracker 要评估目标 pass，
原计划要求比较同一 compiled `triton_experimental` 的 OFF/ON，而不是拿 eager 当 OFF。动态复核发现
产品明确 lowering fallback 后，不再绕过关闭制造 ON，按工作流终止性能测量并记 `PERF_EXEMPT`。

两处社区 benchmark 虽然大 shape 相同，但输出合同不同，必须作为两项 workload：

```python
# PyTorch test/inductor/test_online_softmax.py:28，_prepare_softmax
return xmax, (x - xmax).exp().sum(dim=-1, keepdim=True)

# PyTorch test/inductor/test_torchinductor.py:17907，test_prepare_softmax_with_fast_math 内 f
return x_max, (x - x_max).exp().sum(dim=-1, keepdim=True).log()
```

前者默认 `[1024,2048]`，后者默认 `[128,128]`；全量均为 BF16 `[32768,50304]`。
本轮 reference 强制不继承 `DO_PERF_TEST`，全量性能留给功能门禁后的独立 worker。

## 3. constructor 从 CPU 安全移动到加速器（`AU-post-grad-move-constructors-to-gpu`）

代码位置：`torch/_inductor/fx_passes/post_grad.py:2488`。

```python
# torch/_inductor/fx_passes/post_grad.py:2488
def move_constructors_to_gpu(graph):
    ConstructorMoverPass(
        get_gpu_type(), allow_inputs=..., allow_outputs=...
    )(graph)
```

功能意图：将 `arange(..., device="cpu").to(accelerator)` 这类安全构造器直接放到目标设备，避免
CPU 中间量和 copy。正例是 length=32 的 arange→to→add；负例是给 `index_put_` 提供标量的
constructor，这种受保护依赖不能被移动。NPU 结果不能仅凭 CUDA token 消失判 PASS，必须提供
`triton_experimental` 下等价的 FX/IR、copy 与 kernel/task 证据。

index_put 负例原生测试只断言 codegen token，不比较 eager 数值；该 case 会明确标为
`not-asserted-codegen-only`。这不等于数值失败，也不能代替 NPU 最小适配中的数值/依赖验证。

GPU 实测 2/2 通过。arange 正例的社区原生 `generated_kernel_count=1` 成立；index_put 负例保留 CPU
scalar constructor 依赖。当前 review 包的 FX 捕获点位于 mover/codegen 之前，因此正例 before/after
都可见 CPU iota 与 device_put；这里以原生 kernel-count 断言冻结，不把未变化的 FX 伪装成移动证据。

社区没有性能 benchmark。主测原样复用 length=32，额外长度网格只标为 tracker sensitivity。
OFF 必须只跳过 `move_constructors_to_gpu` 调用，不能关闭整个 post-grad；负例只做功能 guard。

## 4. 功能与性能各看什么

| 单元 | 功能性核心 | 性能性核心 | 来源 |
| --- | --- | --- | --- |
| const scatter | 正例改写、密度/shape/const/dtype/backward guard | 全量 CE backward Event + 峰值内存 | 社区 benchmark 直接复用 |
| prepare softmax | online primitive、fast math、signed zero | compiled OFF/ON 全量 BF16 softmax | 社区 shape/方法复用并校正对照臂 |
| constructor mover | arange copy 消除、index_put 依赖不误移 | 小图端到端、copy 与 task 数 | tracker 从社区功能例派生 |

机器可读执行合同见 `upstream/t080_performance_plan.yaml`，完整 GPU 复核见
`report/t080_gpu_reference_review_20260907.md`，最终 NPU/comparison/performance 位于
`results/current/T-080/` 与三个 acceptance-unit 目录。
