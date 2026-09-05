# T-078 功能与性能测例讲解

> 更新时间：2026-09-06 02:21 CST（UTC+08:00）
> 状态：GPU reference 入口与性能方案已准备；性能 worker 尚未实现，尚无 T-078 动态性能结论。
> NPU 固定后端：`triton_experimental`，必须在导入 `torch/torch_npu` 前选择，并用 fresh process 隔离 OFF/ON。

下文逐单元说明功能测例验证的语义、性能测例的来源，以及二者在结果中应如何区分。

## 1. addcdiv 重融合为 FMA（`AU-post-grad-fuse-addcdiv-to-fma`）

代码位置：`torch/_inductor/fx_passes/post_grad.py:2040-2093`。

```python
# torch/_inductor/fx_passes/post_grad.py:2064-2093
@register_graph_pattern(..., extra_check=_is_addcdiv_fma_eligible)
def _fuse_addcdiv_to_fma(match, inp, t1, t2, value):
    match.replace_by_example(
        lambda inp, t1, t2, value: torch.ops.aten.addcdiv(
            inp, t1, t2, value=value
        ),
        [inp, t1, t2, value],
    )
```

功能意图：前端把 `addcdiv` 分解为 `div→mul→add` 后，本 pass 再识别这个结构，使 Triton lowering
可以生成 FMA 与受控舍入除法。`test_addcdiv_fma_bitwise_equal_cuda` 用 `float32[64,64]`、`value=1/2`
检查 compiled/eager 位级一致；`test_addcdiv_fma_uses_fma_and_div_rn_cuda` 再检查 counter、`tl.fma`
和 `div_rn`，因此“数值对”与“确实走目标 codegen”缺一不可。

性能测例：社区没有独立 benchmark。tracker 只从上述 64×64 正例派生 steady-state OFF/ON，两个
value 分开报告。NPU guard 当前是只列 CUDA/XPU 的通用设备 guard，不是显式产品关闭；必须先完成
最小适配评审和功能命中，之后才允许只抑制目标 handler 形成 OFF。

## 2. partial reduction reuse（`AU-post-grad-reuse-partial`）

代码位置：`torch/_inductor/fx_passes/post_grad.py:2095-2135`。

```python
# torch/_inductor/fx_passes/post_grad.py:2119-2130
def reuse_partial(match, input, reduced_dims, keepdim):
    if not statically_known_true(input.meta["val"].numel() >= 4096):
        return True
    partial = partial_red.target(input, reduced_dims, keepdim)
    complete = full_red.target(partial)
```

功能意图：当图里同时需要部分归约和全归约时，全归约继续归约已有 partial，避免再次扫描整个输入。
正例覆盖 `2048×2048 amax→amax`、`1024×1024 amin→min`、`4096×512 amax→max`；负例覆盖
`4×8` 小输入和 dynamic 输入，验证收益阈值与动态 shape guard 没被误放开。

性能测例：社区没有 benchmark，直接复用三个社区大 shape，逐 shape 报告。OFF 必须只抑制
`reuse_partial`，ON 必须 `partial_reduction_reuse=1`；负例只验证 guard，不参与性能平均。

注意测例应保留**优化前**的两条独立归约，不能把 `full(partial)` 直接写进输入图：

```python
# PyTorch test/inductor/test_pattern_matcher.py:564（test_successful_partial_reuse）
partial = partial_fn(x, [0], True)
full = full_fn(x)
return partial, full
```

`full_fn(partial)` 应由目标 pass 在 ON 图中产生；否则 OFF 已经手工优化，无法测出本 pass 的效果。

## 3. addmm bias unfuse（`AU-post-grad-unfuse-bias-add-to-pointwise`）

`test_unfuse_bias_addmm_half_dtypes_when_flag_disabled` 仅做 FileCheck，没有 eager/compiled 数值比较。
reference 因而标注 `not-asserted-codegen-only`；性能 worker 必须补充同图数值门禁，不能把原生 PASS
解释成已验证 half 数值正确。

代码位置：`torch/_inductor/fx_passes/post_grad.py:1853-1940`。

```python
# torch/_inductor/fx_passes/post_grad.py:1853-1938
def should_prefer_unfused_addmm(match):
    return _is_bias_like_addmm_input(inp, output) and all(
        is_pointwise_use(use) for use in output.users
    )

def unfuse_bias_add_to_pointwise(...):
    mm_result = x1 @ x2
    return inp + mm_result
```

功能意图：若 `addmm` 的 bias 像叶子参数，且结果只被 pointwise 消费，就拆成 `mm + bias`，让 bias
与 GELU/ReLU 等后续 pointwise 一起融合。社区用 `10×15 @ 15×20` 覆盖普通 bias、同 shape leaf、
stride-0 expanded bias、view+GELU；无 pointwise consumer、computed accumulator 和默认 half
精度 gate 是重要负例。功能判定需要看 generated code 中 `extern addmm` 是否按预期保留/消失。

性能测例：社区没有独立 benchmark。FP32 用社区 GELU 正例派生；FP16/BF16 复用社区
`keep_addmm_fused_for_half_dtypes` 两状态，分别报告。目标是判断拆分后 pointwise fusion 的收益，除
端到端 Event 外还要记录 kernel/task 数；不能以关闭整个 pattern matcher 作为 OFF。

## 4. baddbmm bias unfuse（`AU-post-grad-unfuse-bias-baddbmm-to-pointwise`）

代码位置：`torch/_inductor/fx_passes/post_grad.py:1879-1987`。

```python
# torch/_inductor/fx_passes/post_grad.py:1955-1985
def unfuse_bias_baddbmm_to_pointwise(...):
    bmm_result = torch.bmm(x1, x2)
    if alpha != 1:
        bmm_result = alpha * bmm_result
    if beta != 1:
        inp = beta * inp
    return inp + bmm_result
```

功能意图与 addmm 相似，但这里还要保持 batch/broadcast 和 `alpha/beta`。社区 shape 为 batch=4、
`[4,6,5] @ [4,5,8]`，bias `[4,1,8]`，覆盖 GELU、expanded-bias ReLU 和
`alpha=0.8,beta=0.2`；无 pointwise consumer 必须保留 baddbmm。

性能测例：从 broadcast-GELU 与 alpha/beta 正例派生，逐合同报告 `baddbmm` 与 `bmm+pointwise`
的端到端 Event、host、显存和 kernel/task 数。只有 NPU 功能阶段确认合法 ON 路径后才解锁。

## 5. 如何读结果

- 功能通过不等于有收益：必须同时看到正确性、目标 counter/FX/codegen 和负例 guard。
- `REWRITE_APPLIED` 表示图确实被改写；仅进入 handler、最终返回 fallback 不能这样记。
- 性能主指标是同一 `triton_experimental` 后端下的 compiled OFF/ON NPU Event；host 与内存用于解释。
- 完整机器合同见 `upstream/t078_performance_plan.yaml`，GPU 一键入口见 `docs/GPU_TASK_RUNNER.md`。
