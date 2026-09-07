# T-078 功能、性能与 GPU/NPU 对照讲解

> 更新时间：2026-09-08 06:07 CST（UTC+08:00）
> 状态：原 12/12 GPU 原生 community cases 与 4 个 FP32/既有合同结论有效；addcdiv FP16/BF16 覆盖扩展等待 GPU reference。
> NPU 后端：所有动态验证与性能结论固定使用 `triton_experimental`，OFF/ON 每臂使用 fresh process。

T-078 包含 4 个 post-grad acceptance units；原冻结范围为 12 个 GPU cases、20 个 variants。覆盖
复核发现 addcdiv 社区测例只有 FP32，而源码 guard 允许 FP16/BF16，因此现计划为 14 个 GPU cases、
22 个 variants，其中 2 个低精度 variants 为 pending。前 12 条 GPU reference
来自 PyTorch 原生测试；新增两条只改变 dtype 的派生 case 会在结果中显式标识。NPU 只在原生入口不能生成 NPU case 时注入 device、backend 和证据采集，
不改变图、shape、dtype、标量或预期命中。性能测量是一次 compiled 子图调用的端到端耗时，不是
完整模型端到端 benchmark。

## 总结

| Acceptance unit | GPU | NPU 功能 | 性能结论 | 最终产品动作 |
| --- | --- | --- | --- | --- |
| `AU-post-grad-fuse-addcdiv-to-fma` | 原生 FP32 2/2 有效；FP16/BF16 派生 0/2 待测 | FP32 修复后命中、bitwise/codegen 通过；低精度待三臂归因 | `PERF_NEUTRAL`（仅 FP32） | 保留 FP32 窄修复；低精度不外推 |
| `AU-post-grad-reuse-partial` | 原生 2/2 case、5/5 variants 有效 | min2 修复后正负例均通过 | `PERF_MIXED` | 保留 pass，逐 shape 监控 |
| `AU-post-grad-unfuse-bias-add-to-pointwise` | 原生 6/6 cases、8/8 variants 有效 | 门禁前能力正确；最终 gate 保留 addmm | `PERF_REGRESSED` | `disable_unfuse_bias_addmm=true` |
| `AU-post-grad-unfuse-bias-baddbmm-to-pointwise` | 原生 2/2 cases、4/4 variants 有效 | 默认标量启用，非默认标量关闭 | `PERF_MIXED` | 选择性 gate |

正式机器可读逐 variant 对照位于各单元的
`results/current/<acceptance-unit>/comparison_result.json`；本文负责解释为什么这样测、图改写意味着什么。

## 1. addcdiv 重融合为 FMA

Acceptance unit：`AU-post-grad-fuse-addcdiv-to-fma`。

### Pattern 与意图

```python
# PyTorch: torch/_inductor/fx_passes/post_grad.py:2064-2093
@register_graph_pattern(..., extra_check=_is_addcdiv_fma_eligible)
def _fuse_addcdiv_to_fma(match, inp, t1, t2, value):
    match.replace_by_example(
        lambda inp, t1, t2, value: torch.ops.aten.addcdiv(
            inp, t1, t2, value=value
        ),
        [inp, t1, t2, value],
    )
```

前端可能把 `addcdiv(inp, t1, t2, value)` 分解为 `div → mul → add`。该 pattern 在 post-grad
阶段将它重新识别成 `aten.addcdiv`，使 lowering 能生成 round-to-nearest 除法和 FMA；目标不只是
“结果近似正确”，还包括 FP32 位级语义和最终 codegen 形态。

### 功能测例与行为

社区功能来源：

- `test_addcdiv_fma_bitwise_equal_cuda`：`float32[64,64]`，覆盖 `value=1/2`，要求
  compiled/eager bitwise equal。
- `test_addcdiv_fma_uses_fma_and_div_rn_cuda`：要求 counter=1，且生成代码包含 `tl.fma` 与
  `triton.language.div_rn`。

三个 variant 的对照：

- `value-one-bitwise-positive`：GPU 与 NPU 都 bitwise 正确，但 `value=1` 的乘一在 decomposition
  中已消去，实际图是 `div → add`，双方 target hit 都是 0。它是正确性邻接测例，不是性能 ON。
- `scaled-value-bitwise-positive`：GPU 原生 reference 有效；NPU 补齐 FP32 pattern/lowering 后命中
  1 次，改写成 `aten.addcdiv`，bitwise equal。
- `fma-div-rn-codegen-positive`：GPU 原生断言命中并生成 FMA/div_rn；NPU 修复前只有普通
  `/、*、+`，修复后 counter=1 且生成代码同时出现 FMA/div_rn。

NPU 修复保持窄范围：仅 NPU、FP32、非 Tensor scalar；没有把 FP16/BF16 邻域一起放开。

这里的“不放开”不是产品显式 disable。源码覆盖审查已新增 FP16/BF16 两个 pending variants，先在
GPU 上复用相同 `64x64/value=2/bitwise/counter/codegen` 合同。GPU 通过后，NPU 再以相同输入比较
目标 OFF、原分解和目标重融合三臂；专项命令、代码和证据清单见
[addcdiv 低精度覆盖说明](T078_ADDCDIV_LOWP_COVERAGE.md)。

### 性能测例与结论

社区没有 pass-specific benchmark。当前 workload 原样复用社区 `float32[64,64], value=2` 正例，
只增加目标 handler 的 OFF/ON、10 次 warmup、100 次同步计时和三轮交错 fresh process。`value=1`
不命中目标，不能用来制造 ON。

- host p50/p99：改善 0.70% / 4.54%。
- NPU Event p50/p99：改善 1.82% / 2.62%。
- 峰值内存与 wrapper dispatch 不变。

结论为 `PERF_NEUTRAL`：功能修复保留，但不宣称显著性能收益。

## 2. partial reduction reuse

Acceptance unit：`AU-post-grad-reuse-partial`。

### Pattern 与意图

```python
# PyTorch: torch/_inductor/fx_passes/post_grad.py:2095-2135
def reuse_partial(match, input, reduced_dims, keepdim):
    if not statically_known_true(input.meta["val"].numel() >= 4096):
        return True
    partial = partial_red.target(input, reduced_dims, keepdim)
    complete = full_red.target(partial)
```

输入图原本同时计算 partial reduction 与 full reduction。改写后 full reduction 继续归约已存在的
partial 结果，避免第二次扫描整个输入。输入测例必须保持下面的“未优化写法”，不能事先手写
`full_fn(partial)`：

```python
# PyTorch: test/inductor/test_pattern_matcher.py（test_successful_partial_reuse）
partial = partial_fn(x, [0], True)
full = full_fn(x)
return partial, full
```

### 功能测例与行为

- `amax-amax-large-positive`：2048×2048。GPU/NPU 均命中；NPU after 图的 full amax 读取
  partial 结果。
- `amin-min-large-positive`：1024×1024。NPU 原 pattern 已命中，但旧 min2 helper 在 Triton
  Ascend 编译失败；改为 `tl.reduce(..., tl.minimum(..., PropagateNan.ALL))` 后数值和 NaN 邻接通过。
- `amax-max-large-positive`：4096×512。GPU/NPU 均命中，覆盖非方阵和 amax→max 组合。
- `small-numel-negative`：4×8 小于 4096，双方 counter=0，保留两条独立 reduction。
- `dynamic-input-negative`：symbolic numel 不能静态证明收益阈值，双方 counter=0。

这里“发生改写”是 full reduction 的输入从原张量变为 partial 结果；它发生在 Inductor post-grad
FX pass，早于 lowering、scheduler、autotune 与 kernel codegen。

### 性能测例与结论

社区没有独立 benchmark，因此逐 shape 复用三个社区正例，负例只守 guard：

| Workload | host p50/p99 改善 | Event p50/p99 改善 | 结论 |
| --- | ---: | ---: | --- |
| 2048² amax→amax | 47.09% / 44.12% | 56.65% / 54.21% | `PERF_IMPROVED` |
| 1024² amin→min | 1.64% / 8.80% | 2.71% / 27.85% | `PERF_NEUTRAL` |
| 4096×512 amax→max | 23.64% / 32.21% | 33.23% / 37.69% | `PERF_IMPROVED` |

总体为 `PERF_MIXED`：三个 shape 中两个显著改善，一个 p50 中性。不能用平均数掩盖 shape 差异。

## 3. addmm bias unfuse

Acceptance unit：`AU-post-grad-unfuse-bias-add-to-pointwise`。

### Pattern 与意图

```python
# PyTorch: torch/_inductor/fx_passes/post_grad.py:1850-1940
def should_prefer_unfused_addmm(match):
    return _is_bias_like_addmm_input(inp, output) and all(
        is_pointwise_use(use) for use in output.users
    )

def unfuse_bias_add_to_pointwise(...):
    mm_result = x1 @ x2
    return inp + mm_result
```

当 bias 是适合延后处理的 leaf/broadcast 输入，且 addmm 输出只流向 pointwise consumer 时，将
`addmm` 拆成 `mm + bias`，期望让 bias add 与后续 GELU/ReLU 融合。这里的“拆分”不是分解到
autotune 之后，而是 post-grad FX 图改写，随后才进入 lowering 与候选 kernel 选择。

### 功能测例与行为

GPU 原生六个 cases 覆盖八个 variants：

- `gelu-consumer-positive`、`view-gelu-consumer-positive`、`same-shape-leaf-positive`、
  `expanded-bias-positive`：GPU 都允许拆分。
- `no-pointwise-consumer-negative`、`computed-accumulator-negative`：GPU/NPU 都保留 addmm。
- `half-dtype-preserve-negative`：默认 `keep_addmm_fused_for_half_dtypes=true`，FP16/BF16 保留 addmm。
- `half-dtype-gate-disabled-positive`：GPU 测试态关闭 half gate 后允许拆分；该社区 case 是 codegen
  FileCheck，不能扩大解释成 half 数值正确性证明。

`should_prefer_unfused_addmm` 使用的通用 `is_gpu` 不是 NPU 缺能力证据：torch_npu 的
`patch_is_gpu` 已把 NPU 纳入该语义。NPU 只需要社区测试入口的 device/backend 适配。门禁落盘前，
NPU 能正确拆成 mm+pointwise；性能处置后，最终 `disable_unfuse_bias_addmm=true` 在 handler 前拒绝
所有 addmm unfuse，因此正例形成有依据的 `EXPECTED_PRODUCT_DIVERGENCE`，负例结论不变。

最终动态 gate 验证：target hit=0，生成结构保留 addmm、不出现 mm，数值最大误差
`2.384185791015625e-7`。

### 性能测例与结论

社区通用 `benchmarks/operator_benchmark/pt/addmm_test.py` 只测裸 addmm，没有 pointwise consumer，
不会触发本 pattern。因此候选测量复用社区 `10×15 @ 15×20 + bias + GELU` 图。

- host p50/p99：回退 20.81% / 138.27%。
- NPU Event p50/p99：回退 14.81% / 75.20%。
- wrapper dispatch 2→2，峰值内存不变。

候选为 `PERF_REGRESSED`，所以产品全局关闭该 NPU path。显式关闭后不再绕过门禁补造 ON 性能。

## 4. baddbmm bias unfuse

Acceptance unit：`AU-post-grad-unfuse-bias-baddbmm-to-pointwise`。

### Pattern 与意图

```python
# PyTorch: torch/_inductor/fx_passes/post_grad.py:1955-1985
def unfuse_bias_baddbmm_to_pointwise(...):
    bmm_result = torch.bmm(x1, x2)
    if alpha != 1:
        bmm_result = alpha * bmm_result
    if beta != 1:
        inp = beta * inp
    return inp + bmm_result
```

它把 broadcast-bias `baddbmm` 拆成 `bmm + pointwise bias`，同时必须保留 batch、broadcast、
`alpha` 与 `beta`。社区 shape 为 batch=4、`[4,6,5] @ [4,5,8]`、bias `[4,1,8]`。

### 功能测例与行为

- `broadcast-gelu-positive`：GPU 原生拆成 bmm+add+GELU；NPU 最终 gate 对默认
  `alpha=beta=1` 放行，target hit=1，保留 bmm、不保留 baddbmm，误差不超过
  `2.384185791015625e-7`。
- `expanded-bias-relu-positive`：GPU/NPU 都保持 stride-0 broadcast 语义并拆分；该分支只形成
  功能结论，没有借用 GELU 的性能数字。
- `alpha-beta-positive`：GPU 可正确拆成 `0.8*bmm + 0.2*bias`；NPU 门禁前也能正确拆分，但性能
  回退，最终选择性 gate 令 target hit=0、保留 baddbmm，数值误差 0。
- `no-pointwise-consumer-negative`：GPU/NPU 都不匹配，保留 baddbmm。

### 性能测例与结论

社区通用 bmm benchmark 同样没有本 pattern 所需的 bias+pointwise consumer，所以使用两个社区
功能正例派生 workload：

| Workload | host p50/p99 改善 | Event p50/p99 改善 | 内存 | 产品动作 |
| --- | ---: | ---: | ---: | --- |
| 默认标量 broadcast+GELU | 5.30% / 46.65% | 12.24% / 71.39% | 不变 | 保持启用 |
| alpha=.8、beta=.2 | 2.87% / 64.99% | -5.29% / 9.57% | allocated +512B | 显式关闭 |

总体为 `PERF_MIXED`。最终配置只关闭非默认标量：
`disable_unfuse_baddbmm_non_default_scalars=true`；默认标量收益路径保留。

## 5. 如何核验结果

1. 功能通过至少同时看 eager/compiled、目标 counter 或 FX/codegen，以及负例 guard；编译成功不等于命中。
2. `REWRITE_APPLIED` 表示目标 FX 结构真实改变；仅进入 handler 后 fallback 不能记为生效。
3. 性能数字来自相同源码、输入、`triton_experimental`、目标级 OFF/ON 和 fresh process；
   `wrapper_dispatch_count` 是静态 wrapper 调用点，不等于 profiler NPU task 数。
4. GPU 文本 handoff 1.0 保留 case 状态、FX signature 与 inventory hash，但没有原始 FX 正文；
   因此本轮可确认结构签名与原生断言，不能声称完成逐行 GPU/NPU FX 正文对照。
5. 正式汇总见 `results/current/T-078/performance_summary.json`，最终门禁见
   `results/current/T-078/product_gate_verification.json`，逐 variant 解释见四份 `comparison_result.json`。
