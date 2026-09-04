# T-078 第三批 acceptance-unit 映射审核

> 更新时间：2026-09-04 08:55 CST（UTC+08:00）
> PyTorch：`release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> 状态：4 个单元、12 个 direct community cases、20 个 variants 已静态冻结映射，等待 GPU reference

## 1. 批次结论

T-078 从 T-074 的 `no-test-found` 与 `indirect-regression-test` 集合反向检查冻结源码和社区测试，选择
4 个合同。批次规模按独立 optimization contract 计数，不按 registration 行或测试数量计数：

| Acceptance unit | T-074 旧分类 | 审核后事实源 | Cases | Variants |
| --- | --- | --- | ---: | ---: |
| `AU-post-grad-fuse-addcdiv-to-fma` | `no-test-found` | 发现 2 条直接 CUDA/Triton 测例 | 2 | 3 |
| `AU-post-grad-reuse-partial` | indirect | 正例/负例都直接断言目标 counter | 2 | 5 |
| `AU-post-grad-unfuse-bias-add-to-pointwise` | indirect | addmm 专属 generated-code 正负例 | 6 | 8 |
| `AU-post-grad-unfuse-bias-baddbmm-to-pointwise` | indirect | baddbmm 专属 generated-code 正负例 | 2 | 4 |

合计 4 个 acceptance units、12 个 community cases、20 个动态 variants，当前全部采用 `direct`。
只有 GPU 原生 suite 全部 `reference_valid=true` 后，四个单元才进入冻结 denominator。

## 2. 主要映射修正

### 2.1 addcdiv 并非 no-test-found

旧静态索引没有找到 `_fuse_addcdiv_to_fma` 的测试。冻结 commit 实际已经包含：

- `GPUTests.test_addcdiv_fma_bitwise_equal_cuda`：验证 `value=1/2` 两分支 bitwise equal；
- `GPUTests.test_addcdiv_fma_uses_fma_and_div_rn_cuda`：验证 counter、`tl.fma` 和
  `triton.language.div_rn`。

这些测试由 `copy_tests(CommonTemplate, GPUTests, GPU_TYPE)` 动态生成。reference runner 的静态
qualname 检查按社区 `copy_tests` 语义追加 `_cuda` 后缀，无需导入 torch，也不把模板类或不存在的
无后缀方法误当可执行入口。

### 2.2 addmm 与 baddbmm 必须拆开

旧索引把四条 addmm/baddbmm 测试同时映射给两个 handler。源码中它们分别匹配 `aten.addmm` 和
`aten.baddbmm`，replacement、broadcast 语义和 scalar 参数合同不同，因此冻结为两个 acceptance
units，并只让各自专属 community tests 计入动态证据。

### 2.3 pointless_view 映射暂不接受

`AU-joint-graph-pointless-view` 的候选锚点是单个 no-op view handler，但旧索引映射的是
`pointless_view_pair` 两级 view 消除测例。两者是相邻但不同的 registration/contract；本批不猜测、
不复用 pair 测例，继续标为 `needs-review`。

## 3. 四个 pattern 的意图

### 3.1 addcdiv → FMA

位置：`torch/_inductor/fx_passes/post_grad.py:2064-2093`。

```python
# torch/_inductor/fx_passes/post_grad.py:2064-2093
@register_graph_pattern(
    CallFunction(aten.add.Tensor,
        KeywordArg("inp"),
        CallFunction(aten.mul.Tensor,
            CallFunction(aten.div.Tensor, KeywordArg("t1"), KeywordArg("t2")),
            KeywordArg("value"))),
    pass_dict=pass_patterns[2],
    extra_check=_is_addcdiv_fma_eligible,
)
def _fuse_addcdiv_to_fma(match, inp, t1, t2, value):
    match.replace_by_example(
        lambda inp, t1, t2, value: torch.ops.aten.addcdiv(
            inp, t1, t2, value=value
        ),
        [inp, t1, t2, value],
    )
```

`torch.addcdiv` 在 Inductor 看到前已经分解为 div→mul→add。该 pass 把子图重新组合成 addcdiv，使
lowering 能生成舍入语义受控的 `tl.fma` 与 `div_rn`。这里要求同时验证 rewrite、bitwise correctness
和最终 codegen，不能只看 matcher counter。

### 3.2 partial reduction reuse

位置：`torch/_inductor/fx_passes/post_grad.py:2095-2133`。

```python
# torch/_inductor/fx_passes/post_grad.py:2119-2130
def reuse_partial(match, input, reduced_dims, keepdim):
    if not statically_known_true(input.meta["val"].numel() >= 4096):
        return True

    def replacement(inp):
        partial = partial_red.target(inp, reduced_dims, keepdim)
        complete = full_red.target(partial)
        return (partial, complete)

    counters["inductor"]["partial_reduction_reuse"] += 1
    match.replace_by_example(replacement, [input])
```

意图是已有 partial reduction 时，让 full reduction 从 partial 继续归约，避免再次读取和归约完整输入。
T-078 保留三组大输入正例以及小输入、dynamic 两组负例，不修改 `4096` 收益阈值。

### 3.3 addmm bias unfuse

位置：`torch/_inductor/fx_passes/post_grad.py:1853-1940`。

```python
# torch/_inductor/fx_passes/post_grad.py:1853-1876,1905-1938
def should_prefer_unfused_addmm(match):
    if not is_gpu(match.kwargs["inp"].meta["val"].device.type):
        return False
    ...
    return _is_bias_like_addmm_input(inp, output) and all(
        is_pointwise_use(use) for use in output.users
    )

def unfuse_bias_add_to_pointwise(...):
    mm_result = x1 @ x2
    ...
    return inp + mm_result
```

当 bias-like addmm 后面只有 pointwise consumers 时，将 addmm 拆成 mm 与 pointwise bias，使后续
pointwise 融合有机会吞并 bias/activation。无 pointwise consumer、computed accumulator 和默认 half
dtype 精度 gate 都是必须保留的负向合同。

### 3.4 baddbmm bias unfuse

位置：`torch/_inductor/fx_passes/post_grad.py:1879-1987`。

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

该合同与 addmm 相似，但必须独立验证 batch/broadcast bias、stride-0 expanded bias，以及非默认
`alpha=0.8, beta=0.2` 的标量语义。

## 4. GPU/NPU 与性能工作线

当前尚未产生 T-078 GPU/NPU verdict。执行顺序固定为：

```text
GPU 12/12 原生 community cases
→ 20/20 variants reference valid
→ 冻结 4 个 acceptance units
→ NPU 原生入口优先
→ blocker 成立后才设计最小 adapter
→ triton_experimental fresh-process 功能对比
→ REWRITE_APPLIED / TEMPLATE_SELECTED 分层记录
→ 正确性门禁通过后处理性能
```

冻结源码中未发现这四个 pattern 的独立社区性能测例。GPU reference 有效后，如 NPU 存在合法 ON
路径，将按工作流从相应 community functional case 原样派生 workload，只增加 OFF/ON、同步计时、
Event、显存和 generated-code 采集；不会缩小 shape 或改写正负合同。

## 5. GPU 一键入口

```bash
bash "${TRACKER_ROOT}/scripts/run_gpu_reference_task.sh" \
  --task T-078 \
  --gpu 2
```

脚本自动维护：

```text
/data/z50063656/tmp/t078-reference-results/latest
/data/z50063656/tmp/t078-reference-results/latest-text-handoff.json
```

因此不需要人工寻找 `reference-<timestamp>`。
