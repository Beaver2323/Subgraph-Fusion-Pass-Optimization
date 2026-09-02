# T-076 pattern 源码、GPU/NPU 行为对照导读

> 更新时间：2026-09-02 17:42 CST（UTC+08:00）
> PyTorch：`release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> GPU reference：A100，13/13 direct cases valid
> NPU：Ascend910B2，`triton_experimental`

本文是 `results/current/*/comparison_result.json` 的学习视图。JSON 中每个 variant 都含
`intent/source_locations/gpu_behavior/npu_behavior`；本文补充带文件位置的关键代码块。判断顺序是
pattern 是否命中、replacement 是否发生、实际 lowering/codegen 路径、最后才是数值与性能。

## 1. post-grad mm + mm

意图是把 `mm(A,B) + mm(C,D)` 识别成一个合同。两路内部 K 分别合法且输出 M/N 相等时 pattern
命中；是否真正用单 kernel/extern，由 lowering 再决定。

```python
# torch/_inductor/fx_passes/post_grad.py:979-1017
def is_valid_mm_plus_mm(match):
    ...
    if m1 != m2 or n1 != n2:
        return False
    return True

@register_lowering_pattern(
    CallFunction(aten.add, CallFunction(aten.mm, ...), CallFunction(aten.mm, ...)),
    extra_check=is_valid_mm_plus_mm,
)
def mm_plus_mm(match, mat1, mat2, mat3, mat4):
    return inductor.kernel.mm_plus_mm.tuned_mm_plus_mm(mat1, mat2, mat3, mat4)
```

```python
# torch/_inductor/kernel/mm_plus_mm.py:128-152
def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None):
    ...
    if (
        m1 * n1 == 0
        or m2 * n2 == 0
        or not V.graph.sizevars.statically_known_list_equals(mat1.get_size(), mat3.get_size())
        or not V.graph.sizevars.statically_known_list_equals(mat2.get_size(), mat4.get_size())
    ):
        return lowerings[aten.add](
            lowerings[aten.mm](mat1, mat2), lowerings[aten.mm](mat3, mat4)
        )
```

| variant | 意图 | GPU | NPU | 对比 |
| --- | --- | --- | --- | --- |
| `same-k-positive` | 相同 K 的两路 mm 可直接进入融合候选 | 命中 `1/3` | 命中，extern `_mm_plus_mm` | `BEHAVIOR_UNCHANGED` |
| `different-k-positive-pattern-fallback` | 不同 K 仍可识别合同，但当前 lowering 可保守回退 | 命中后 two-mm-plus-add | 同样命中，两个 extern mm + Triton add | `BEHAVIOR_UNCHANGED` |
| `output-shape-mismatch-negative` | M/N 不同必须拒绝目标 pattern | 目标不命中；全局 1/2 来自其他 pattern | 目标不命中，原图正确 | `BEHAVIOR_UNCHANGED` |

## 2. joint-graph pad-mm

意图是在矩阵维度不利于 GEMM 时，把 M/K/N 补齐到对齐尺寸，计算后 slice 回用户形状。是否值得
padding 由 `should_pad_mm` 决定。

```python
# torch/_inductor/fx_passes/pad_mm.py:812-882
def mm_pattern(mat1, mat2):
    return aten.mm(mat1, mat2)

def should_pad_mm(match):
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.mm)

def mm_replace(mat1, mat2):
    ...
    return pad_mm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length)
```

NPU 当前安装态明确关闭整个 pad family：

```python
# torch_npu/_inductor/triton_experimental/overrides.py:74-84
def _disable_pad_mm_pass():
    if not ncfg.disable_pad_mm:
        return
    from torch._inductor import config as inductor_config
    inductor_config.shape_padding = False
```

| variant | 意图 | GPU | NPU | 对比 |
| --- | --- | --- | --- | --- |
| `dynamic-m-positive` | 保持动态 M，只补齐可确定的未对齐维 | padded-K 命中，数值正确 | gate 关闭，原始 mm 动态正确 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `training-inference-registration` | training/inference 两个 registration 共享一个合同 | 静态注册证据 | 同一静态证据 | `BEHAVIOR_UNCHANGED` |
| `stride-preservation` | padding/slice 后 stride 不能漂移 | padding 后 stride 与 eager 相同 | 不 padding，stride `(2,1)` 仍正确 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `disabled-baseline-negative` | 证明 NPU 未命中来自显式产品 gate | N/A | `disable_pad_mm=True`，无 marker/template/cache | `EXPECTED_PRODUCT_DIVERGENCE` |

## 3. joint-graph pad-bmm

bmm 与 mm 共享 padding 思路，但还要保留 batch 维，并覆盖训练图的 dtype/梯度回归。

```python
# torch/_inductor/fx_passes/pad_mm.py:885-935
def bmm_pattern(mat1, mat2):
    return aten.bmm(mat1, mat2)

def should_pad_bmm(match):
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.bmm)

def bmm_replace(mat1, mat2):
    ...
    return pad_bmm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length)
```

```python
# torch/_inductor/fx_passes/pad_mm.py:999-1024
gen_register_replacement(f"{name}_training", ..., joint_fwd_bwd, ...)
gen_register_replacement(f"{name}_inference", ..., fwd_only, ...)
```

| variant | 意图 | GPU | NPU | 对比 |
| --- | --- | --- | --- | --- |
| `dynamic-batch-positive` | batch 动态，M/K/N padding 后再 slice | padded-K 命中 | gate 关闭，原始 bmm 正确 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `training-inference-registration` | 两条图管线共享合同 | 静态双 registration | 静态证据一致 | `BEHAVIOR_UNCHANGED` |
| `dtype-gradient-related-regression` | 防错误 autocast、NaN 梯度 | 关联回归通过 | MaskedMHA forward/backward、dtype、梯度通过 | `BEHAVIOR_UNCHANGED` |
| `disabled-baseline-negative` | 验证 pad-bmm 关闭态 | N/A | dynamic/static 均无 padding marker | `EXPECTED_PRODUCT_DIVERGENCE` |

## 4. joint-graph pad-addmm

addmm 除矩阵 padding 外还必须同步处理 bias broadcast、`alpha/beta`。CUDA 的 `beta=0` 对不可广播
bias 有专属保护：既然 bias 数值应被忽略，就不能让 padding 路径重新读取它。

```python
# torch/_inductor/fx_passes/pad_mm.py:197-223
def addmm_pattern(input, mat1, mat2, beta, alpha):
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

def should_pad_addmm(match):
    mat1, mat2, input = fetch_fake_tensors(match, ("mat1", "mat2", "input"))
    beta = match.kwargs["beta"]
    if beta == 0 and input.is_cuda and not _is_statically_expandable_to(...):
        return False
    return should_pad(match, mat1, mat2, torch.ops.aten.addmm, input=input)
```

```python
# torch/_inductor/fx_passes/pad_mm.py:226-264
def pad_addmm(input, mat1, mat2, m_pad, k_pad, n_pad, beta=1.0, alpha=1.0, ...):
    mat1 = pad_mat1(...)
    mat2 = pad_mat2(...)
    if input is not None:
        if n_pad != 0 and input.dim() == 2 and input.shape[1] != 1:
            input = pad_dim(input, n_pad, 1)
        elif n_pad != 0 and input.dim() == 1 and input.shape[0] != 1:
            input = pad_dim(input, n_pad, 0)
        if m_pad != 0 and input.dim() == 2 and input.shape[0] != 1:
            input = pad_dim(input, m_pad, 0)
    res = aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    if m_pad != 0:
        res = res[:-m_pad, :]
    if n_pad != 0:
        res = res[:, :-n_pad]
    return res
```

| variant | 意图 | GPU | NPU | 对比 |
| --- | --- | --- | --- | --- |
| `dynamic-m-positive` | 动态 M + padded-K | 命中，正确 | gate 关闭，原始 addmm 正确 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `bias-broadcast-positive` | 1D/2D bias padding 后仍正确广播 | 六种 bias 合同通过 | 不 padding，六种 bias 仍正确 | `EXPECTED_PRODUCT_DIVERGENCE` |
| `beta-zero-mismatched-bias-negative` | CUDA beta=0 时拒绝读取不可广播 bias | 不 padding | 源码限定 CUDA，NPU 为 N/A | `EXPECTED_PRODUCT_DIVERGENCE` |
| `training-inference-registration` | 两条图管线共享 addmm-padding 合同 | 静态证据 | 静态证据一致 | `BEHAVIOR_UNCHANGED` |
| `disabled-baseline-negative` | 验证 NPU pad-addmm gate | N/A | gate 关闭，运行 case 正确 | `EXPECTED_PRODUCT_DIVERGENCE` |

## 5. post-grad add + mm → addmm

该 pattern 与 pad-addmm 不同：它把图中的 `bias + mm` 或 `mm + bias` 直接替换为 `aten.addmm`。
guard 要求 bias 是 Tensor、能广播到二维 mm 输出、三者 dtype 相同，并通过后端偏好检查。

```python
# torch/_inductor/fx_passes/post_grad.py:1983-2037
def is_valid_addmm_fusion(match):
    mat1, mat2 = match.args
    inp = match.kwargs["inp"]
    if not (isinstance(inp, torch.fx.Node) and isinstance(inp.meta["val"], torch.Tensor)):
        return False
    in_shape = inp.meta["val"].shape
    mm_shape = mat1.meta["val"].shape[0], mat2.meta["val"].shape[1]
    if not is_expandable_to(in_shape, mm_shape):
        return False
    inp_dtype = inp.meta["val"].dtype
    if inp_dtype != mat1.meta["val"].dtype or inp_dtype != mat2.meta["val"].dtype:
        return False
    return not should_prefer_unfused_addmm(match)

# 两个对称 registration：add(mm(...), inp) 与 add(inp, mm(...))
def addmm(match, mat1, mat2, *, inp):
    match.replace_by_example(lambda inp, a, b: aten.addmm(inp, a, b), [inp, mat1, mat2])
```

原 T-076 分类错误来自“源码 overlay”和“实际安装包”混淆。当前 Pass 安装态是：

```python
# Pass site-packages/torch_npu/_inductor/triton_experimental/config.py:132-133
disable_addmm_fusion: bool = True

# Pass site-packages/torch_npu/_inductor/triton_experimental/fx_passes.py:541-550
if not ncfg.disable_addmm_fusion:
    return
...
entry.extra_check = lambda match: False
```

P-018 候选把默认值改为 False，并始终安装 live wrapper；显式 True 仍可在导入后关闭：

```python
# torch_npu/_inductor/triton_experimental/fx_passes.py:541-578（P-018 wheel）
def gated_extra_check(match, _original_check=original_check):
    if ncfg.disable_addmm_fusion:
        return False
    return _original_check(match)
```

| variant | GPU | 当前 NPU 安装态 | P-018 候选 | 正式解释 |
| --- | --- | --- | --- | --- |
| `matrix-bias-both-orders-positive` | `2/4`，融合 | gate 下 `0/0`，mm+add | `2/4`，两个 extern addmm | 安装态为 `EXPECTED_PRODUCT_DIVERGENCE`，候选已验证 |
| `vector-bias-both-orders-positive` | `2/4`，融合 | gate 下 `0/0` | `2/4`，两个 extern addmm | 同上 |
| `non-expandable-or-batched-bias-negative` | `0/0` | `0/0` | `0/0`，保留 mm+add | `BEHAVIOR_UNCHANGED` |
| `python-or-symbolic-scalar-negative` | `0/0` | `0/0` | `0/0`，dynamic 图也不融合 | `BEHAVIOR_UNCHANGED` |

P-018 exact upstream 证据位于
`results/current/AU-post-grad-addmm/p018_candidate_result.json`。它没有新增 addmm lowering，也没有手写
Triton addmm；正例直接复用已有 NPU extern addmm lowering。T-022 wrapper 只用于当前 editable headers
环境中的 fresh Triton launcher 编译，不能被描述为产品修复。

## 6. 如何读 verdict

- `BEHAVIOR_UNCHANGED`：GPU 与 NPU 的目标合同或负向 guard 一致。
- `EXPECTED_PRODUCT_DIVERGENCE`：GPU 启用，但 NPU 有明确、可定位的产品 gate；原图仍需正确。
- `NPU_REGRESSION`：没有明确产品控制，且同一稳定 reference 合同在 NPU 首次出现非预期分歧。
- pattern counter 只证明改写是否发生；最终还要结合 transformed FX、`output_code.py` 和数值判断。
