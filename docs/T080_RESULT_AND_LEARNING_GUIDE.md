# T-080 功能、适配、修复与性能学习指南

> 更新时间：2026-09-07 20:50 CST（UTC+08:00）
>
> PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
>
> torch_npu：`83cc452480c3546fd5cccf853bfe3a360ce9dbfc` 加当前 T-080 Python 补丁
>
> NPU 后端：仅 `triton_experimental`

本文既是结果导航，也是代码学习入口。GPU 结论来自 T-080 冻结 handoff；NPU 结论来自 fresh-process
实测，不能与 default/DVM/MLIR 后端互换。

每个 acceptance unit 均在下文同时给出功能测例、性能测例、源码意图、GPU/NPU 行为和处置边界。

## 1. `scatter_upon_const_tensor`（`AU-joint-graph-scatter-upon-const-tensor`）

### 1.1 pattern 和意图

位置：`torch/_inductor/fx_passes/joint_graph.py:1150`、`:1203`。

```python
# torch/_inductor/fx_passes/joint_graph.py
@register_graph_pattern(
    CallFunction(
        aten.scatter.value,
        CallFunction(aten.full, KeywordArg("shape"), KeywordArg("background_val"), ...),
        KeywordArg("dim"), KeywordArg("selector"), KeywordArg("val"),
    ),
    pass_dict=patterns,
    extra_check=scatter_upon_const_tensor_extra_check,
)
def scatter_upon_const_tensor(...):
    indices = torch.arange(length, device=selector.device, dtype=torch.int64)
    mask = selector.expand(shape) == indices.view(*view_shape)
    return torch.where(mask, val, background_val).to(dtype)
```

意图是把“先构造完整常量张量，再做稀疏 scalar scatter mutation”改成按坐标生成 mask 的 pointwise
`where`。这样避免 full、scatter mutation 和额外 copy，尤其降低 CrossEntropy backward 的中间显存。
guard 保证 selector 在非 scatter 维覆盖完整 shape，且 scatter 维 size 必须为 1。

### 1.2 GPU/NPU 对照

- GPU：8 个原生社区 case 全部通过；正例 target metric=1，三类负例不改写。
- NPU 原生入口：模块 `__main__` 因 `HAS_GPU=false` 返回 0，但执行 0 tests。
- NPU 最小适配：只注入 `GPU_TYPE=npu`、默认设备、显式 TestCase 实例化和 artifact capture。
- NPU：3D、非末维、负 dim、FP16/BF16 和 CrossEntropy 正例均 target metric=1；short-index、dense、
  non-const 均为 0，数值/梯度通过。
- non-const 的 CUDA StarDep 过估算等式比 NPU 实际值多 8192 bytes；FX 仍保留
  `aten.scatter.value`，IR 仍有 mutation，因此只适配这一后端 metrics 口径，不放宽 guard。

性能使用社区完整 `B=32,T=1024,D=768,V=50257` BF16 Linear→CrossEntropy→backward。该 shape
首次暴露两个 experimental codegen 缺口，详见第 4 节；修复后 3×OFF/3×ON 全部正确，但 ON 相比
OFF 的 Event p50/p99 回退 10.50%/10.55%，host p50/p99 回退 10.57%/10.60%，峰值 allocated/reserved
增加 35.87%/5.51%，判定 `PERF_REGRESSED`。因此产品新增
`disable_scatter_upon_const_tensor=true`：默认保留原 Scatter，显式 False 可逆恢复 pointwise；真实
NPU 双臂均 PASS。

## 2. `prepare_softmax`（`AU-post-grad-prepare-softmax`）

### 2.1 pattern 和意图

位置：`torch/_inductor/fx_passes/post_grad.py:490-520`。

```python
# torch/_inductor/fx_passes/post_grad.py
def prepare_softmax_pattern(x, dim):
    xmax = x.amax(dim=dim, keepdim=True)
    xsub = x - xmax
    xexp = xsub.exp()
    xsum = xexp.sum(dim=dim, keepdim=True)
    return xmax, xsum, xsub, xexp

def prepare_softmax_replacement(x, dim):
    xmax, xsum = prepare_softmax_online(x, dim)
    xsub = x - xmax
    return xmax, xsum, xsub, xsub.exp()
```

意图是让 max 与指数和由 online-softmax reduction 一次遍历产生，减少读写和 reduction pass，同时保留
训练反向所需的 `xmax/xsum`。

### 2.2 GPU/NPU 对照

- GPU：fast-math、strict signed-zero、community perf 默认入口 3/3 通过并生成 online kernel。
- NPU upstream generic guard 只接受 `cuda/xpu`。测试态只替换 `register_replacement` 闭包中的这一格
  `extra_check`，其余 traced-pattern/shape 检查保持不变。
- 探针后 FX 确实出现 `prims.prepare_softmax_online.default`，数值路径可运行。
- 但 `torch_npu/_inductor/lowering_fallback_list.py` 明确列出该 prim；generated code 调用外部 prim，
  没有 `online_softmax_reduce` Triton kernel。因此这是结构改写，不是端到端优化生效。
- strict signed-zero 的旁路 compiled `amax` 还会在 Triton-Ascend
  `ConvertLinalgRToBinary` 报未实现；测试探针用 eager amax 保留位级 oracle，但不宣称底层 amax 已修复。

结论：产品 lowering 明确关闭，按规则性能免测；不得临时删除 fallback 制造 ON 路径。

## 3. `move_constructors_to_gpu`（`AU-post-grad-move-constructors-to-gpu`）

### 3.1 pass 和意图

位置：`torch/_inductor/fx_passes/post_grad.py:2488`，调用点约 `:310`。

```python
# torch/_inductor/fx_passes/post_grad.py
def move_constructors_to_gpu(graph):
    ConstructorMoverPass(
        get_gpu_type(),
        allow_inputs=allow_inputs_outputs,
        allow_outputs=allow_inputs_outputs,
    )(graph)
```

这不是 `register_graph_pattern`，而是 post-grad 图遍历 pass。它把安全的 CPU `arange/full/...` 构造器
直接改成目标设备构造，消除 CPU→设备 copy；若构造器参与 `index_put_` scalar 依赖等不可移动链，则必须
留在 CPU。

### 3.2 GPU/NPU 对照

- GPU：arange 正例为 1 个 generated kernel；index_put scalar 负例不产生 CUDA scalar allocation。
- NPU 原生：`GPUTests/TritonCodeGenTests` 类未生成，方法体未进入。
- NPU 适配：正例直接调用原 `CommonTemplate.test_move_arange`，只把 `common/device` 绑定到 NPU；原
  `generated_kernel_count=1` 断言通过。
- 负例完整复用函数、shape 与 mutation，只把无意义的 CUDA token 断言替换为
  `empty_strided_npu(())` 不存在；数值和生成代码均通过。
- 三轮 OFF/ON：社区 length=32 Event p50 改善 0.46%，p99 回退 0.50%，结论 `PERF_NEUTRAL`；
  1024/65536/1048576 仅是 tracker sensitivity，不能冒充社区 benchmark。

## 4. 完整 Scatter Benchmark 暴露的修复与产品门禁

调用栈：

```text
test_cross_entropy_loss / t080_performance_worker.py
  -> torch.compile(fn)
  -> compile_fx._recursive_post_grad_passes
  -> joint_graph.scatter_upon_const_tensor（ON 时）
  -> GraphLowering / NPUTritonKernel codegen
  -> NPUCachingAutotuner._precompile_config
  -> triton.compile(ASTSource, NPUOptions)
```

第一处：`auto_blockify_size` 是 Ascend compiler option，却被塞进 AST constexpr constants；大 grid
下 Triton 查找不存在的 kernel 参数并报错。修复位于
`torch_npu/_inductor/triton_experimental/npu_triton_heuristics.py::_precompile_config`：从
`cfg.kwargs` 移出该键，改由 `options` 传递。

第二处：group-dispatch 在查询 logical ndim 后把物理 tile 展为二维，零初始化仍生成
`tl.full([1], 0, ...)`，无法与 `[1, XBLOCK]` store mask 广播。修复位于
`torch_npu/_inductor/triton_experimental/codegen/triton.py::NPUTritonKernelOverrides.constant`：零常量
生成 typed scalar，使其不依赖后续物理 rank。

两处补丁均由完整社区 shape 触发；小 shape 功能 case 不足以覆盖它们。

代码能正确运行不代表值得默认启用。修复 codegen 后，完整图的三轮数据稳定回退且显存明显增加，
所以在 `torch_npu/_inductor/triton_experimental/config.py` 增加默认关闭项，并在
`fx_passes.py` 只包装 NPU 的已注册 handler：

```python
# torch_npu/_inductor/triton_experimental/config.py
disable_scatter_upon_const_tensor: bool = True

# torch_npu/_inductor/triton_experimental/fx_passes.py
def gated_extra_check(match, _original_check=original_check):
    output = match.output_node().meta.get("val")
    is_npu = getattr(getattr(output, "device", None), "type", None) == "npu"
    if ncfg.disable_scatter_upon_const_tensor and is_npu:
        return False
    return _original_check(match)
```

`default-disabled` 验证 handler/metric 均为 0；`gate-disabled` 验证 handler/metric 均大于等于 1；
两臂数值正确、backend 均为 `triton_experimental`。正式机器可读结果位于
`results/current/T-080/` 和三个 acceptance-unit 目录。

## 5. 报告与证据导航

| 主题 | 独立可读报告 | 机器可判定结果 |
| --- | --- | --- |
| Scatter 入口适配 | `issues/REF-scatter-const-cross-entropy-e2e-native/适配报告.md` | `results/current/AU-joint-graph-scatter-upon-const-tensor/npu_result.json` |
| Scatter 两处 codegen 根因 | `issues/REF-scatter-const-cross-entropy-e2e-native/根因分析.md` | 同目录 `comparison_result.json` |
| Scatter 修复、性能回退和产品门禁 | `issues/REF-scatter-const-cross-entropy-e2e-native/修复验证报告.md`、`代码合入描述.md` | `results/current/T-080/performance_summary.json`、`product_gate_verification.json` |
| Softmax generic guard 适配与产品 fallback | `issues/REF-prepare-softmax-fast-math-native/适配报告.md` | `results/current/AU-post-grad-prepare-softmax/` |
| Constructor mover 适配 | `issues/REF-move-constructors-arange-native/适配报告.md` | `results/current/AU-post-grad-move-constructors-to-gpu/` |

报告中会重复必要的代码框、调用链和边界，便于脱离其他文档单独阅读；最终自动校验字段以机器结果为准。
