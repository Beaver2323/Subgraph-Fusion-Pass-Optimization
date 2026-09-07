# mm_plus_mm different-K standalone 覆盖扩展报告

## 1. 结论

T-017 至 T-019 将 standalone candidate 从 fp16/contiguous/static forward 扩展到三 dtype、真实转置 stride、dynamic replay 和训练语义。全部功能用例通过，但 dynamic 新 shape/divisibility 可能触发一次 Triton specialization 编译；新 dtype/layout 还没有 paired 性能数据，因此当前仍不能正式接入。

## 2. Dtype 与 layout

| 配置 | baseline | candidate/repeat | 关键证据 |
|---|---|---|---|
| fp16/shape-A/contiguous | max/mean abs `0/0` | `0/0` | 既有路径回归通过 |
| bf16/shape-A/contiguous | max/mean abs `0/0` | `0/0` | 输出保持 bf16，容差 `3e-2` |
| fp32/shape-A/contiguous | max/mean abs `0/0` | `2.8610e-05/1.4306e-06` | 输出保持 fp32，容差 `1e-4` |
| fp16/unaligned/transposed | max/mean abs `0/0` | `0/0` | 四输入均 non-contiguous，实际 stride 正确承接 |

转置输入 stride 为 `(1,191)`、`(1,255)`、`(1,191)`、`(1,127)`；launcher 不再用 contiguous 人为 gate，kernel 的四组 stride 参数完成真实寻址。

## 3. Dynamic replay

shape-A 和 unaligned 都只创建一个 `torch.compile(dynamic=True)` baseline callable，first/replay 后 Dynamo counters 均为 `unique_graphs=1`、AOTAutograd `ok=1/total=1`。

| profile | candidate first | candidate replay | 结果 |
|---|---|---|---|
| shape-A | max abs `0` | max abs `0.015625` | first/replay/repeat 均通过 fp16 容差 |
| unaligned | max abs `0` | max abs `0.015625` | first/replay/repeat 均通过 fp16 容差 |

shape-A 首轮的新 replay shape 曾出现约 3.66 s Triton 编译；磁盘 cache 已形成后，相同 replay specialization 首次/立即 repeat 为 3.562/0.290 ms。unaligned replay/repeat 为 0.380/0.278 ms。由此只能判定“dynamic 语义可用，新的 shape/divisibility 可能产生一次 specialization 编译”，不能声称任意新 shape 都零编译开销。这些单次值不是稳态 benchmark。

## 4. Backward 承接

PyTorch `compile_fx.py` 的顺序是 AOTAutograd 先创建 joint graph并切分 forward/backward，再分别调用 forward/backward compiler；`post_grad_passes()` 会分别作用于 forward 和 backward graph。因此正式 different-K forward fusion 不要求直接 Triton kernel暴露 eager autograd，梯度应由 AOTAutograd 生成的 backward graph承接。

T-019 同时验证了：

- 当前 compiled safe fallback 在 shape-A/unaligned 的 output 与 4 个输入梯度全部 max/mean abs `0/0`；
- 仅限审计的 `torch.autograd.Function` 使用 candidate forward，并以 `dA=grad@B.T`、`dB=A.T@grad`、`dC=grad@D.T`、`dD=C.T@grad` 承接 backward；两个 shape 的 output 和 4 个梯度也全部 `0/0`；
- AOTAutograd 两次均为 `ok=1/total=1`，成功产生独立 forward/backward debug trace；按成功分流不展开 `output_code.py`。

wrapper 只证明 standalone 训练语义可闭环，不是正式接入方式。真实 integration 必须在 post-grad/lowering 后重新验证编译出的 forward/backward 图。

## 5. 设备与结论边界

T-017 使用当时空闲的物理 NPU 7；T-018/T-019 因外部任务变化改用当时空闲的物理 NPU 6。所有阶段只做功能或编译诊断，不做跨卡性能比较，也没有终止外部进程。

当前功能状态可写为：`standalone-functional-dtype-layout-dynamic-backward-covered`。性能状态仍只对 T-016 的 fp16/contiguous/static 两 shape成立。下一步必须对 bf16、fp32、transposed 和更大 shape做同配置 fallback/candidate paired benchmark，并复核 additional peak memory，才能设计 capability gate。

## 6. 证据

- `results/t017_mmplus_different_k_coverage_20260821/`
- `results/t018_mmplus_different_k_dynamic_20260821/`
- `results/t019_mmplus_different_k_backward_20260821/`
- [T-014–T-016 正确性/profile/性能报告](t014_t016_mmplus_different_k_candidate_20260821.md)
