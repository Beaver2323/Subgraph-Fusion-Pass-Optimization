# T-024 mm_plus_mm different-K workspace 审计（2026-08-21）

## 结论

T-024 没找到同时通过显存和 task-duration 门槛的替代配置。它证明 T-023 的 384 KiB workspace 可以通过减少 launch grid 降到 192 KiB，但在当前小 shape 上会因并行度减半使 task p50 超过预登记的 10 μs 筛选线。按合同停止搜索，未修改 torch_npu 产品源码、未重建或重装 wheel。

## 闸门

- shape-A、fp16、contiguous、static：`(M,K1,N,K2)=(192,256,320,128)`。
- 正确性：`rtol=atol=0.01`，输出有限且 dtype/shape一致。
- profiler：warmup 1、active 10，要求 10/10 每 step 唯一同名 task。
- 显存：candidate additional allocated peak不高于 `246,784 + 123,392 = 370,176 B`。
- task 筛选：p50 不高于 10 μs；超过即不进入 unaligned 和集成 paired benchmark。
- 每个配置 fresh compile、独立结果和 Triton cache；从 `/home/z50063656/tmp` 启动。

## 七个配置

| 配置 | grid | per-block/total workspace | additional peak | task p50 | 结果 |
|---|---:|---:|---:|---:|---|
| 128×128×128, multibuffer=True | 6 | 65,536 / 393,216 B | 517,120 B | 9.02 μs | 性能过、显存不过 |
| 128×128×128, multibuffer=False | 6 | 65,536 / 393,216 B | 517,120 B | 8.97 μs | 关闭选项无显存效果 |
| 256×128×128 | 3（计划值） | 未生成 | 未运行 | 未运行 | 编译失败：L0C 需求 2,097,152 bit > 1,048,576 bit |
| 128×256×128 | 4（计划值） | 未生成 | 未运行 | 未运行 | 编译失败：同上 |
| 256×256×128 | 2（计划值） | 未生成 | 未运行 | 未运行 | 编译失败：L0C 需求 4,194,304 bit > 1,048,576 bit |
| 128³ linear group2 | 3 | 65,536 / 196,608 B | 320,512 B | 10.42 μs | 显存过、性能不过 |
| 128³ M-group2 | 3 | 65,536 / 196,608 B | 320,512 B | 10.54 μs | 显存过、性能不过 |

所有成功编译配置的 max/mean absolute error均为 0，且 profiler 为 10/10 单 task。`multibuffer=False` 不改变编译 callback 给出的 per-block workspace。大 accumulator tile 直接触碰 L0C 硬上限，不能通过 launcher 或环境参数解决。

## grouped-program 解释

group2 保留 128³ 片上 accumulator，让一个 physical program 在内层顺序处理两个 logical output tile：

```text
logical output tiles: 6
tiles per program:    2
launch grid:          3
workspace:            64 KiB × 3 = 192 KiB
```

linear group2 按扁平 logical id 配对；M-group2 固定 N tile并连续处理两个 M tile，以尝试复用 B/D。两者都把实测 peak 降到 320,512 B，但 p50 为 10.42/10.54 μs；局部性变化没有补回减少物理 program 带来的并行度损失。

虽然该 1.4–1.5 μs device 增量未必会让端到端收益跌破 10%，预登记合同已经要求先过 10 μs direct screen。项目没有事后放宽阈值或用 host 同步开销稀释 device 回退，因此没有运行集成 paired，也没有接入产品。

## 源码和产物

- audit worker：`t024_mmplus_different_k_workspace_screen.py`
- audit-only grouped kernel：`t014_mmplus_different_k_triton.py`
- 结果：`results/t024_mmplus_different_k_workspace_20260821/`
- 决策记录：`change_control.md:E-087` 至 `E-091`

T-024 只影响审计目录。T-023 产品 kernel仍是 128×128×128、每个 program一个 output tile。
