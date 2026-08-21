# T-012 mm_plus_mm different-K 当前 fallback 成对基线

## 结论

`mm_plus_mm_positive_different_k` 在 `triton_experimental` backend 下功能正确，current 模式每轮都命中一次目标 pattern，disabled 模式每轮都成功禁用一个目标 entry 且匹配数为 0。12/12 个性能 worker 全部 `compile-correct`，compiled 对 eager 的最大绝对误差均为 0。

current 相对 disabled 的三轮 p50 中位数变化为：shape-A 回退 0.30%，unaligned 改善 2.53%。两者都低于项目 10% 主门槛，应判为 `supported-neutral` 的 different-K 现状证据。它说明“匹配后再安全 unfuse”没有带来稳定运行时收益，但不能单独证明专用 Triton/AscendC/vendor kernel 一定有收益。

## 方法

- 环境：Conda `Pass`；PyTorch `2.14.0a0+git8e86e0a`；torch_npu `2.14.0a0+git83cc452`；Triton Ascend metadata `3.2.2`；CANN 9.0.1；Ascend 910B2。
- 设备：物理 NPU 6；功能哨兵前、性能采样前和性能采样后均无其他进程。
- backend/case：`triton_experimental` / `mm_plus_mm_positive_different_k`。
- 输入：fp16、contiguous、static；shape-A `(M,K1,N)=(192,256,320)`、`K2=128`；unaligned `(191,255,319)`、`K2=127`。
- 对照：同一 backend 的 `current` 与 `disabled`，每个 worker 为 fresh process；第 2 轮反转执行顺序。
- 计时：warmup 10、runs 100、3 轮；汇总使用每种模式的三轮中位数，收益定义为 `(disabled-current)/disabled`。
- cache：设置 `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`，避免跨 worker 复用编译产物。

## 触发与正确性

| 阶段 | current | disabled |
|---|---:|---:|
| 单轮 debug 哨兵 | 1/1 compile-correct，pattern match=1 | 1/1 compile-correct，patched entry=1，pattern match=0 |
| 两 shape × 三轮性能 | 6/6 compile-correct，pattern match 每轮为 1 | 6/6 compile-correct，patched entry 每轮为 1，pattern match 每轮为 0 |
| 最大绝对误差 | 0 | 0 |

成功用例已设置图模式调试目录；按图模式验证规则，本轮没有展开成功 run 的 `output_code.py`。current 的 different-K 安全 unfuse 行为沿用此前已记录的源码和语义矩阵证据。

## 三轮性能

| shape | current p50 三轮（ms） | disabled p50 三轮（ms） | p50 中位数变化 | p99 中位数变化 | mean 中位数变化 |
|---|---|---|---:|---:|---:|
| shape-A | 0.298635 / 0.511125 / 0.321445 | 0.338765 / 0.320190 / 0.320490 | 回退 0.30% | 改善 1.68% | 改善 0.29% |
| unaligned | 0.305330 / 0.328370 / 0.321665 | 0.329670 / 0.333305 / 0.330000 | 改善 2.53% | 改善 8.72% | 改善 4.53% |

shape-A current 第 2 轮出现 p50 `0.511125 ms`、p99 `1.682840 ms` 的明显尾部抖动，原始值保留。三轮中位数避免该轮直接支配结论；即便忽略该异常轮，shape-A 第 3 轮的 p50 也只回退 0.30%，没有形成稳定收益。

## 编译与内存

| shape | current 首次编译+执行中位数 | disabled 中位数 | current 相对变化 | current/disabled 峰值 allocator |
|---|---:|---:|---:|---:|
| shape-A | 21.220 s | 19.640 s | 慢 8.05% | 2,122,752 / 2,122,752 bytes |
| unaligned | 19.663 s | 19.739 s | 快 0.39% | 1,981,440 / 1,981,440 bytes |

编译时间没有一致方向，峰值 allocator 在同一 shape 下完全相同。因此当前证据不支持为了 different-K 保留一次无承接的 pattern 匹配而宣称编译或内存收益。

## 工程判断与下一闸门

1. 当前 different-K 路径功能可用且 fallback 安全，但性能为 neutral。
2. 不能直接解除现有 Triton template 的 size guard；第二条 K 循环需要独立的 K2 边界、mask 和 autotune 设计。
3. 后续 T-013 已完成 current fallback 的 kernel 级 profile：每次执行为两个 aclnnMm + 一个 Triton add，步内 gap+add 的端到端理论上限为 17.68%/16.02%。
4. 该上限允许在审计目录做不接入源码的 different-K 微原型；候选仍需三 dtype、非对齐、转置、dynamic、forward/backward 和三轮 paired benchmark。
5. 在候选 p50 稳定超过 10% 且 p99、编译时间、峰值内存可接受前，P-005 不进入功能源码实现。

## 证据

- 本地功能哨兵：`results/t012_mmplus_different_k_sentinel_20260821/`（运行产物不纳入本仓库）
- 本地性能原始 JSON：`results/t012_mmplus_different_k_paired_20260821/p0_gate_probe.json`（不纳入本仓库）
- 变更与测试合同：[`change_control.md`](../change_control.md)
- 前置语义证据：[`p0_semantic_matrix_20260821.md`](p0_semantic_matrix_20260821.md)
- 后续 kernel profile：[`t013_mmplus_different_k_profile_20260821.md`](t013_mmplus_different_k_profile_20260821.md)
