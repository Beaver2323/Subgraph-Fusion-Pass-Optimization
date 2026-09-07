# T-039 dtype/index/mask safe-positive NPU 性能报告

日期：2026-08-25  
范围：`dtype_optimal_pass`、`fold_iota_arithmetic_pass`、
`broadcast_const_mask_compress`  
环境：PyTorch `2.14.0a0+git8e86e0a`、torch_npu
`2.14.0a0+git83cc452`、Triton 3.2.0、CANN 9.0.1、Ascend910B2

## 结论

T-038 修复后的三个安全正例均完成三轮单 pass paired benchmark。`dtype_optimal_pass`
和 `fold_iota_arithmetic_pass` 明确通过预登记性能门槛：三轮中位 p50 分别改善
**52.06%**、**55.78%**，p99 分别改善 **50.48%**、**54.51%**，额外 allocated peak
不增加。因此两项在当前 safe cohort 上均为
`supported-beneficial-development-audit-shim`。

`broadcast_const_mask_compress` 的图化简与完整语义正确，但三轮中位 p50 只改善
**0.30%**，task 数和显存不变，归类为 `supported-neutral-development-audit-shim`。
删除 FX 节点不等于端到端性能提升；本轮不为它编写替身 Triton kernel，也不扩大 pass
capability。

所有结论都受 audit-only launcher 垫片限制：它们证明当前 wheel 中 pass 与 NPU 生成
kernel 的开发态相对性能，不能替代匹配 C++/header 合同后的无垫片正式复验。

## 方法与门禁

- 每条 pass 使用 `B1,C1,C2,B2,B3,C3` 六个 fresh process，baseline 仅跳过目标 pass，
  candidate 正常执行目标 pass；每个 worker 使用独立 Inductor/Triton cache。
- 稳态时间为 warmup 10、runs 100；显存为 warmup 3、runs 10。B1/C1 另做 profiler
  warmup 1、active 10。
- 每次计时前后都检查严格逐元素值、mismatch count、shape、dtype、stride、
  `requires_grad`、输入 alias 与目标图门禁。18/18 worker 的前后 mismatch count 都为 0。
- 放行要求：三轮中位 p50 改善严格超过 10%，p99 不回退超过 5%，additional allocated
  peak 不增加。仅 task 数或显存明确下降且 p99 合格时才允许 resource-only verdict。
- 物理 NPU 1 在批次前后均无外部进程。测量遵循 NPU 性能分析约束，记录 CANN、NPU、
  warmup/runs、mean±stdev、p50/p99、设备任务与显存，而非只给单次耗时。

## 汇总结果

下表中的 mean、stdev、p50、p99 均为三个 fresh worker 对应统计量的中位数。profiler
device duration 是 10 个 active step 的总和；括号内为每个 task 的均值。

| pass | baseline mean±stdev (ms) | candidate mean±stdev (ms) | p50 改善 | p99 改善 | profiler device duration | task/step | allocated peak 变化 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dtype_optimal_pass` | 0.55154±0.00556 | 0.26471±0.00492 | 52.06% | 50.48% | 3591.94→67.78 µs（359.194→6.778 µs/task） | 1→1 | 0 B | beneficial |
| `fold_iota_arithmetic_pass` | 0.55932±0.00712 | 0.24804±0.00601 | 55.78% | 54.51% | 3584.54→54.88 µs（358.454→5.488 µs/task） | 1→1 | 0 B | beneficial |
| `broadcast_const_mask_compress` | 0.24537±0.00426 | 0.24449±0.00460 | 0.30% | 1.01% | 36.90→35.46 µs（3.690→3.546 µs/task） | 1→1 | 0 B | neutral |

三个 case 的 additional allocated peak 基线/候选分别同为 1,049,088 B、1,049,088 B、
4,194,816 B；additional reserved peak 均为 0。首次 compile+run 的三轮中位数约为
17.1–17.9 秒，两侧相近且不参与性能放行。

## 为什么两个降宽 pass 收益很大

`dtype_optimal_pass` 的候选把只被布尔比较消费的 int64 arange 和 `.to(int64)` 都降为
int32；`fold_iota_arithmetic_pass` 把安全范围内的 int64 iota 降为 int32。两者仍生成一个
融合 Triton task，收益不是“少发 kernel”，而是 NPU 上该生成代码避免了昂贵的 int64
逐元素路径。profiler 中每 task 约 359 µs 降到 5–7 µs，与端到端约 52%–56% 改善方向
一致。

这个结果不能反推“所有 int64 都应降为 int32”。T-038 已证明直出 dtype、大 float 转换、
Inf/NaN 与整数溢出会破坏语义；收益只属于比较闭包和静态安全范围内的 capability。

## 为什么 mask 化简仍是中性

候选图确实把 where `1→0`、full `2→0`，但 Inductor 在 baseline 中已把
`full→where→convert` 融成单个逐元素 Triton kernel。因此两侧均为 1 task/step，额外显存
相同；device kernel 仅由约 3.690 µs/task 降到 3.546 µs/task，远小于 host/runtime 固定
开销，也只有一次 profiler 捕获，不能作为 resource-only 放行依据。三轮端到端 p50 仅
0.30%，所以保守记 neutral。

## 原始证据与中性尝试

- worker 与 profiler：
  `results/t039_dtype_index_mask_performance_20260825/`。
- 最终聚合：各 case 的 `*_aggregate_final/aggregate.json`。早期 `*_aggregate/` 是聚合器
  审计中间产物；初版错误地把单次 device duration 略降也算 resource 改善，发现与
  预登记规则不一致后收紧为只接受 task/显存下降。原始性能样本未重跑、未筛选。
- 首次批量聚合 shell 在 `source env.sh` 后因环境脚本返回值以退出码 1 静默终止；改用已知
  Python 解释器逐 case 聚合后成功。这是汇总命令问题，不是 pass 或 NPU worker 失败。
- 功能/launcher 环境失败与中性尝试见
  `t038_dtype_index_mask_semantic_fix_20260825.md`，本轮没有把 audit shim 当成正式环境通过。

## 决策与下一步

1. 保留 T-038 的保守语义 guard；`dtype_optimal_pass` 与 safe iota downcast 已同时获得功能
   和性能证据，不恢复已证伪的激进 cmp-sub 改写。
2. `broadcast_const_mask_compress` 保持功能可用、性能中性。当前无需手写 Triton：baseline
   已融合成一个 kernel，替身无法减少 task；除非新 cohort 显示真实带宽/任务缺口，否则
   不投入 kernel 实现。
3. P1 B2 继续进入尚未关闭的 9 个 custom pass；同时把三项 T-039 结论写回 251 条评估
   矩阵。正式无 shim launcher 复验仍作为独立环境支线。
