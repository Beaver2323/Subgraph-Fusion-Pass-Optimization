# T-043～T-046：B2 最后三个复合 custom pass 闭环

## 结论

截至 2026-08-25，B2 批次最后三个 pass 已有独立结论：

| pass | 910B2 结论 | 关键原因 |
|---|---|---|
| `batch_embedding_fusion_pass` | `supported-neutral-resource-beneficial` | 两个 safe cohort 的稳态延迟与任务数显著改善，但 allocated peak 和首次编译增加，完整性能 gate 未过 |
| `fused_matmul_relu_pass` | `not-applicable` | 当前 pass 与 fused op 都由 Ascend950 capability gate 控制，910B2 不注册；不能外推 A5 结果 |
| `fusion_attention_v3_pass` | `supported-pass-disabled-performance-rejected` | B2 上旧/v3 都落到同一个 FlashAttentionScore task，v3 P50/P99 回退；最终仅在 Ascend950 保留升级能力 |

本批没有手写 Triton。batch 已有工作的 vendor embedding 与 Inductor pointwise/reduction
kernel，真实缺口是图语义保护和内存/编译 trade-off；attention 旧/v3 已调用同一设备 kernel，
再写一份 Triton attention 既不能解释当前差异，也会扩大精度、tiling 与 workspace 风险。

## 1. T-043 静态反例与源码修复

修复前的 CPU/FX 反例记录在
`results/t043_b2_composite_static_20260825/result.json`：

- batch slice `step=2` 被错误融合，两个输出最大绝对误差为 `9` 和 `129`；
- 显式 `sum(dtype=float32)` 被改成 float16 输出；
- 普通 batch 正例数值误差为 0，但两个原本独立的 reduce 输出改写后共享 storage；
- legacy attention 有 24 参数/7 返回，v3 有 21 参数/6 Tensor 返回；旧实现会传递多余参数、
  生成 `getitem(v3, 6)`，并把旧 7-tuple meta 复制给 v3；
- inference PRE runner 会调用 `fusion_attention_v3_pass` 两次；
- 910B2 的 `is_ascend950=false`，`fused_matmul_relu_pass` 不注册且 fused op 不解析。

P-010 对 `ascend_graph_pass.py` 和 PRE runner 做了保守修复：

- batch 只接受可静态证明的非负 dim、`step==1` 和默认 reduce dtype/options；
- 非 cat-collapse 输出在 select 后插入 contiguous clone，恢复独立 storage；cat-collapse 不插 clone；
- attention 只承接 v3 共有参数、`actual_seq_* is None` 且全部用户为索引 0～3 的 getitem；
- v3 节点重新运行 fake op 生成 6-tuple meta，失败即保持旧图；
- inference PRE 列表只执行一次，training 仍只允许 attention 特例。

源码构建前纯 FX 测试为 84/84；P-011 增加芯片门禁测试后为 85/85。源码测试使用已安装
`_C` 加 source `_inductor` 路径的隔离引导，并在日志中断言实际
`ascend_graph_pass.py` 来源，避免把 installed 文件误当 source。

## 2. T-044 910B2 功能验证

P-010 wheel 下，NPU1 的 5/5 fresh default/fullgraph worker 均通过：

- batch default/cat 正例：embedding/sum `2/2→1/1`；default 有 2 个 clone 且两个输出仍不 alias，
  cat 路径无 clone；
- slice-step/reduce-dtype 负例：embedding/sum 均保持 `2/2`；
- attention safe output0～3：legacy/v3 `1/0→0/1`，shape、dtype、stride、alias 与值完全一致；
- 所有有效 worker 合计 mismatch 为 0。

聚合证据为
`results/t044_b2_composite_compile_20260825/aggregate/aggregate.json`。最初未带 include shim
的 `ATen/ATen.h` 缺失作为中性环境失败保留，不计入功能结果。

## 3. T-045 paired 性能

环境固定为 Ascend910B2、CANN 9.0.1、PyTorch `2.14.0a0+git8e86e0a`、
torch_npu `2.14.0a0+git83cc452`。每侧 3 个 fresh process/cache，顺序
`B1-C1/C2-B2/B3-C3`，每轮 warmup 10、runs 100、memory 3/10；round1 另做
warmup1/active10 profiler。主 gate 要求 P50 改善严格超过 10%、P99 回退不超过 5%、
allocated peak 不增加。

| cohort | P50 baseline→candidate | P99 baseline→candidate | tasks/step | additional allocated peak | 首次编译中位数 | verdict |
|---|---:|---:|---:|---:|---:|---|
| batch default + alias-safe clones | `0.528130→0.404005 ms`（+23.50%） | `0.634600→0.664170 ms`（-4.66%） | `9→3` | `17,041,920→18,873,856 B` | `20.98→45.41 s` | neutral/resource-beneficial |
| batch cat-collapse | `0.757215→0.424760 ms`（+43.90%） | `0.915350→0.472880 ms`（+48.34%） | `13→3` | `5,245,440→18,873,856 B` | `20.92→46.54 s` | neutral/resource-beneficial |
| attention legacy→v3 | `0.333150→0.349320 ms`（-4.85%） | `0.416890→0.549120 ms`（-31.72%） | `1→1` | `49,286,144→49,286,144 B` | `2.752→2.764 s` | performance-regressed |

batch 的 profiler 显示 default tasks `9→3`、cat tasks `13→3`，说明 steady 收益确实来自
更少的 embedding/slice/reduction 调度；但 fused 中间结果扩大峰值，不能只按 P50 宣称全面
有益。attention 两侧均是同名
`aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore`，candidate P50 三轮均慢，
P99 又在 2/3 轮明显变差，因此触发关闭而不是继续写 kernel。

三个 aggregate 分别位于：

- `results/t045_b2_composite_performance_20260825/batch_default_clone/aggregate/aggregate.json`
- `results/t045_b2_composite_performance_20260825/batch_cat_collapse/aggregate/aggregate.json`
- `results/t045_b2_composite_performance_20260825/fusion_attention_v3/aggregate/aggregate.json`

最初 batch baseline 使用 `1e-3` 容差时，把合法 fp16 sum/add rounding 判成失败；原结果保留，
重试使用 `rtol=atol=1e-2`，同时继续严格检查 dtype、shape、stride、NaN、语义和图门禁。

## 4. T-046 B2 性能拒绝落地

P-011 在 `fusion_attention_v3_pass` 入口增加 `if not is_ascend950: return`。A5 路径仍保留
P-010 的 schema、用户和 meta guard；当前没有 A5 设备，因此只标为待验证。

新 wheel：

- 文件：`torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`
- SHA256：`beee993d4c803ed72d26284dcdc06eac97cedaf450a54398ec11285d2711d54b`
- 1318 个唯一 archive 条目；产品 pass/runner 模块与 source byte-equal；包含 `_C` 与
  `libtorch_npu.so`；不含 TorchAir、`libtensorpipe.so` 或 legacy egg-info；
- 通过 `pip install --no-deps --force-reinstall` 安装；source/installed 完整 FX 均为 85/85。

P-010 回滚 wheel 已保存为
`artifacts/torch_npu_t043_before_t046_attention_b2_gate.whl`，SHA256
`44f2aad2465d59d6285fcd17739186a9560f90483dfa4e5de92948e848e461d8`。

最终 NPU1 worker 见
`results/t046_attention_b2_gate_compile_20260825/worker/result.json`：设备为 Ascend910B2，
legacy/v3 `1/0→1/0`，四个输出的 value/dtype/shape/stride/input alias/output alias 全通过，
累计 mismatch 为 0。测试结束后 NPU1 无残留进程。

## 5. 阅读提示

“数值误差为 0”在本批再次不是充分条件：修复前 batch 普通正例仍破坏 output-output alias，
显式 reduce dtype 反例即使选定输入恰好可精确表示也改变了 dtype。pass 的可用性合同至少应
同时覆盖值、dtype、shape、stride、storage alias、图是否真正命中，以及性能和资源门禁。
