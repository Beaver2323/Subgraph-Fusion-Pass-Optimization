# Inductor Pass 在 NPU 上的替代与优化计划

> 当前状态（2026-08-28）：主环境已纠正为
> `/home/z50063656/Pass/activate_pass.sh` 激活的 Conda `Pass`。此前 Benchmark/独立 venv
> 结果保留为历史证据；任何新验证先在 `change_control.md` 登记，并从
> `/home/z50063656/tmp` 启动。T-011、T-023、T-029/T-031、T-036、T-038、T-040、T-042、
> T-043、T-046 与 T-047/P-012 的历史验证不因环境口径更正而被删除。

## 当前结论

当前 `Pass/src` 概念级清单为 251 条；新的动态基线是
`Conda Pass + CANN 9.0.1 + 8×Ascend910B2`，尚待按交接文档完成关键结果复核。原
`benchmark-py311` 数据保留其历史标签。P1 B2/B3 已关闭；B4 八个代表 attention family 功能
通过。pattern 1 latency-beneficial，pattern 13 resource-beneficial；5/21/29 因 additive float
mask 不满足 vendor bool/None gate 而安全 math fallback，无 mask pattern 30 走 vendor。这里不用
Triton 掩盖语义错误、环境依赖或 capability 分流。全量记录见 `report/pass_src_20260820/`。

源码证据表明，当前后端已经存在若干明确的 NPU 约束：

| 区域 | 当前证据 | 处理结论 |
|---|---|---|
| `mm_plus_mm` | 共享 pattern 的 NPU gate保持不变；T-023 另行注册 default-off different-K NPU handler/template，extern fallback-first | 首批两 shape p50 改善 15.29%/18.04%，但多 270,336 B peak allocated；保持默认关闭、output cap与fallback，等待无 shim环境复验 |
| `pad_mm` | T-025证明测试侧绕过device gate后mm/bmm/addmm图可正确承接；T-026 p50分别回退72.65%/65.31%/120.63% | 保持gate；不实现独立padding替身。未来只有大shape/特殊layout先证明收益时才评估融合masked-load GEMM |
| `addmm` fusion | Triton experimental override 可将已注册的 add+mm -> addmm entries 的 `extra_check` 置 false | 与 NPU `addmm` lowering、CATLASS 和 fallback 三路做新鲜 paired benchmark |
| `fold_reduce` | 直返输入违反 alias；clone 正确但 p50/p99 回退 3.06%/6.72% | 最终保留原 sum、禁用折叠；现有 sum 比 Triton copy 更快，不做替身 |
| `cat_to_view_pass` | alias-safe clone 正负例通过，p50 +2.29%，task 3→1、allocated peak -4,195,840 B | 保留 clone 修复，记 latency-neutral/resource-beneficial；不再手写重复 copy kernel |
| `fold_where` | where→clone 保持新 storage；device kernel 时间下降约 26.46%，但端到端 p50 仅 +1.16%、task/显存不变 | 保留既有 FX pass并记 `supported-neutral`；固定开销主导时手写另一条 Triton copy 没有意义 |
| `dtype_optimal_pass` / safe iota | 比较闭包内 int64→int32 后 p50 改善 52.06%/55.78%；直出、大 float、Inf 与 overflow 风险路径已保护 | 保留缩窄后的 FX pass；收益来自现有生成 kernel 的降宽，不另写 Triton，也不恢复危险改写 |
| `broadcast_const_mask_compress` | equal-shape 删除 where/full，但 baseline 已融合为 1 task；p50 仅 +0.30%、显存不变 | 记 `supported-neutral`；替身 Triton不能把 1 task 再减少，不投入实现 |
| `masked_add_compose_pass` / `bool_cast_mul_to_where_pass` | 浮点 signed-zero 与 `0*Inf/NaN` 反例已由整数/布尔 dtype guard 隔离；masked-add 单 task且中性；bool-cast view-chain p50/p99 +36.30%/+39.90%，direct p99 -19.02% | masked-add 保留为 neutral；bool-cast 只允许 non-empty view-chain，direct/float 保持原图；不重写重复 Triton |
| `sign_diff_hamming_fuse_pass` | 特殊浮点、整数 keepdim 正例与 multi-user 负例 NPU 通过；paired p50/p99 仅改善 3.64%/5.49%，仍为单 task | 保留现有 FX pass并记 `supported-neutral`；没有多 task 或资源瓶颈，不实现专用 Triton |
| `batch_embedding_fusion_pass` | step/dtype/alias 修复后两个 safe cohort P50 +23.50%/+43.90%、tasks 9→3/13→3，但 peak 分别 +1,831,936/+13,628,416 B，首次编译约翻倍 | 保留保守 pass并明确资源/编译 trade-off；已有 vendor+Inductor kernel，无证据支持另写 Triton |
| `fusion_attention_v3_pass` | B2 旧/v3 都是 1 个同名 FlashAttentionScore task、显存相同，v3 P50/P99 -4.85%/-31.72% | 非 A5 保留 legacy；A5 待真机。当前不是 kernel 缺失，不写 Triton attention |
| `fused_matmul_relu_pass` | 910B2 下由 `is_ascend950` gate 正确不注册 | `not-applicable`；只在 A5 环境验证现有 fused vendor op，不绕过芯片门禁 |
| `dvm_graph_fusion` | 15/15 NPU；相对 default P50/P99 +24.20%/+39.93%，compile 20.32→2.81 s，allocated peak 相同 | 保留现有 DVM codegen；虽 task 1→3，但总 device duration 更短，不用 Triton 复制 |
| DVM sum/expand 子 pass | sum 越界 dtype 会损精度，bare fp16 DVM sum 不可用；expand helper 当前 aggregate 不可达 | P-012 做 dtype/调用合同修复；sum 不写替身，expand 先评估 partition capability 列表 |
| DVM/MLIR wrapper | DVM backend 无 `torch_mlir` 仍 32/32；完整 MLIR loader 被依赖阻塞 | DVM 结论保留；MLIR 在独立匹配环境补测，不能用 Triton 替代缺失 Python/MLIR 依赖 |
| NPU custom FX pass | `ascend_custom_passes` 注册 27 个 pass，主要是 fold/view/cat/reduce/attention/embedding 等图重写 | 先做图正确性和 kernel 数量检查；这些 pass 通常不需要手写 Triton |
| `inductor-npu-ext` | 通过 `pre_grad_custom_pass` / `post_grad_custom_pre_pass` 注册 legacy scatter、pad-slice、batch-embedding pass | 作为 AscendC 后端独立测量，不能与 Triton backend 的结果混用 |
| attention | B2 legacy→v3 在 910B2 已性能拒绝；B4 pattern 1 latency-beneficial、pattern 13 resource-beneficial；pattern 5 rewrite P50 回退 103.23% | 非 A5 保留 legacy；1/13 继续 vendor；P-013 精确停用 pattern 5 后 P50 +50.28%、task 8→3；继续测 21/29，当前不手写完整 attention |

## Pass 分层与替代策略

| 优先级 | Pass 类型 | 代表记录 | 首选实现 | 何时考虑手写 Triton |
|---|---|---|---|---|
| P0 | 已明确被 NPU backend 拒绝的 lowering/fusion | `mm_plus_mm`、`pad_mm`、`addmm` fusion | NPU lowering + CATLASS/AscendC/vendor op | 只有在 NPU Triton 已支持所需 `dot/layout/atomic` 且 profile 证明收益时 |
| P0 | 大幅减少 kernel 数量的 NPU custom pass | `fusion_attention_v3_pass`、`batch_embedding_fusion_pass` | 图重写到 NPU vendor op 或 AscendC fused kernel | vendor op 缺失且融合结构稳定时，做专用 Triton kernel |
| P1 | 内存带宽受限的 elementwise/reduction fusion | pointwise fusion、`masked_add_compose_pass`、`bool_cast_mul_to_where_pass` | 保持通用 Inductor fusion，校验 NPU Triton scheduling | fallback 产生多个 kernel、profile 显示 launch/读写开销主导时 |
| P1 | 纯图清理/布局折叠 | `fold_*`、`cat_to_view_pass`、`repeat_to_expand_pass`、`constant_fold_uniform_value` | 继续用 FX pass，不生成 kernel | 不应手写 Triton；替代会增加 launch 和维护成本 |
| P2 | 通信与调度 | bucketing、DDP/FSDP、overlap、multi-stream scheduler | HCCL/NPU stream 原生实现，分布式 paired run | Triton 不能替代 collective；禁止用单卡结果下结论 |
| P2 | CPU/CUDA 专属路径 | MKLDNN、CUDA graph、CUDA-only decomp | 明确标记为 NPU 不适用或实现 NPU 分支 | 只有算子本身可在 NPU Triton 表达时才评估 |

## 测量矩阵

`run_npu_probe.py` 用以下代表性图触发 pass 组，并在每个 backend 独立进程中记录观察到的 pass 名称：

| Case | 主要 pass 组 | 正确性 | 性能指标 |
|---|---|---|---|
| `pointwise` | vertical/pointwise fusion | eager 对比，fp16 默认 `rtol=atol=1e-2` | first-run、稳态 mean/stdev/p50/p99、peak memory |
| `mm_plus_mm` | `mm_plus_mm`、GEMM lowering | eager 对比 | 同上，另外记录 CATLASS/ATen/fallback choice |
| `cat_slice_cat` | `split_cat`、`cat_to_view` | eager 对比和 stride | kernel 数、稳态 latency |
| `reduce` | reduction/partial-reduction/locality | eager 对比 | reduction kernel 数、稳态 latency |
| `layernorm` | reduction + pointwise + layernorm lowering | eager 对比 | first-run 与稳态差值 |
| `embedding` | `batch_embedding_fusion_pass` | eager 对比 | embedding/reduce kernel 数、峰值内存 |
| `attention` | `fusion_attention_v3_pass` / SDPA lowering | eager 对比 | vendor op/Triton/fallback 路径与 p50/p99 |

每个 backend 必须新进程运行：`triton`、`ascendc`、`mlir`。性能结论必须同时记录 CANN 版本、SoC、PyTorch/torch_npu 版本、warmup、runs、均值、标准差、p50、p99、峰值内存。基线和候选实现必须是同一机器、同一输入、同一编译配置下紧邻执行的 paired run。

## Triton 替代的准入条件

1. 先确认是 pass 产物无法 lowering，还是 lowering 成功但性能差；不能只看到 graph break 或 fallback 就直接写 kernel。
2. 先有 eager/reference 和 NPU vendor/AscendC 对照；handwritten Triton 必须通过相同 dtype、shape、stride、动态 shape、训练反向场景。
3. 只在 profile 证明收益时替代：稳态 p50 至少改善 10%，且 p99、峰值内存和编译时间没有超过门槛；阈值按业务模型另行确认。
4. 对 GEMM、attention、collective 优先复用 NPU 专用实现；Triton 适合作为 elementwise/reduction 或缺失融合 epilogue 的候选，不应默认替代 vendor kernel。
5. pass 默认关闭，通过独立环境变量或 backend capability gate 灰度开启；出现精度/shape/编译失败时自动回退到原始图。

## 下一步执行

当前主线不再重跑全量旧探针。pad family、B2/B3 已完成，B4 八个代表 family 功能、pattern 1/13/5 性能与 P-013 已关闭。下一步按 `p1_batch_design.md` 做 pattern 21/29 paired，再扩到剩余 attention。完整 MLIR 与 T-023 环境支线只在匹配依赖/headers 的独立环境补测。以下命令仅作为重新生成静态清单/广域探针的参考，必须从`/home/z50063656/tmp`运行并改用当前`Pass/src`路径：

```bash
python /home/z50063656/Pass/inductor_pass_npu_audit/audit_passes.py \
  --pytorch-root /home/z50063656/Pass/src/pytorch \
  --torch-npu-root /home/z50063656/Pass/src/torch_npu \
  --output /home/z50063656/Pass/inductor_pass_npu_audit/report/pass_src_20260820

python /home/z50063656/Pass/inductor_pass_npu_audit/run_npu_probe.py \
  --output /home/z50063656/Pass/inductor_pass_npu_audit/report \
  --backends triton,ascendc,mlir --warmup 10 --runs 100
```

然后按 `npu_probe.json` 的 `observed_passes` 与 `pass_inventory.json` 对齐，逐条补齐 `available`、`unsupported`、`fallback`、`slow` 四种运行结论；不要把 `skip` 当成 `pass`。
