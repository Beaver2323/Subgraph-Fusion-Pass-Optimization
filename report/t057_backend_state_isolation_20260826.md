# T-057 backend 全局状态隔离快照（2026-08-26）

## 结论

在 P-014 source registrar overlay 允许 default/experimental 往返后，状态探针完整执行
`default→experimental→default→experimental`。结果为
`backend-selection-functional-state-isolation-failed`：codegen backend 能切换，但 pass/config/
decomposition 全局状态没有完整恢复。当前禁止在同一进程内做 default 与 experimental 的 pass
性能归因；两侧必须使用独立 fresh process。

## 运行边界

- 环境：`/home/z50063656/Benchmark/env.sh`，当前 P-013 installed wheel；
- 设备：运行前后空闲的物理 NPU 1，仅用于 SoC 查询，没有编译或执行模型；
- cwd：`/home/z50063656/tmp`；
- overlay：只把 installed default registrar 替换成 P-014 current-source function，源码 SHA256
  `bd6ac8acea23b8347ff01fdf9095e1ea592b6508fd99a267d4ee028da07ac373`；
- 探针：`t057_backend_state_probe.py`；
- 原始结果：`results/t057_backend_state_20260826/state.json`，SHA256
  `07d070f60d5edc899bc848f720b78595187fd35483a679d0394274e2b8212cdc`。

为了避免“默认值本来就是 False”的歧义，进入 experimental 前主动把四个 Inductor config 和
可用的 Dynamo recursive-tag guard 设为 True sentinel。

## 状态对比

| 状态对象 | default sentinel | experimental first | default after experimental | 结论 |
|---|---|---|---|---|
| `layout_optimization` | True | False | False | 泄漏 |
| `coordinate_descent_tuning` | True | False | False | 泄漏 |
| `split_reductions` | True | False | False | 泄漏 |
| `shape_padding` | True | False | False | 泄漏 |
| Dynamo recursive dict tags | True | False | False | 泄漏 |
| 34 个 addmm `extra_check` | upstream `is_valid_addmm_fusion` | experimental False lambda | False lambda | 泄漏 |
| `post_grad_custom_post_pass` | `AscendCustomPostPass` | experimental composed pass | `AscendCustomPostPass` | 正确恢复 |
| matmul `should_fold` | upstream | experimental NPU wrapper | experimental NPU wrapper | 泄漏 |

addmm entry 数量始终是 34，问题不是重复注册，而是相同 entry 的 `extra_check` 被不可逆替换。
因此 default 回切仍会错误地保持 addmm fusion disabled，pointwise wrapper marker 无法发现它。

## decomposition 对比

experimental first 正确安装了 GELU、softmax backward、RMSNorm、native dropout 前后向 override；
回到 default 后：

- erfc 由 P-014 source registrar 安全重新注册；
- GELU forward 被 default cleanup 清除；
- `post_grad_custom_post_pass` 被 default loader 重设；
- 但 softmax backward、GELU backward、RMSNorm、native dropout forward/backward 仍指向
  experimental 函数；
- matmul decomposition 模块的 `should_fold` 仍是 experimental wrapper。

这证明 `restore_inductor_baseline()` 只恢复 lowering/scheduler 不够，不能把“没有 duplicate
registration”当作 decomposition 隔离完成。

## 对 P-014 的影响

P-014 的单行 erfc cleanup 仍是必要且已验证的局部修复：它消除了真实崩溃，并让状态审计可以
跑完。但 P-014 不升级为完整 backend-isolation 修复，也不扩张成大范围 registry snapshot；其
installed wheel 验证仍保持 pending。

## P-015 设计边界

新增 P-015，目标是建立可重复的 per-backend 状态恢复合同。实现前必须先决定哪些状态属于：

1. 上游 Inductor baseline；
2. torch_npu default backend；
3. triton_experimental backend；
4. 用户/`torch.compile(options=...)` 临时配置。

候选实现应在 loader 层集中 capture/restore，而不是在每个测试里手工恢复。至少覆盖：

- addmm pattern 原始 `extra_check`；
- experimental 修改的五项 config/guard；
- decomposition 表中被 experimental 覆盖或删除的 exact keys；
- `torch._decomp.decompositions.should_fold`；
- custom-pass config identity。

恢复后要求同一进程两轮往返的所有 snapshot 与各 backend 首次进入一致，再运行 addmm、GELU、
dropout、RMSNorm 和 matmul-fold 代表图。不能只比较 callable 名称；必须实际编译并检查图/数值。

## 当前执行规则

P-015 关闭前：

- default/experimental paired 必须为独立 fresh process；
- 不允许一个 worker 先运行 experimental 再把后续 default 结果当干净 baseline；
- T-057 correctness 边界可在 experimental-only fresh process 继续；
- 不重建共享 wheel，不把当前 source 中其他未安装 diff 一并带入环境。
