# T-064 triton_experimental reduction tiling 验证报告

## 1. 结论

T-064 / TE-CG-007 已完成。`rtree_real_block`、autotune、reduction where elision 和
small-outer flattening 均完成运行态覆盖；同时发现并修复两个独立正确性缺陷：

- P021：zero-kept-x scalar reduction 被错误提升，引用未定义 `xmask`；
- P022：flattened R-tree 使用错误 loop bound 和 direct inner index，造成静默错算。

安装态 target UT 分别为 P021 6/6、P022 7/7；最终选定 NPU 正确性矩阵 18/18 通过，
P022 flatten 开/关性能样本 6/6 有效。

最终状态：
`verified-with-p021-p022-keep-flatten-disabled`。

`rtree_real_block` 在携带 P021 kept-axis guard 后保持默认开启；
`flatten_small_outer_rnodes` 虽恢复 opt-in 正确性，但代表 workload 的 device P50 回退
147.96%，因此保持默认关闭。当前不提交、不推送。

## 2. 审计范围

| 配置 | 默认值 | 结论 |
|---|---:|---|
| `rtree_real_block` | True | 携带 P021 guard 后保留 |
| `rtree_real_block_autotune` | True | kept-x 正例在 on/off 均通过 |
| `elide_reduction_where` | True | sum/max on 省去 `tl.where`，off 保留 |
| `flatten_small_outer_rnodes` | False | P022 修正 opt-in，默认仍关闭 |
| small-outer cap | 2048 | 仅作为 flatten applicability tuning knob |

P021 与 P022 均以 detached 基线 `83cc452480c3546fd5cccf853bfe3a360ce9dbfc`
构建独立 wheel，并使用独立 venv 验证。

## 3. P021：zero-X promotion guard

修复前 `rtree_scalar_total` 在默认 `rtree_real_block=True` 下生成：

```text
tmp2_broadcast_guard = tl.where(r0_1mask & xmask, tmp0, 0.0)
NameError('xmask is not defined')
```

关闭 real-block 后同图通过，定位为 promotion applicability 缺口。P021 新增
`_npu_rtree_promotion_has_kept_axis()`，仅当非 reduction tree 中存在未被 mapping
消去的自由 node 时允许 promotion。

修复后默认 scalar total 的 guard trace 为 `calls=1/accepted=0/rejected=1`；动态 kept-x
正例为 `calls=1/accepted=1/rejected=0`，说明修复没有全局禁用优化。

## 4. P022：flattened R-tree bound/index

P021 状态下显式打开 `flatten_small_outer_rnodes` 后，动态多轴 reduction 只遍历
`r0_1numel`，最大绝对误差为 `203.0915/181.1489`。第一版只改 composite bound 后仍
错算，证明 direct-index rewrite 也不适用于 flattened linear loop。

P022 为 flatten 状态保存 composite `r0_numel` code，统一三处 loop rewrite，并只在
nested 模式把 inner modulo 替换为 direct index。最终 generated code 同时满足：

```text
for r0_offset in range(0, r0_numel, R0_BLOCK):
r0mask = r0_index < r0_numel
r0_1 = r0_index % ks0
r0_2 = r0_index // ks0
```

FP32 dynamic 两个 shape 均通过，最大绝对误差 `4.577637e-05`。

## 5. NPU 正确性矩阵

所有测试从 `/home/z50063656/tmp` 启动，使用 fresh process、独立 cache、安装态 wheel，
并与 eager NPU 对照。

| 分组 | 覆盖 | 结果 |
|---|---|---:|
| P021 zero-x | FP16/BF16 static、FP32/BF16 dynamic、默认 blocker | 5/5 |
| P021 kept-x/开关 | default、autotune off、rtree off | 3/3 |
| P021 where elision | sum/max × on/off | 4/4 |
| P022 flatten | FP32 on/off、FP16 on、BF16 on | 4/4 |
| P022 cumulative regression | zero-x、kept-x default | 2/2 |

合计 18/18。FP16/BF16 reduction 使用 dtype-scaled tolerance，最大绝对误差分别为
`0.125/1.0`；严格审计阈值失败样本保留在结果目录，不包装成结构通过。

## 6. P022 三轮性能对照

FP32 dynamic `reduction_modulo`，`rtree_real_block=False`，warmup 10/runs 100；flatten
开/关各三轮 fresh process。表中正值表示开启后更慢。

| 指标 | off 中位 | on 中位 | on 相对 off |
|---|---:|---:|---:|
| device P50 | `0.238570 ms` | `0.591570 ms` | `+147.96%` |
| device P99 | `0.263420 ms` | `0.635720 ms` | `+141.33%` |
| host P50 | `0.290860 ms` | `1.161635 ms` | `+299.38%` |
| host P99 | `0.535600 ms` | `4.967670 ms` | `+827.50%` |
| 编译首跑 | `22.120 s` | `23.221 s` | `+4.97%` |
| peak | `7,685,120 B` | `7,685,120 B` | `0%` |

三轮 device P50 均同向大幅回退，不能启用默认值。P022 仍有必要合入，因为 opt-in
正确性缺陷属于静默错算；性能策略保持保守关闭。

## 7. 构建、静态检查与证据

- P021 wheel SHA256：
  `f0f953026aad5aa0441b3331a350238e44c400d2ef06a329de191d43bd062bfa`
- P021 source/installed SHA256：
  `c67e76fe8c6e570a01c2f189df95a2a4b51d070fbd7020f3d707b91327daa35d`
- P022 wheel SHA256：
  `1903bcbb8dd98dfd8822929063b45bc79aa4e7f0aa7c9d55eb2a46ef18fb45ef`
- P022 source/installed SHA256：
  `aad7bdf90056ba9b7f5a6616fef7f553fa34af725e4bcf263949d4928f728c0a`
- P022 修改文件 FLAKE8、NEWLINE、SPACES、TABS：0 问题；未执行自主修复。
- Python 编译、shell 语法和 `git diff --check`：通过。

证据入口：

- 汇总：`results/t064_rtree_real_block_20260828/t064_summary.json`
- 运行探针：`t063_range_tree_runtime_probe.py`
- 性能脚本：`run_t064_p022_performance.sh`
- 聚合脚本：`aggregate_t064_reduction_tiling_results.py`
- P021 文档：`issues/P021_zero_x_rtree_guard/`
- P022 文档：`issues/P022_flatten_rtree_bound/`

## 8. 下一步

继续 T-065 / TE-CG-008。T064 的两项产品修改继续留在本地独立 worktree，不与前序 pass
混推；待剩余 pass 全部完成本地测试与优化后，再按用户约定统一决定提交和推送顺序。
