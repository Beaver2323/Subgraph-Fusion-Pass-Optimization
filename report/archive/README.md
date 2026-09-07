# 历史报告归档

> 更新时间：2026-09-07 10:20 CST（UTC+08:00）

本目录保存当前 community-native compatibility tracker 之前的实验报告和静态清单。归档只改变路径，
不改写历史环境、backend、结果或结论。

## 目录

| 目录 | 时间范围 | 内容 | 当前证据地位 |
| --- | --- | --- | --- |
| `legacy-20260820-0828/` | 2026-08-20～2026-08-28 | P0、mm/pad、torch_npu custom pass、DVM/MLIR、attention、早期 `triton_experimental` feature family | 历史/辅助证据，不能直接计入当前 acceptance-unit verdict |

当前主线报告仍位于 `report/` 一级目录，从 [报告索引](../README.md) 进入；当前机器可读真值为
[acceptance-unit 矩阵](../current_acceptance_unit_matrix.md) 和 [`results/current/`](../../results/current/)。

引用归档结果时必须同时核对 source revision、输入合同、backend、gate 生命周期与测量方法。任何一项
不一致，都只能标为 non-counting historical evidence。
