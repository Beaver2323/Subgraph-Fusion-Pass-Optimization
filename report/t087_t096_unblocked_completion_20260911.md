# 非补证阻塞工作：功能、性能与产品缺口

> 更新时间：2026-09-11 19:00 CST（UTC+08:00）；后端：triton_experimental。7个单元完成，含两项已部署Pass修复。

## 已完成的五个单元

T-088、T-089、T-090全批闭环；T-087训练单元闭环，设备解析后续已修复完成，见下节。
已依次完成原生阻断留证、保留社区合同的最小设备适配、精确目标改图和数值/梯度检查、生成代码审查、独立 OFF/ON 功能门禁、互斥六臂性能测量。

| T / 单元 | 功能 | 性能结论 | host p50 / p99 改善 | NPU Event p50 / p99 改善 | 学习与证据入口 |
| --- | --- | --- | ---: | ---: | --- |
| T-087 训练局部性重排 | 完成 | PERF_MIXED | -5.99% / 15.20% | -4.75% / 6.30% | [代码与训练合同](../results/current/T-087/reorder-locality_讲解.md) |
| T-088 select-cat | 完成 | PERF_IMPROVED | 10.14% / 11.76% | 11.78% / 11.22% | [完整/不完整切片对照](../results/current/T-088/select-cat-aten_讲解.md) |
| T-088 split-cat | 完成 | PERF_IMPROVED | 5.50% / 6.25% | 7.83% / 6.41% | [正例与 singular 负例](../results/current/T-088/split-cat-aten_讲解.md) |
| T-089 move-view-after-cat | 完成 | PERF_IMPROVED | 7.23% / 4.19% | 6.48% / 9.45% | [实际 kernel 变化](../results/current/T-089/move-view-after-cat_讲解.md) |
| T-090 normalize-cat | 完成 | PERF_NEUTRAL | 3.50% / 1.23% | 4.83% / 2.93% | [规范化与下游优化关系](../results/current/T-090/normalize-cat-aten_讲解.md) |

负数表示回退。三臂各求分位数后取中位数；每臂新进程、10 次预热、100 次采样，顺序 OFF1/ON1/ON2/OFF2/OFF3/ON3。编译耗时单列，不混入稳态。
这不是完整模型性能：测例来自社区功能图及原 shape/dtype，而非社区原生 benchmark。训练计时包含前向与反向，不含 SGD；其他项是目标子图一次调用。
没有根据上述局部收益修改产品默认开关。

## 为什么不只看 FX 或数值 PASS

```python
# 实际证据：issues/REF-move-view-after-cat-aten-native/evidence/
# performance-benchmark-20260911T144735+0800/.../on1/debug/.../output_code.py
# OFF：3 次 NPU cat extern
# ON：1 个 Triton 切片/复制核 + 2 次 NPU cat extern
triton_unk_fused_cat_split_with_sizes_view_0.run(...)
buf1 = torch.ops.aten.cat.default([...], 1)
buf2 = torch.ops.aten.cat.default([...], 1)
```

`torch_npu/_inductor/triton_experimental/lowering.py:223` 将 cat 显式注册为 NPU extern；训练反向 matmul_backward 也保留图内 NPU 原生调用。
报告将这些与 CPU / 图外 fallback 分列，不能把“没有 graph break”说成“全部是 Triton 融合核”，也不能声称 CUDA 和 NPU 的 lowering 完全一样。
完整步骤及原社区调用链仍见 [前一阶段学习报告](t087_t090_npu_progress_20260911.md)。

## 不属于 GPU 补证阻塞的两项产品工作

| 单元 | 安装态真实问题 | 当前处置 | 报告 |
| --- | --- | --- | --- |
| T-087 设备解析 | 缺current_device_idx_expr；第一版候选又被原断言检出DeviceProperties(index=0) | 两文件修复已部署；原例1/1与近邻4/4通过，含两设备codegen；无合法OFF性能免测 | [修复验证/代码/日志](../issues/REF-respecialize-current-device-native/修复验证报告.md) |
| T-096 E8M0 | NPU未注册；普通值不改图，one-ULP/gh178045数学编码错误；旧候选不能安全扩至负值/NaN/subnormal | 三文件修复已部署；原例3/3、功能对照7/7及22值域检查通过；额外域标PARTIAL_ALIGNED，七元素性能PERF_REGRESSED | [修复验证/代码/数学oracle/性能](../issues/REF-e8m0-log2-pattern-native/修复验证报告.md) |

用户授权后仅修改Pass安装态experimental文件，保留逐文件备份；其他后端、冻结PyTorch和系统环境未改。原失败/隔离候选原件不覆盖，新建安装态运行和归档。
源码局部分支提交T-087 `6cff38b16`、T-096 `6408f950c`；未推送/社区合入，不等同发布。

T-096性能只测原社区普通七元素正确向量：host p50/p99从0.470425/0.518290ms升至0.572790/0.669906ms，慢21.76%/29.25%；Event从0.356400/0.379270ms升至0.457340/0.484155ms，慢28.32%/27.65%。错误边界OFF不用于算收益；这是派生子图微基准，不是社区原生benchmark或模型E2E。
NPU图内view/rshift extern与CUDA不同，新增范围保护也进入实际生成核；没有逐算子profiler，不能把回退全部归因于某一算子。默认ON是精确数学编码修复，不是基于性能打开旧显式关闭优化。

## 计数与补证队列

活动表仍为 71 个 ID、70 个独立单元；T-112 是 T-084 别名，不重复计数。
已冻结GPU reference从33增至40；NPU/comparison从33增至40；性能处置从33增至40（含T-087设备解析免测和T-096回退）。
新完成7个单元不代表T-076/T-077严格历史再认证通过；原pending=41 / exempt=3的补证门禁独立保留。

GPU 待办不变：T-091 与 T-102～T-107 共 28 个单元补精确目标归因；T-098/T-100 等首次回传；T-078 等正确 FP16 value=1 邻接 reference。
逐项状态以 [当前矩阵](current_acceptance_unit_matrix.md) 为准。历史 evidence 不覆盖，候选/失败/正式通过分别保留。

## 本轮交付检查

2026-09-11 15:30 启动完整 `scripts/validate_all.py --write-audit`，退出码 0，`tooling_gate=passed`；194 项单元测试通过，5 个新完成单元的归档校验、8 组修复前/候选证据校验及各批 reference plan 校验通过。
这里的门禁是工具与归档一致性检查，不代表重新执行了所有历史 NPU 测例；历史再认证仍为 `pending=41 / exempt=3`。
完整输出见 [本轮门禁日志](../results/audits/audit-20260911T153058+0800-yxzrji2n/tooling_gate.log)。

上述15:30为当时工具门禁，安装态新增证据由后续门禁单独核验。两项修复已有独立源码本地提交，tracker工作区尚未提交/推送。诊断/Inductor/验证流程要求的原件审查和近邻回归已完成，未混用其他后端历史结果。

18:55补跑两项修复共同安装后的T-087原例，仍1/1通过；[联合安装记录](../issues/REF-respecialize-current-device-native/combined_install_review_20260911.json)绑定全部5个产品文件。
18:59完整统一检查退出0，200项单元测试通过；安装态4个原合同、11个近邻/控制臂、6个性能臂及联合回归的证据链离线校验通过，性能从100项逐样本独立复算分位数和改善比例。
[最新完整工具门禁日志](../results/audits/audit-20260911T185903+0800-_awul565/tooling_gate.log)中临时`run.sh`语法错误是检查器单测故意构造的负例，不是项目脚本失败。`tooling_gate=passed`仍不等于历史再认证：后者保持pending41/exempt3。
