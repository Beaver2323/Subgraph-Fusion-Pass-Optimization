# T-102 交付索引：attention pattern 1～5

> 更新时间：2026-09-15 20:15 CST（UTC+08:00）。5/5 原合同、安装态修复、独立OFF/ON和性能处置完成。

## 结论

五个编号原先训练图不能匹配CUDA预生成注册，现在通过活动NPU decomposition重建匹配图，安装态原例均通过；不是仅有隔离候选，也没有修改上游、容差或显式禁用配置。
共同候选五原方法、安装态五原方法、部署前后各六项设备边界通过；原社区每方法均1test、0skip。
3/4原方法没有随机输出/梯度数值oracle，部分对齐继续保留；5号带mask最终是NPU数学展开，不等于融合FA内核。

## 每个pattern的功能、代码与性能

正改善率表示时延下降，负值表示回退；均为half training-forward微图，不含backward、完整冷启动或模型端到端。

| Pattern及讲解 | 性能结论 | host p50 / p99 改善 | Event p50 / p99 改善 |
|---|---|---|---|
| [1](pattern-1_讲解.md) | PERF_IMPROVED | +27.90% / +22.95% | +34.38% / +30.17% |
| [2](pattern-2_讲解.md) | PERF_IMPROVED | +28.30% / +26.62% | +34.63% / +32.68% |
| [3](pattern-3_讲解.md) | PERF_IMPROVED | +41.99% / +45.37% | +47.93% / +47.18% |
| [4](pattern-4_讲解.md) | PERF_IMPROVED | +41.89% / +39.60% | +47.12% / +45.44% |
| [5](pattern-5_讲解.md) | PERF_REGRESSED | -40.27% / -29.91% | -53.11% / -42.08% |

本批5项均实测，没有默认关闭免测或特许归因受限。各臂10次预热、host及Event各100样本，六个新进程独占串行；实际OFF无attention、ON本编号精确改图，且数值通过才计时。收益不用于擅改默认配置。

## 证据阅读顺序

1. 上表逐pattern讲解：源码位置、search/replacement、GPU/NPU、实际生成代码、输入与时延。
2. [GPU复核](gpu_reference_review.json)、[功能汇总](npu_functional_summary.json)、[性能汇总及原样本引用](performance_summary.json)。
3. [机器可读验签包](attention_bundles/)及[执行过的功能门禁](performance_gates/)。
4. [共同修复报告](../../../issues/REF-sfdp-pattern-1-native/训练注册修复报告.md)、[部署备份/原例/边界](../../../issues/REF-sfdp-pattern-1-native/deployment-20260915-training/)、[独立产品补丁](../../../issues/REF-sfdp-pattern-1-native/sfdp_training_registration.patch)。其余四个issue也有自包含修复说明。
5. 性能入口真实问题：[1号推理OFF邻接接替](../../../issues/REF-sfdp-pattern-1-native/性能合同修订说明.md)、[scale占位适配](../../../issues/REF-sfdp-pattern-2-native/性能输入适配报告.md)。旧失败不覆盖。

`npu_stage_reviews.json`保存修复阶段证明，并非最新最终判定；最终以本页及functional/attention_bundles为准。
产品分支`fix/t102-training-registration-20260915`、本地提交`3a0b8179d8c6db72ae1c86ea02e88ea5229556ef`，未推送产品仓/发起社区PR/运行社区CI或发布wheel。T-102新部署不自动重新认证其他批次旧源码指纹。

## 离线核验

```bash
cd /home/z50063656/tmp
/home/z50063656/envs/Pass/bin/python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/review_attention_completion.py --check-current
```

此命令只读仓库证据和哈希，不导入torch，也不会执行归档生成代码；输出device_execution=false指此次离线检查不使用设备，不否定原件的真机执行。
整体验收和历史再认证边界见[部分交付说明](../../../report/部分交付说明_20260915.md)。
