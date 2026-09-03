# Upstream Contract 与 Acceptance Unit 数据

> 更新时间：2026-09-03 07:30 CST（UTC+08:00）
> 状态：T-076 已完成；T-077 功能/comparison 与性能处置均 5/5、pending=0；MM 修复已验证、尚未合入。

本目录保存 tracker 的活动数据入口：

- `manifest.schema.json`：acceptance-unit manifest 的结构合同；
- `manifest.yaml`：首批人工审核后的 acceptance units；
- `pass_map.yaml`：registration candidate、生成式 registration、community test 与 unit 的
  多对多证据；
- `reference_plan.schema.json`：GPU/reference 执行计划合同；
- `reference_plan.yaml`：13 个原生 community cases、variant 覆盖和非动态处置；
- `t077_manifest.yaml`：第二波 5 个已冻结并完成 NPU comparison 的单元与 17 个 variants；
- `t077_reference_plan.yaml`：T-077 的 11 个 direct cases 与精确参数化入口；
- `t076_performance_plan.yaml`、`t077_performance_plan.yaml`：同 backend 性能实测、显式关闭免测、
  capability 评估与候选拒绝依据；
- `../scripts/validate_tracker_data.py`：零第三方依赖的一致性检查。
- `../schemas/npu_result.schema.json`、`../schemas/comparison_result.schema.json`：统一
  NPU 执行与 GPU/NPU comparison 合同；
- `../scripts/validate_comparison_data.py`：统一结果与 manifest/hash 的零 torch 导入检查；
- `../results/current/`：当前正式 NPU/comparison 记录。
- `../report/t076_gpu_reference_20260901.md`：GPU 环境、13 个 direct 结果、FX signature 和证据哈希。
- `../report/t077_gpu_preparation_20260902.md`：第二波选取、人工复核、参数化入口与风险说明。

`manifest.yaml` 和 `pass_map.yaml` 使用 JSON-compatible YAML：它们是合法 YAML，同时可以用
Python 标准库 `json` 解析，避免 GPU 机器额外安装 PyYAML。

## 计数边界

- T-074 v1 的 207 行 candidate CSV 继续作为 inventory 输入；
- 本目录当前只有 5 个已静态人工复核的 acceptance units；
- 这 5 个单元均为 `frozen`/`yes-frozen`，构成首版 denominator；当前正式闭环为 5/5；
- T-076 已为 20 个 variants 中 14 个建立原生动态 case 映射；3 个 registration-only 和 3 个
  NPU-only gate 已显式列为 reference 非动态项；
- 当前 GPU adapter/extracted case 数为 0；13 个 direct 均 valid，不再设计 GPU adapter；
- 188/158 的 T-074 heuristic 统计没有被覆盖或重写；
- 后续单元只能在 community contract 人工审核后增量进入 manifest。
- T-077 的 5 个单元已由 11/11 direct GPU cases 冻结，并全部形成正式 NPU/comparison；
  decompose-MM 的 lowering 回归保留在 known issues，候选修复已通过同合同回归。
- 性能处置不改变 denominator：T-076 为 2 measured + 3 exempt-explicitly-disabled；T-077 为
  4 measured + 1 capability-assessed-no-effective-template，pending=0。device guard 未包含 NPU
  没有被直接当成显式 disable，而是完成最小适配与收益评估；
  default/DVM/MLIR 结果不得迁移为 experimental verdict。

## 验证

NPU 控制节点从 `/home/z50063656/tmp` 执行；GPU 机器从
`/data/z50063656/tmp` 执行并替换为 `/data/z50063656/src/pytorch`：

```bash
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_tracker_data.py \
  --pytorch-root /home/z50063656/Pass/src/pytorch
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_comparison_data.py
```

该检查验证 JSON-compatible YAML、必填字段、唯一 ID、candidate 映射、community test 文件和
测试方法是否存在，并校验 13 个 reference cases 对 20 个 variants 的完整处置；它不导入
`torch`，也不运行 GPU/NPU。GPU 执行入口与人工交接说明见
[`REFERENCE_RUNNER_GPU.md`](../docs/REFERENCE_RUNNER_GPU.md)。
T-077 使用独立入口 `scripts/run_t077_reference_all.sh`，人工说明见
[`T077_REFERENCE_RUNNER_GPU.md`](../docs/T077_REFERENCE_RUNNER_GPU.md)。
