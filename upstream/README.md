# Upstream Contract 与 Acceptance Unit 数据

> 更新时间：2026-09-01 18:00 CST（UTC+08:00）
> 状态：T-075 首批 5 个单元完成静态人工复核；T-076 direct reference plan 已就绪，等待 GPU 执行。

本目录保存 tracker 的活动数据入口：

- `manifest.schema.json`：acceptance-unit manifest 的结构合同；
- `manifest.yaml`：首批人工审核后的 acceptance units；
- `pass_map.yaml`：registration candidate、生成式 registration、community test 与 unit 的
  多对多证据；
- `reference_plan.schema.json`：GPU/reference 执行计划合同；
- `reference_plan.yaml`：13 个原生 community cases、variant 覆盖和非动态处置；
- `../scripts/validate_tracker_data.py`：零第三方依赖的一致性检查。

`manifest.yaml` 和 `pass_map.yaml` 使用 JSON-compatible YAML：它们是合法 YAML，同时可以用
Python 标准库 `json` 解析，避免 GPU 机器额外安装 PyYAML。

## 计数边界

- T-074 v1 的 207 行 candidate CSV 继续作为 inventory 输入；
- 本目录当前只有 5 个已静态人工复核的 acceptance units；
- 这 5 个单元仍为 `pending-reference`，不能计入正式闭环；
- T-076 已为 20 个 variants 中 14 个建立原生动态 case 映射；3 个 registration-only 和 3 个
  NPU-only gate 已显式列为 reference 非动态项；
- 当前 adapter/extracted case 数为 0，只有 GPU direct artifacts 证明阻塞后才允许新增；
- 188/158 的 T-074 heuristic 统计没有被覆盖或重写；
- 后续单元只能在 community contract 人工审核后增量进入 manifest。

## 验证

NPU 控制节点从 `/home/z50063656/tmp` 执行；GPU 机器从
`/data/z50063656/tmp` 执行并替换为 `/data/z50063656/src/pytorch`：

```bash
python /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/validate_tracker_data.py \
  --pytorch-root /home/z50063656/Pass/src/pytorch
```

该检查验证 JSON-compatible YAML、必填字段、唯一 ID、candidate 映射、community test 文件和
测试方法是否存在，并校验 13 个 reference cases 对 20 个 variants 的完整处置；它不导入
`torch`，也不运行 GPU/NPU。GPU 执行入口与人工交接说明见
[`REFERENCE_RUNNER_GPU.md`](../docs/REFERENCE_RUNNER_GPU.md)。
