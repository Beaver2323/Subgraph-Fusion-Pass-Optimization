# Tracker 验收加固与后续准备边界

> 更新时间：2026-09-06 02:21 CST（UTC+08:00）
> 范围：tracker 本地代码、计划、文档；不修改 PyTorch/torch_npu 产品代码、环境或已有实测结果。

## 已实现

| 缺口 | 处理 | 验收方式 |
| --- | --- | --- |
| 部分 skip / expected failure 被算通过 | 任一 skip/expected failure 均拒绝 valid；direct 测试数须与选定方法数一致；缺少 OK 摘要不得通过 | 模拟 unittest 输出与关联 variants 回归 |
| 错后端结果仍可被纳入 | NPU schema、NPU/comparison 校验均要求 triton_experimental | default/DVM/MLIR 即使重新计算正确指纹也被拒绝 |
| 数值失败无法规范落盘 | runtime failed 可作为合法证据；必须保持未修复 NPU_REGRESSION；禁止计为性能测量、收益或正式闭环 | 原始失败/伪造成功/伪造 verified 等回归 |
| 文本导出失败仍读到旧成功 | handoff 放在本轮目录；文本入口经 latest 寻址；失败时写本轮 export-failed 状态；原子切换 latest | 连续成功→失败、无 artifacts、错误 run_id、普通目录保护 |
| 性能准备校验过宽 | 正整数迭代数、逐单元来源归属、非空 nodeids、去重 workload/unit、带时区时间戳 | 负数/布尔迭代数、跨单元来源、重复记录等回归 |
| 功能 reference 继承性能环境开关 | 子进程去掉 DO_PERF_TEST 与 USE_LARGE_INPUT | mock 子进程环境回归；不访问设备 |

新测试位于 `tests/test_tracker_contracts.py`。已知 codegen-only 社区测试不再自动声称数值正确；
它们可以满足自身 codegen reference 合同，但性能前需要额外数值门禁。
当前 T-076/T-077 的 10 份结果继续通过；未修复回归不计入 formally_closed，MM 已验证候选的产品合入状态仍单列。

## 性能准备的真实状态

T-078～T-080 目前只有性能方案，**worker 尚未实现**，并非仅等 GPU 回传自动解锁。
本轮修正入口、计划、README/TODO/工作流的表述；这不是完成 11 个单元的性能执行准备。
后续需落实每个目标的 OFF/ON 隔离、数值/目标命中、源码/环境绑定、计时/内存采集与测试。
有 generic device guard 的单元还须等待 GPU 证据并评审最小 NPU 适配，不能提前绕过。

方案修正包括：partial/full 归约原函数都读取 x；cat-slice-cat 明确 end=19；cat→split 保留平方 consumer；
两个 prepare-softmax 社区 workload 区分 log/no-log；CrossEntropy 明确 BF16 参数/输入、int64 label、weight.grad；
新增 shape 网格单列 tracker sensitivity，不能冒称社区 benchmark。

## 后续排期与范围

`docs/TASK_BACKLOG.md` 和 `upstream/task_backlog.json` 对应 T-081～T-113 的 33 个草案批次，
暂列剩余 137 个 provisional eligible 单元；另有 30 条非计数结构记录待审。
通过 constructor mover 的旧 cuda 名称到实际 gpu 名称映射避免重复计数，保留原始 T-074 文件。
每批最多 5 个暂列单元，须人工确认合同后才能创建可执行 manifest/reference/性能 worker。
独立 lowering/template 注册的全量 inventory 仍待补，不在该 33 批数量内。

## 零设备复核命令

本轮结果：28 项 unittest 回归通过；10 个 comparison 单元、37 个 variants 的现有结果通过；
T-076～T-080 共 53 个 reference cases 的静态入口校验通过（不代表新动态执行）；
三批性能准备和 backlog 一致性校验通过。修改文件的 Python AST、shell 语法、JSON 解析、
55 处本地 Markdown 链接与 `git diff --check` 均通过。

以下命令从 `/home/z50063656/tmp` 执行；仅 tracker 标准库测试和静态校验，不导入 torch：

```bash
cd /home/z50063656/tmp
TRACKER_ROOT=/home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s "${TRACKER_ROOT}/tests" -v
python "${TRACKER_ROOT}/scripts/validate_comparison_data.py"
python "${TRACKER_ROOT}/scripts/validate_prepared_tasks.py"
python "${TRACKER_ROOT}/scripts/build_task_backlog.py" --check
bash "${TRACKER_ROOT}/scripts/run_t078_reference_all.sh" --pytorch-root /home/z50063656/Pass/src/pytorch --validate-only
bash "${TRACKER_ROOT}/scripts/run_t079_reference_all.sh" --pytorch-root /home/z50063656/Pass/src/pytorch --validate-only
bash "${TRACKER_ROOT}/scripts/run_t080_reference_all.sh" --pytorch-root /home/z50063656/Pass/src/pytorch --validate-only
```

本轮没有新 GPU/NPU 实测、没有产品修复合入，也没有改写历史实测哈希或将其他后端数据改标签。
