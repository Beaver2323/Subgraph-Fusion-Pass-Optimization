# GPU handoff 接收目录

> 更新时间：2026-09-06 06:00 CST（UTC+08:00）

此目录用于接收 GPU 回传文本，尚不是已验收的正式结果。
仓库通过 `.gitkeep` 保留 T-076～T-080 目录，clone/pull 后无需再次创建。

将 GPU 的 `latest-text-handoff.json` 完整内容复制到控制节点对应路径：

```text
results/incoming/
├── T-076/text-handoff.json
├── T-077/text-handoff.json
├── T-078/text-handoff.json
├── T-079/text-handoff.json
└── T-080/text-handoff.json
```

上面的 JSON 是待保存的文件名，不是仓库提供的空 JSON 模板。例如 T-078 保存为：

```text
/home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/results/incoming/T-078/text-handoff.json
```

同名文件已经存在时另存新文件，不覆盖旧证据。保存后告诉 Agent 任务号和文件路径；
Agent 读取复核后，才按工作流更新正式结果。

只将接收说明和目录占位纳入 Git；实际 JSON、日志、归档和临时文件默认忽略。
不需要为了让 Agent 读取而提交这些文件，也不要覆盖 `results/current/` 或 `results/audits/`。
目录存在、JSON 能解析均不代表 reference 已通过；紧凑 handoff 也不能替代原始日志和 FX 正文。

完整运行、等卡、复制与验证命令见[GPU 一键运行说明](../../docs/GPU_TASK_RUNNER.md)。
