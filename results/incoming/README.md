# GPU handoff 接收目录

> 更新时间：2026-09-09 18:05:53 CST（UTC+08:00）

此目录用于接收 GPU 回传文本，尚不是已验收的正式结果。T-076～T-080 均已有有效文件，目录由
Git中的实际handoff保留。T-081～T-083回传已于2026-09-08完成复核：11/11 cases、24/24
variants有效；机器复核记录见`results/current/T-081`～`T-083/gpu_reference_review.json`。
T-084～T-086 的接收目录已建立，但尚无设备结果；其中 README 仅是任务合同和上传说明，不是 PASS。

将 GPU 的 `latest-text-handoff.json` 完整内容复制到控制节点对应路径：

```text
results/incoming/
├── T-076/text-handoff.json
├── T-077/text-handoff.json
├── T-078/text-handoff.json
├── T-079/text-handoff.json
└── T-080/text-handoff.json
```

统一 runner 默认生成单行 JSON；超过 96 KiB 时自动生成并推荐上传分片。T-076、T-077、T-079、
T-080 的 1.3 review 已接收并通过校验。T-076 使用 `manifest.json` 与五个 part，旧的截断
`text-handoff.json` 已移除；缺少分片或文件名不符合 manifest 的文件不参与验收。

若 GitHub 网页不接受单个长文件，将 manifest 与所有 part 放在同一目录。仓库当前使用任务目录根部：

```text
results/incoming/T-079/
├── manifest.json
├── part-0001.json
└── ...
```

manifest 和全部分片共同构成一份 handoff；缺一片、错序、复制截断或混入其他轮次都会校验失败。
控制节点直接读取 manifest，无需人工拼接。当前默认成品约 66 KiB/片；若网页仍拒绝，可在 GPU
导出时指定 `--split-part-bytes 16384`，得到约 23 KiB/片。

上面的 JSON 是待保存的文件名，不是仓库提供的空 JSON 模板。例如 T-078 保存为：

```text
/home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/results/incoming/T-078/text-handoff.json
```

通过 GitHub 网页提交新一轮同名文件时，旧内容由 Git 历史保留；需要在同一 checkout 并存多轮时，
可另存带 run ID 的文件。保存后告诉 Agent 任务号、文件路径和 commit；Agent 拉取、校验和复核后，
才按工作流更新正式结果。

不要覆盖 `results/current/` 或 `results/audits/`。网页回传的 handoff 可以提交到对应 incoming 任务
目录；日志、完整 archive、二进制和本地临时文件仍不进入仓库。
目录存在、JSON 能解析均不代表 reference 已通过。默认 1.3 review 可恢复摘要、FX、case 结果与
benchmark，评审范围外文件保留哈希；1.2 archive 用于完整 UTF-8 文本排障。旧格式 1.0 只有摘要
与哈希，不能恢复正文。

完整运行和等卡见[GPU 一键运行说明](../../docs/GPU_TASK_RUNNER.md)；格式区别、文本复制、整包校验、
安全恢复和 FX 查看命令见[GPU 原文 handoff 指南](../../docs/GPU_TEXT_HANDOFF.md)。
