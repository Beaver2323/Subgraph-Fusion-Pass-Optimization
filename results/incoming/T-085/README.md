# T-085 GPU handoff 接收目录

> 更新时间：2026-09-09 00:35 CST（UTC+08:00）

本目录只接收 T-085 真实 GPU reference 的文本 handoff，不代表结果已通过。

优先上传单文件：

```text
results/incoming/T-085/text-handoff.json
```

若 GPU 一键脚本提示 `handoff_upload_mode=split`，则上传同一轮生成的 manifest 和全部分片：

```text
results/incoming/T-085/
├── manifest.json
├── part-0001.json
└── ...
```

不要混用不同 run ID 的分片，不要只上传 manifest，也不要覆盖 `results/current/`。T-085 的 overlap
社区例要求两张可见 GPU 和真实 2-rank NCCL；单卡、skip/xfail、fake process group 或 CPU-only
结果均不能冻结分母。
