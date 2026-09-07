# T-083 GPU回传入口

> 更新时间：2026-09-07 22:36 CST（UTC+08:00）

将GPU `/data/z50063656/tmp/t083-reference-results/latest-text-handoff.json` 的内容保存为此目录的 `text-handoff.json`。
若自动分片，则将本轮 `manifest.json` 与所有 `part-*.json` 放在本目录，勿混入旧轮次。
原生GPU功能是world_size=1，仅一张卡；真实2rank性能还需独立功能证据。
当前仅有此说明，尚无GPU执行结果。完整方法见 [统一接收说明](../README.md)。
