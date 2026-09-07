# T-083 GPU回传入口

> 更新时间：2026-09-08 00:43 CST（UTC+08:00）

将GPU `/data/z50063656/tmp/t083-reference-results/latest-text-handoff.json` 的内容保存为此目录的 `text-handoff.json`。
若自动分片，则将本轮 `manifest.json` 与所有 `part-*.json` 放在本目录，勿混入旧轮次。
原生GPU功能是world_size=1，仅一张卡；真实2rank性能还需独立功能证据。
本轮`text-handoff.json`已回传并通过复核；证据仍限定world_size=1。完整方法见
[统一接收说明](../README.md)，冻结记录见`../../current/T-083/gpu_reference_review.json`。
