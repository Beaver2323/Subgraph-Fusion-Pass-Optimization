# BF16 性能证据

> 整理时间：2026-09-08 07:51:59 CST（UTC+08:00）

本目录保存 T-078 addcdiv BF16 的三轮 fresh-process OFF/ON 性能汇总，以及首轮两臂的
`result.json`、生成代码、FX 和 IR。重复的 debug/cache 副本已经按字节一致性核对后去重。

`performance_summary.json` 是正式机器可读结论：目标 OFF 三轮 hit=0、ON 三轮 hit=1，功能均
位级一致，最终为 `PERF_NEUTRAL`。测试方法和指标解释见 `../../低精度三臂与BF16修复报告.md`。
