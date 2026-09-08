# 低精度三臂证据

> 整理时间：2026-09-08 07:51:59 CST（UTC+08:00）

本目录保存 T-078 addcdiv 在 NPU `triton_experimental` 后端的 FP16/BF16 三臂原始证据。
`off` 是产品/目标关闭路径，`decomposed` 直接编译除乘加，`re-fused` 启用候选 addcdiv FMA。

每个 arm 均包含结构化结果、FX 前后图、IR 前后文本和 `output_code.py`。runner 捕获的
`generated_code.py` 与 `output_code.py` 逐字节相同，仓库只保留后者。汇总见
`three_arm_summary.json`；解释、调用栈和结论见 `../../低精度三臂与BF16修复报告.md`。

这些文件来自六个 fresh process，输入由同一个 CPU seed 生成。FP16 三臂 compiled 输出完全相同但
共同偏离 eager；BF16 三臂和 eager 完全相同。
