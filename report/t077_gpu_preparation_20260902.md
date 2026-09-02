# T-077 第二波 GPU/reference 准备报告

> 更新时间：2026-09-02 21:33 CST（UTC+08:00）
> 状态：`SUPERSEDED_BY_VALID_REFERENCE`（准备阶段记录保留）
> PyTorch 基线：`release/2.14@8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`

## 1. 结论

T-077 已从 T-074 provisional inventory 中人工收敛出第二波 5 个 optimization contracts，形成
独立 manifest 与 reference plan。范围不是按 registration 行数机械抽取，而是以 community tests
可直接验证的上游行为合同分组。

静态校验结果：

```text
reference_plan_validation=OK acceptance_units=5 cases=11 community_tests=11 variants=17 executed_variants=17 non_executed_variants=0
torch_imported=0 gpu_executed=0
```

后续 GPU 运行已完成：11/11 direct cases 与 17/17 variants 有效，5 个单元已冻结；没有创建 GPU
adapter。正式动态结果见 `t077_gpu_reference_20260902.md`，本报告保留运行前设计与风险边界。

## 2. 选择与收束理由

本轮选择 Gumbel-max、B2B GEMM 和 memory-bound matmul decomposition 三组相邻 Inductor 优化：

1. 均有明确源码 handler/guard 与 direct-trigger community tests；
2. 均能在既有 A100 reference 环境中原生执行；
3. 同时覆盖 pre-grad、post-grad，正例、负例、shape threshold、dynamic 与 mixed precision；
4. 避开 T-074 中仍为 `no-test-found`、indirect、distributed 或 CPU-only 的不确定单元。

人工复核新增了 T-074 自动索引遗漏的两个 B2B 负例：bad-pattern/good-shape 与
good-pattern/bad-shape。另一方面，decompose 的 CPU 专用方法被明确排除，不与 CUDA/GPU 合同混计。

## 3. 参数化入口处理

`instantiate_parametrized_tests` 会移除基础方法并生成长名称。runner 新增 `direct_args`，仍以源码中的
基础 community method 作为 mapping 锚点，但执行时传入冻结 commit 真实生成的 unittest 名称：

- BMM：3 个 CUDA 参数化名称；
- MM fp32：3 组 shape × `has_bias=True/False`，共 6 个名称；
- MM mixed precision：同样 6 个名称；
- 所有 `*_cpu` 方法均不执行。

这不是 adapter，也未复制测试体；执行的仍是冻结 PyTorch 源码中的原生 unittest。

## 4. 关键风险

- Gumbel 测例包含 1,000,000 行统计采样，不得缩小 M 或放宽 10% 相对容差；
- B2B 正例依赖 fp16 与 fp32 accumulation 参考的既有容差；不能仅以 correctness 推断 target hit；
- decompose-addmm 使用 `M=19494144` 和 dynamic compile，资源与耗时最高，保留 7200 秒 case timeout；
- NPU 静态源码的 `check_device` 当前只识别 CUDA/XPU/CPU，预期后续会出现 NPU 产品差异；这不影响
  GPU reference，且不得在 GPU 阶段提前改写；
- 任一 direct case 缺少 FX before/after 时，即便 unittest 通过也不是 valid reference。

## 5. 交付物

| 文件 | 用途 |
| --- | --- |
| `upstream/t077_manifest.yaml` | 第二波 5 单元人工映射与 17 variants |
| `upstream/t077_reference_plan.yaml` | 11 个 direct cases、精确参数化名称和 timeout |
| `scripts/run_t077_reference_all.sh` | T-077 固定入口，复用通用 runner |
| `docs/T077_REFERENCE_RUNNER_GPU.md` | GPU 环境、整批/单 case、文本回传说明 |
| `runners/reference_runner.py` | 支持 task_id、多 manifest/plan 与 direct_args 的通用 runner |

## 6. GPU 返回后的门禁

1. 核对 environment fingerprint、PyTorch source/runtime commit 与 clean tree；
2. 核对 11 个 case 的执行数、skip/timeout、FX signatures、reference result 与 inventory SHA256；
3. 只有完整 suite valid 才将 5 个单元从 `pending-reference` 更新为 `yes-frozen`；
4. 随后才制定 NPU direct-first 执行与 case-specific adapter 决策。
