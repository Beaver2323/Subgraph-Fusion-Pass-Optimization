# T-078 GPU reference、NPU 修复与性能处置闭环

> 更新时间：2026-09-06 09:50 CST（UTC+08:00）

> 2026-09-08 06:07 CST 覆盖修订：本文“闭环”仅适用于当时冻结的 20 个 variants，addcdiv
> 社区 case 实际只有 FP32。上游 guard 允许 FP16/BF16，现已新增 2 个 pending dtype variants；
> 在 GPU reference 与 NPU 三臂精度归因完成前，不得把本文 FP32 功能/性能结论外推至低精度。
> 详见 `docs/T078_ADDCDIV_LOWP_COVERAGE.md`。
> PyTorch：`8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b`
> torch_npu 基线：`83cc452480c3546fd5cccf853bfe3a360ce9dbfc`；本文涉及的产品改动仍在共享工作树，尚未由本报告宣称已合入。
> GPU reference 后端：`inductor-default`；NPU 动态验证与性能后端：`triton_experimental`。

## 1. 闭环结论

T-078 的 4 个 post-grad acceptance units 已完成 reference → NPU → compare → repair →
performance → final product gate：

- GPU：12/12 原生 community cases passed、reference valid，20/20 variants 均有处置；没有 GPU adapter。
- NPU：4/4 单元形成正式 `npu_result.json` 与 `comparison_result.json`。
- 修复：addcdiv FP32 pattern/lowering 与 partial min2 Triton Ascend 兼容修复通过同合同验证。
- 性能：4 个候选单元均完成三轮交错 OFF/ON；最终保留 2 个、选择性保留 1 个、显式关闭 1 个。
- 产品门禁：addmm unfuse 全局关闭；baddbmm unfuse 仅允许默认 `alpha=beta=1`。

因此 T-078 的 4 个单元均冻结进入 denominator，项目累计已冻结/闭环 14 个单元；T-079/T-080
仍等待 GPU reference。

## 2. GPU reference 复核

GPU 文本 handoff：`results/incoming/T-078/text-handoff.json`。

| 字段 | 值 |
| --- | --- |
| run id | `reference-20260905T214703+0800-rarkggw1` |
| 设备 | NVIDIA A100-SXM4-80GB |
| 结果 | 12 passed / 0 failed / 12 reference valid |
| variants | 20/20 已执行并有效 |
| 环境指纹 | `ad8df1e2357e5e0351014dc7c3dd578f8ae640b4726d1138bc991cc30da10dc3` |
| payload SHA256 | `03300f564d4e16435a17eb0b641dccdf55dccded2be9064208951ef1208f52ed` |

该 1.0 紧凑 handoff 包含每 case 的状态、FX before/after 捕获标记和签名、结果/inventory 哈希，
足以冻结原生断言与结构签名；它不含 FX 正文和完整日志，因此本文不声称做了逐行 GPU/NPU FX
正文比较。后续若需逐节点教学或排查，应从 GPU 端额外文本导出相应 FX 文件。

## 3. NPU 功能与修复

### 3.1 addcdiv FMA

原始 NPU `triton_experimental` 没有对应的 FP32 re-fusion registration，且 fallback 扫描覆盖了
可用的 addcdiv lowering。修复后：

- `value=2` 命中 `addcdiv_fma_fused=1`，生成 `tl.fma` 与 `triton.language.div_rn`；
- compiled/eager bitwise equal；
- `value=1` 保持 counter=0，因为 decomposition 已消去乘一，只作为 bitwise 邻接；
- FP16/BF16 不扩围。

正式结果：
`results/current/AU-post-grad-fuse-addcdiv-to-fma/{npu_result,comparison_result}.json`。

### 3.2 partial reduction reuse

三个大 shape 的 pattern 都能命中；`amin→min` 在旧 helper 上出现 Triton Ascend 编译失败。
将 min2 表达为 `tl.reduce` + `tl.minimum(..., PropagateNan.ALL)` 后：

- 三个正例均 target hit=1、两个输出与 eager 一致；
- NaN 邻接语义通过；
- 4×8 与 dynamic 两个 guard 均 target hit=0。

正式结果：`results/current/AU-post-grad-reuse-partial/{npu_result,comparison_result}.json`。

### 3.3 addmm / baddbmm unfuse

上游 `is_gpu` guard 不是 NPU 缺能力 blocker：torch_npu 的 `patch_is_gpu` 已包含 NPU。仅需把社区
测试设备入口切到 NPU 并固定 backend。门禁前功能验证表明两类 unfuse 都能正确改写。

性能后，最终产品策略为：

```python
# torch_npu/_inductor/triton_experimental/config.py
disable_unfuse_bias_addmm: bool = True
disable_unfuse_baddbmm_non_default_scalars: bool = True
```

门禁只包裹相应 registered handler，不关闭全局 pattern matcher。最终动态验证结果：

| Case | target hit | 必须存在 | 必须不存在 | 正确性 |
| --- | ---: | --- | --- | --- |
| addmm 默认标量 | 0 | addmm | mm | max error `2.384185791015625e-7` |
| baddbmm 默认标量 | 1 | bmm | baddbmm | max error `2.384185791015625e-7` |
| baddbmm alpha=.8、beta=.2 | 0 | baddbmm | bmm | max error `0` |

正式门禁文件：`results/current/T-078/product_gate_verification.json`，仓库副本 SHA256：
`e5f719948793d0a1db909de283fd03caf81d57459f5121f557dbf91ef1dbf63d`。

## 4. 性能方法与结果

社区没有这四个 pass 的专属 benchmark。PyTorch 通用 addmm/bmm operator benchmark 只含裸算子，
没有目标 pointwise consumer，无法触发 unfuse；所以 performance worker 原样复用 community
functional case 的图、shape、dtype 与 scalar，仅添加目标级 OFF/ON、host/Event、内存和结构门禁。

每个 arm 为 fresh process，顺序固定为
`OFF1 → ON1 → ON2 → OFF2 → OFF3 → ON3`，warmup=10、runs=100。

| 单元/workload | host p50/p99 改善 | Event p50/p99 改善 | 判定 | 产品动作 |
| --- | ---: | ---: | --- | --- |
| addcdiv value=2 | 0.70% / 4.54% | 1.82% / 2.62% | `PERF_NEUTRAL` | 保留修复 |
| partial 2048² amax | 47.09% / 44.12% | 56.65% / 54.21% | `PERF_IMPROVED` | 保留 |
| partial 1024² amin | 1.64% / 8.80% | 2.71% / 27.85% | `PERF_NEUTRAL` | 保留、逐 shape 监控 |
| partial 4096×512 amax | 23.64% / 32.21% | 33.23% / 37.69% | `PERF_IMPROVED` | 保留 |
| addmm + GELU | -20.81% / -138.27% | -14.81% / -75.20% | `PERF_REGRESSED` | 全局关闭 unfuse |
| baddbmm 默认标量 + GELU | 5.30% / 46.65% | 12.24% / 71.39% | `PERF_IMPROVED` | 保持启用 |
| baddbmm alpha=.8、beta=.2 | 2.87% / 64.99% | -5.29% / 9.57% | `PERF_REGRESSED` | 非默认标量关闭 |

完整正式数据：`results/current/T-078/performance_summary.json`。原始运行根目录为
`/home/z50063656/tmp/t078-performance-results/performance-20260906T083844+0800`，任务级原始汇总
SHA256 为 `1c4759582da3a19b382504606932e14002117334c609a544d4b136627f142b17`。

## 5. 证据与后续边界

- 人工导读：`docs/T078_FUNCTION_PERFORMANCE_GUIDE.md`。
- 正式逐 variant 对照：四个 `results/current/AU-*/comparison_result.json`。
- 一键性能入口对 T-078 现只校验并展示正式结果，不再重跑显式关闭的 ON 分支。
- 修复/门禁的产品文件在共享 torch_npu 工作树中，与其他未提交改动共存；提交产品仓前必须单独
  做 diff 归属审查，不能把整个脏工作树打包进 T-078。
- 下一主线任务是 T-079 GPU reference；T-078 只保留回归与 drift 检查，不再作为开放开发任务。
