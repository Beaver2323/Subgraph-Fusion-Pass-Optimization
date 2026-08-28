# T-062 experimental generate-list fallback 审计（2026-08-28）

## 当前状态

`fallback-policy-verified-global-expansion-rejected`。现有 `GENERATE_LIST`/fallback 路由合同、
安装态 fallback 和 audit-only generate 路径均已验证；没有修改 PyTorch、torch_npu 或 Triton
产品源码，也没有构建新 wheel。`ceil` 在混合图中有明确融合收益，但全局加入 allowlist 会使
standalone 场景的设备时间、峰值内存和首编成本回退，因此本轮不扩大 allowlist。

## 源码合同与安装边界

- `triton_experimental/lowering.py::_register_npu_inductor_fallbacks()` 把 `GENERATE_LIST` packet 及
  overload 展开；不在 generate/decomposition/keep 集合中的 lowering 改为 ATen fallback；
- `KEEP_UPSTREAM_LOWERING` 保留 `control_deps`、四类 assert packet 和 `index_put/index_put_`，
  避免 generic fallback 覆盖专用 Subgraph/assert/index_put lowering；
- P-019 独立 venv 用于本轮安装态验证。current source、P-019 worktree 和安装态
  `lowering_override_list.py` SHA256 均为
  `cc750008ad3818ff94578f2e9aaf465949d90ed07a6f3cdd8d7a9f6cd4d29944`，没有 source overlay；
- 列表共有 80 个 entry、78 个唯一 entry，展开后 450 个 generate key；keep 为 7 个 entry、
  展开后 17 个 key。合成 registry probe 只把排除且非 decomposition 的 `aten.ceil.default`
  送入 fallback，六项合同全部通过；既有 control-deps 与 Bernoulli 定向 UT 2/2 通过。

## 功能与结构矩阵

所有 worker 从 `/home/z50063656/tmp` 启动，fresh process、独立 cache，显式使用
`options={"npu_backend": "triton_experimental"}`，并检查实际 backend、生成代码和逐元素数值。

| 用例 | 路由/结构 | 结果 |
|---|---|---|
| ceil FP32 baseline | ATen fallback，0 Triton | exact |
| ceil FP32 audit generate | 1 Triton | exact |
| floor/trunc/round FP32 audit generate | 各 1 Triton | 3/3 exact |
| isnan/isinf FP32 special values | 各 1 Triton | 2/2 exact，mismatch 0 |
| ceil FP16/BF16 | 各 1 Triton | 2/2 exact |
| ceil FP32 dynamic 65536→98304 | 1 dynamic Triton | 两 shape exact |
| sin→ceil→cos baseline | 2 Triton + 1 ATen ceil | exact |
| sin→ceil→cos audit generate | 融合为 1 Triton | exact |

最终有效矩阵 12/12。首轮沙箱内四 worker 在张量搬运前统一报 `aclInit 507008`，属于设备访问
权限边界；原目录和日志保留，不计产品失败。registry 初版又假设 P-019 wheel 已含 P-020 后新增
的 `_register_int64_fallbacks`，在设备启动前失败；兼容 pre-P020/P020 后 `registry_retry1` 通过。

## 三轮交错性能

同一 NPU7，FP32 65536 elements，warmup 10、runs 100，顺序交错，保留全部样本。

| cohort | Event mean | Event P50 | Event P99 | host mean | host P50 | host P99 |
|---|---:|---:|---:|---:|---:|---:|
| standalone ceil：generate 相对 fallback 中位改善 | -7.11% | -7.39% | +0.16% | +3.65% | +2.87% | +10.61% |
| mixed：generate 相对 fallback 中位改善 | +32.76% | +33.53% | +28.15% | +28.05% | +32.10% | +0.28% |

standalone Event mean/P50 仅 1/3 轮改善，P99 2/3；generate peak 从 `787,968 B` 增到
`1,313,280 B`（+`525,312 B`），forced-fresh 首编+首跑中位数从 `2.716 s` 增到 `18.341 s`
（+575.40%）。它不满足全局放行门槛。

mixed 的 Event mean/P50/P99 和 host mean/P50 均 3/3 改善；host P99 2/3 改善，其中 round2
回退 4.44%。kernel 数从 2 Triton + 1 ATen 降为 1 Triton，peak 从 `2,101,248 B` 降到
`1,313,280 B`（-`787,968 B`），首编+首跑从 `38.509 s` 降到 `19.032 s`（改善 50.58%）。

## 结论与下一步

当前 allowlist 是全局策略，无法只在有前后邻接融合收益时放行 `ceil`。因此不以 mixed 收益掩盖
standalone 回退，不新增 `ceil/floor/trunc/round/isnan/isinf`，也不为它们手写 Triton kernel。
TE-LOW-001 关闭为 `fallback-policy-verified-global-expansion-rejected`；混合图可另立上下文感知
融合提案，但不在本轮扩大范围。正式无 shim launcher 仍是独立环境边界。

下一项为 T-063 / TE-CG-005：range-tree split/collapse/scalar odometer。
