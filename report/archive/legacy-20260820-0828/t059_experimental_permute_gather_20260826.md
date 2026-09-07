# T-059 experimental permute-gather 审计（2026-08-26）

## 当前状态

`supported-beneficial-host-tail-memory-environment-monitor`。本轮只在现有 installed wheel 上审计
`triton_experimental` 的 `realize_permute_gather`，不构建/安装 wheel，不修改 PyTorch、
torch_npu 或 Triton 源码。

环境启动后没有设置 `TORCHINDUCTOR_NPU_BACKEND`，因此未显式选择时仍解析为 `default`；本任务
runner 必须在每次 `torch.compile` 中传入
`options={"npu_backend": "triton_experimental"}`，并要求
`_InductorNpuRegistry._loaded_backend == "triton_experimental"`。直接在 torch_npu 配置注入完成前
读取 `torch._inductor.config.npu_backend` 会得到 `AttributeError`，这是一次中性探测失败，不是
experimental backend 不可用。

## 源码背景与风险

- 开关：`torch_npu/_inductor/triton_experimental/config.py:realize_permute_gather`，当前 installed
  wheel 与 current source 均默认开启；
- 实现：`triton_experimental/lowering.py:npu_permute()`；当 permute 把非 1 stride 推到输出内轴、
  内轴长度大于 1，且输入是可实现的非 InputBuffer producer 时，物化一份 contiguous copy 并加入
  `no_fuse_buffer_names`；
- 目标：避免 T5 relative-position bias 的 key 轴 stride-H load 在 consumer reduction 中退化成
  scalar gather；
- 风险：额外 copy/kernel/临时 buffer 对较小 shape 或不受 gather 限制的 consumer 可能回退；
  transpose-for-matmul、stride-1 inner axis、singleton inner axis必须保持不触发。

## 首轮验收合同

1. 每个 ON/OFF worker 都从 `/home/z50063656/tmp` 启动，使用独立 Inductor/Triton cache；
2. 正例使用 T5 同构图：embedding 产生 `[S,S,H]` bias，随后
   `permute(2,0,1).unsqueeze(0)` 与 `[B,H,S,S]` scores 相加并沿 key 轴归约；
3. FP16 static 首先验证 compiled 对 NPU eager 和 CPU reference 的正确性；
4. generated code 必须保存，统计 experimental wrapper marker、Triton kernel 数及中间 buffer，
   并确认实际加载 backend；
5. 通过功能与结构哨兵后，按 ON/OFF 交错 fresh-process 做三轮 paired；每轮 warmup 10、runs 100，
   同时记录 host mean/stdev/P50/P99、NPU event、首次编译和峰值内存；
6. 只有同 backend 的 ON 相对 OFF 有稳定收益，才把默认开启判为 beneficial；中性、回退和失败均
   原样归档。

## 待执行

- [x] 新增 fresh-process runner 并做语法检查；
- [x] FP16 static ON/OFF 功能、backend、generated-code 哨兵；
- [x] 三轮 paired performance；
- [x] BF16/FP32、dynamic 和三类负例覆盖；
- [x] 主线转入 `rsplit_outer`。

## 中间结果：smoke 与首轮代表 shape（2026-08-27）

- 首个 sandbox worker 在 `aclInit` 前因设备不可见失败；显式设备权限后不再复现，属于 sandbox
  权限失败；
- 首个真实 NPU OFF worker在 fresh Triton launcher 因 editable PyTorch include view 缺
  `ATen/ATen.h` 失败；后续按已登记 T-022 流程，仅在 worker 内复用 audit-only wheel-header/
  C++20/CANN 声明垫片。它不计正式无垫片环境通过；
- FP16 smoke `B=1,H=12,S=64`：OFF/ON 都实际加载 `triton_experimental`，对 CPU/NPU eager
  通过 `rtol=atol=0.01`；OFF 为一个 kernel，bias load key 轴地址含 `12*r`，ON 为 permute-copy
  加 reduction 两个 kernel，consumer bias load 改为 stride-1；
- 代表 shape `B=2,H=12,S=256` 的 OFF1/ON1 都得到相同 max diff `0.0625` 和相同 max index 31，
  但沿用 smoke 的 `atol=0.01` 导致两侧共同失败。因为它是 256 项 FP16 reduction 的共同舍入，
  不能归因于 permute-gather，也不能直接删除；两轮 timing 仅作为失败伴随数据，不进入三轮聚合。

下一次 runner 修订必须同时记录 mean absolute error、reference scale、逐元素失败数和 deterministic
output hash；代表 reduction 合同预登记为 `rtol=0.01, atol=0.1`，并要求 ON/OFF output hash 相同，
以证明放宽只覆盖共同 reduction 舍入而没有掩盖 pass 差异。修订后从新目录重跑三轮。

修订后 `perf2_off_round1/perf2_on_round1` 都以 0 个失败元素通过上述 CPU/NPU eager 合同，
max diff 均为 `0.0625`，mean diff 分别约 `0.00395414/0.00395450`；但 compiled output SHA256
不同，exact-hash 强合同未满足。ON1 的初步 device P50 为 `0.41669 ms`，OFF1 为
`0.45537 ms`，峰值 allocated 为 `6,844,416/5,283,840 B`；这些只作诊断，尚不进入正式聚合。

下一步先让 runner 保存小型 compiled output，在两个 fresh worker 上逐元素比较 ON-vs-OFF，记录
max/mean diff、不同元素数和最大差位置。只有确认差异仍是允许的 FP16 reduction 重排、且相对
CPU/NPU eager 不劣化后，才修订“exact hash”合同并重新开始三轮性能。

逐元素诊断已完成：6144 个输出中 6142 个 exact equal，仅 2 个不同；diff histogram 为
`0:6142, 0.000244140625:1, 0.001953125:1`，max/mean diff 为
`0.001953125/3.57628e-7`。最大位置 OFF/ON 为 `-2.669921875/-2.671875`，是该量级 FP16 的
1 ULP；两侧相对 CPU/NPU eager 的 max diff 均仍为 `0.0625`，优化没有扩大误差上界。

因此 exact output hash 被判为不适用于该 FP16 reduction，而不是 pass correctness 失败。正式
补充合同改为：同 seed ON-vs-OFF max diff 不超过 1 FP16 ULP，且 ON 相对 CPU/NPU eager 的
max/mean error 不劣化到已登记容差外。`perf2_off_round1/perf2_on_round1` 的 compiled hash 与
诊断 worker 分别完全复现，可作为未挑选的 round 1；后续继续 round 2/3。

## 三轮代表性能

固定 FP16 `B=2,H=12,S=256,buckets=32`，交错顺序为 OFF1/ON1/ON2/OFF2/OFF3/ON3；每个
fresh worker warmup 10、runs 100，Ascend 910B2/NPU 1。三轮正确性、backend marker 和 kernel
结构均重复通过，OFF/ON compiled output hash 各自在三轮内完全稳定。

| round | host P50 OFF/ON ms | host P99 OFF/ON ms | device P50 OFF/ON ms | device P99 OFF/ON ms |
|---:|---:|---:|---:|---:|
| 1 | `0.50463 / 0.46671` | `0.52628 / 0.50391` | `0.45537 / 0.41669` | `0.47196 / 0.43050` |
| 2 | `0.50955 / 0.70855` | `0.57050 / 5.00725` | `0.46032 / 0.42148` | `0.49090 / 0.49652` |
| 3 | `0.50885 / 0.47530` | `0.61730 / 0.52779` | `0.45885 / 0.42671` | `0.49292 / 0.44676` |
| 中位数 | `0.50885 / 0.47530` | `0.57050 / 0.52779` | `0.45885 / 0.42148` | `0.49090 / 0.44676` |

中位数改善：host P50/P99 `6.59%/7.49%`，device P50/P99 `8.14%/8.99%`。三轮 device
P50 都改善 `7.00%–8.49%`；device P99 的 round2 回退 `1.14%`，另两轮改善。host round2
出现 P50/P99 `0.70855/5.00725 ms` 的明显长尾，不能删除；紧邻 OFF2 没有同量级长尾，因此最终
installed/生产判断仍需扩大 host-tail 采样。

ON 将 kernel 数从 1 增至 2，观测 peak allocated 固定从 `5,283,840` 增到 `6,844,416 B`，
额外 `1,560,576 B`。首次编译+运行中位数从约 `22.13 s` 增到 `41.74 s`，因为 ON 需要编译
额外 permute-copy kernel；强制禁缓存条件下该数值不外推生产 SLA。

当前只把目标 T5 同构代表 shape 判为 device 侧 `supported-beneficial`；默认 ON 的完整 verdict
仍为 `host-tail-memory-coverage-pending`。下一步先验证 BF16/FP32、dynamic replay，以及
InputBuffer、stride-1 inner、singleton inner 三类 guard negative，再决定是否需要缩小 gate。

## 覆盖扩展合同

- positive：BF16/FP32 static smoke 与 FP16 dynamic `S=64→72`，都必须实际加载 experimental，
  数值通过且生成 `permute-copy + consumer` 两个 Triton kernel；
- InputBuffer negative：非 unit inner stride 但 permute 输入直接是图输入，必须保持 lazy view，
  只生成一个 consumer kernel；
- stride-1 negative：输入是 pointwise producer，但 permute 后 inner stride 可证为 1，必须一个 kernel；
- singleton negative：输入是 pointwise producer、inner stride 非 1，但输出 inner length 为 1，
  必须一个 kernel；
- 每项 fresh process、独立 cache，固定 ON；保存 output code、backend、精度和 kernel count。

## 覆盖结果与最终边界

6/6 fresh worker 通过：

| case | dtype/shape | 预期/观察 kernel | 数值结果 |
|---|---|---:|---|
| positive | BF16 static S64 | `2/2` | vs eager max/mean `0.25/0.01618`，3%/0.5 合同通过 |
| positive | FP32 static S64 | `2/2` | vs eager max/mean `0/0` |
| positive dynamic | FP16 S64→S72 | `2/2` | 两 shape max diff 均 `0.03125` |
| InputBuffer negative | FP16 S64 | `1/1` | max diff `0` |
| stride-1 inner negative | FP16 | `1/1` | max diff `0.0625`，共同 reduction 容差通过 |
| singleton inner negative | FP16 | `1/1` | max diff `0` |

coverage runner SHA256 为 `eeb5cca7...ce7fd32`；六份 result JSON SHA256 已保存于结果目录。
上述结果证明当前 dtype/dynamic 与三个关键 guard 边界可用，不需要修改 `npu_permute()`，也不需要
手写 Triton 替身。

T-059 最终 verdict 为 `supported-beneficial-host-tail-memory-environment-monitor`：目标代表 shape
device P50/P99 改善 `8.14%/8.99%`，功能和 guard 通过；代价是额外一个 copy kernel、peak
allocated +`1,560,576 B`、强制禁缓存首编增加，且一轮 host P99 达 `5.00725 ms`。此外所有
Triton fresh compile 仍使用 T-022 audit-only host shim，正式匹配 headers 环境需复验。当前保持
installed 默认 ON，不做 source 修改；主线进入 `rsplit_outer`。
