# mm_plus_mm different-K large profiler、tile 与内存分解报告

## 1. 结论

T-022 将 T-020 large `(M,K1,N,K2)=(512,768,640,384)` 的不稳定结果拆成了 device kernel、host synchronized paired benchmark 和 allocator peak 三层。

三个 candidate tile 都正确，且 profiler 的 10 个 active step 都是单个 `different_k_mm_plus_mm_kernel`。device p50 由 `64x64x64` 的 `29.27 μs` 降到 `64x128x64` 的 `24.33 μs`，再降到 `128x128x128` 的 `14.41 μs`；task 本体没有出现 T-020 的毫秒级长尾。

但是 paired host p50 的改善只有 `6.64%/6.97%/7.55%`，三个 tile 都低于预先登记的 10% 门槛。`128x128x128` 的 device kernel 只占 candidate 同步样本约 5.6%，启动、同步和 runtime 固定开销吞没了大部分 tile 收益。因此 large 的结论从 T-020 的 `p50-pass-but-unstable-hold` 收敛为 `supported-neutral-hold`：功能可用、kernel 稳定，但暂不进入首批自动 capability gate。

内存分解显示 candidate 稳态 additional allocated peak 比 baseline 多 `655,360 B`，恰为一个逻辑 fp16 output 大小；T-020 的 `5,899,264 B` 受 benchmark 前首轮输出持续存活影响，不能继续作为 steady candidate workspace。reserved peak delta 为 0 只表示 allocator 已预留缓存，不能解释为零 workspace。

本轮没有向 Inductor 注册 candidate，也没有修改 PyTorch、torch_npu 或 Triton Ascend 产品源码。

## 2. 环境与运行合同

| 项目 | 值 |
|---|---|
| Python | 3.11.15 |
| PyTorch | `2.14.0a0+git8e86e0a`，commit `8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b` |
| PyTorch 安装形态 | editable，指向 `/home/z50063656/Benchmark/pytorch-upstream` |
| torch_npu | source-built wheel，commit `83cc452480c3546fd5cccf853bfe3a360ce9dbfc` |
| torch_npu wheel | `dist/torch_npu-2.14.0a0+git83cc452-cp311-cp311-linux_aarch64.whl`，SHA256 `263ffec23ae37c651f3d57199c0cfa8b14f398ea603f8e1434ea14b237792704`，`--no-deps` |
| Triton | runtime 3.2.0；Triton Ascend source commit `8bd9f380d2786002b84b5248f00838c26f900515` |
| CANN / NPU | CANN 9.0.1 / Ascend910B2 |
| backend | `triton_experimental` |
| shape / dtype / layout | `(512,768,640,384)` / fp16 / contiguous / static |
| profiler | warmup 10；Level0/AiCoreNone；profile warmup 1、active 10；每 step 同步 |
| paired | baseline/candidate 各 warmup 10、runs 100、3 轮交错 |
| memory | pure output/baseline/candidate 各 warmup 3、runs 10；同时记录 allocated/reserved |

所有测试从 `/home/z50063656/tmp` 启动，使用运行前后进程表均为空的 NPU 6；首个 `64x64x64` profile 先前在运行前后空闲的 NPU 7 完成。paired baseline 都设置 `TORCH_COMPILE_DEBUG=1`、独立 debug 目录和 `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`。baseline 全部成功，按图模式诊断规则没有读取成功用例的 `output_code.py`。

不同 fresh process 的绝对 host 时间会漂移，所以 tile verdict 以每个进程内 baseline/candidate 成对改善率判断；不拿不同进程的 baseline 绝对时延作硬件结论。

## 3. Fresh launcher 环境限制与审计垫片

共享 Triton cache 被其他任务清理后，新 tile 首次暴露出当前安装组合不能 fresh compile host launcher：

1. PyTorch 是 editable 安装，`torch.__file__` 指向源码树，但源码树当前没有 `torch/include`，首先报 `ATen/ATen.h` 缺失。
2. wheel 中存在完整 PyTorch headers，但 PyTorch 2.14 headers 要求 C++20，而安装的 Triton Ascend 3.2.0 launcher 固定追加 `-std=c++17`。
3. installed torch_npu headers 引用了 CANN 9.0.1 headers 中没有声明的 `aclmdlRICondHandle` 和 `aclmdlRICondTaskParams`。

这与 `torch_npu/test/dynamo/test_compile_trigger.py` 对“PyTorch 2.13+ C++20 / older CANN missing conditional graph types”的环境限制说明一致。未获取 NPU 设备权限时还出现过 `drvGetDevNum=DRV_ERROR_INNER_ERR(7)`；显式设备权限下设备枚举正常，因此那两次结果只属于执行沙箱权限，不是 NPU 运行时故障。

为继续测量 device kernel，本轮使用两个审计文件：

- `t022_launcher_cc_wrapper.sh`：只把 host launcher 命令的 `-std=c++17` 改为 `-std=c++20`。
- `t022_cann_header_compat.h`：只前置声明 launcher 不会调用的两个 conditional graph 类型。

运行时再以 `CPATH` 指向 wheel headers、`TRITON_DISABLE_PRECOMPILE=1` 跳过 GCH，并把 `TRITON_CACHE_DIR` 隔离到 T-022 results。这个垫片只影响 host launcher 的 C++ 解析，不修改 Triton device kernel；正确性、单 task 和 device duration 仍可用于 tile 比较。它不是产品修复，也不能证明正式环境的 fresh compilation 已通过。正式接入必须在匹配 PyTorch C++20、torch_npu 和 CANN headers 的干净环境中复验。

## 4. 正确性与 device profiler

三个 tile 在 profile 前后都得到相同的 candidate 误差：max/mean absolute error 为 `0.0625/8.7023e-7`，输出 dtype 为 fp16、shape 为 `(512,640)`，并通过预设 `rtol=atol=0.01`。

| tile | active tasks | kernel groups | mean±stdev | p50 | p99 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| `64x64x64` | 10/10 | 1 | `29.280±0.598 μs` | `29.27 μs` | `30.04 μs` | 单 task，正确 |
| `64x128x64` | 10/10 | 1 | `24.408±0.349 μs` | `24.33 μs` | `25.14 μs` | 单 task，正确 |
| `128x128x128` | 10/10 | 1 | `14.448±0.329 μs` | `14.41 μs` | `14.98 μs` | 单 task，正确，device 最优 |

三个 profile 都有 ACL→NPU flow 关联告警，但导出的 `kernel_details.csv` 各有完整 10 条同名 kernel；本报告只使用 device task duration，不用包含显式同步的 step gap。

## 5. Paired 稳态性能

### 5.1 三轮中位数

单位为 ms。

| tile | baseline p50 | candidate p50 | p50 改善 | baseline p99 | candidate p99 | p99 改善 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `64x64x64` | 0.299175 | 0.279320 | 6.64% | 0.324950 | 0.299940 | 7.70% | neutral |
| `64x128x64` | 0.272655 | 0.253640 | 6.97% | 0.300030 | 0.274450 | 8.53% | neutral |
| `128x128x128` | 0.275920 | 0.255100 | 7.55% | 0.292130 | 0.271480 | 7.07% | neutral |

三组都低于预设的 `>10%` p50 gate。T-020 large 隔离重测得到的 11.58% 是高方差、单轮回退条件下的三轮中位数；T-022 三组更稳定的 paired 结果优先用于 capability verdict。

### 5.2 每轮 mean±stdev

| tile / mode | round 1 | round 2 | round 3 |
|---|---:|---:|---:|
| `64x64x64` baseline | 0.304933±0.007842 | 0.304547±0.017242 | 0.300130±0.008764 |
| `64x64x64` candidate | 0.281282±0.006143 | 0.279953±0.006106 | 0.280669±0.006696 |
| `64x128x64` baseline | 0.280695±0.027991 | 0.276082±0.008541 | 0.275959±0.009252 |
| `64x128x64` candidate | 0.256691±0.005624 | 0.255918±0.005711 | 0.255787±0.009913 |
| `128x128x128` baseline | 0.276201±0.004962 | 0.277245±0.004658 | 0.278002±0.003981 |
| `128x128x128` candidate | 0.257691±0.015490 | 0.255106±0.003757 | 0.256136±0.003797 |

仅观察到几个 0.4–0.55 ms 单点，没有复现 T-020 的多毫秒长尾。编译期继续提示当前设备 SM 数不足以采用 max-autotune GEMM，所以 baseline 表示当前实际 different-K fallback，不声称已达到 vendor GEMM 理论最优。

### 5.3 Device 与 host 的关系

| tile | device p50 | candidate host p50 | device/host |
|---|---:|---:|---:|
| `64x64x64` | 29.27 μs | 279.32 μs | 10.5% |
| `64x128x64` | 24.33 μs | 253.64 μs | 9.6% |
| `128x128x128` | 14.41 μs | 255.10 μs | 5.6% |

host sample 包含 Python launcher、runtime enqueue 和每次显式 `torch.npu.synchronize()`。三组相减得到的非 kernel 部分约为 0.23–0.25 ms，且跨 fresh process 有漂移；不能把它当作精确分层计时，但足以说明 large 的主要端到端瓶颈已不在 candidate device task。

## 6. 内存分解

三个 tile 得到完全一致的内存结果：

| 项目 | allocated before | max allocated | additional allocated peak | reserved before/max | additional reserved peak |
|---|---:|---:|---:|---:|---:|
| pure `torch.empty(512,640,fp16)` | 3,312,128 B | 3,968,000 B | 655,872 B | 73,400,320 B | 0 B |
| baseline | 3,312,128 B | 4,623,872 B | 1,311,744 B | 73,400,320 B | 0 B |
| candidate | 3,312,128 B | 5,279,232 B | 1,967,104 B | 73,400,320 B | 0 B |

逻辑 output 大小为 `512*640*2=655,360 B`，allocator 对齐后 pure output 为 `655,872 B`。candidate 相对 baseline 多 `655,360 B`，即一个逻辑 output；candidate 超过 pure output 的部分为 `1,311,232 B`，可能来自 Triton workspace、padding 或 runtime 临时量，当前 API 不能继续细分。

T-020 在 paired rounds 前保留了首次 `baseline_output` 和 `candidate_output`，T-022 在采样前显式删除并同步；因此 T-022 的 steady decomposition 比 T-020 的 `5,899,264 B` 更适合 capability gate。reserved delta 为 0 是 allocator cache 已预热的结果，不代表 kernel 不需要临时量。

## 7. Capability 与下一步

| 范围 | T-022 结论 | 处置 |
|---|---|---|
| large fp16/contiguous/static | 三 tile 正确、单 task、device 稳定 | 功能支持 |
| large 端到端性能 | p50 改善 6.64%–7.55% | `supported-neutral-hold`，不自动启用 |
| large 首选 tile | `128x128x128` device p50 最低 | 只作为后续候选配置，不是自动 gate 证据 |
| host launcher fresh compile | 当前安装组合不兼容 | 正式集成前必须修复环境合同并复验 |
| 已通过的较小 static cohort | T-016/T-020 多个配置 p50 >10% | 可继续 T-021 的默认关闭、fallback-first 正式接入设计 |

T-022 不授权对任意 shape 全局启用 candidate。下一阶段应把 large 排除在首批 gate 外，先在默认关闭开关下接入已验证的较小 static cohort；同时建立干净的 PyTorch C++20 / torch_npu / Triton Ascend / CANN headers 组合，验证 fresh launcher compile。任何 product path 都必须始终保留当前两个 mm + add extern fallback。

## 8. 原始证据

- `64x64x64` profiler：`results/t022_mmplus_different_k_large_20260821/profile_bm64_bn64_bk64/result.json`
- `64x128x64` profiler：`results/t022_mmplus_different_k_large_20260821/profile_bm64_bn128_bk64_audit_shim/result.json`
- `128x128x128` profiler：`results/t022_mmplus_different_k_large_20260821/profile_bm128_bn128_bk128_audit_shim/result.json`
- `64x64x64` paired：`results/t022_mmplus_different_k_large_20260821/benchmark_bm64_bn64_bk64_audit_shim/result.json`
- `64x128x64` paired：`results/t022_mmplus_different_k_large_20260821/benchmark_bm64_bn128_bk64_audit_shim/result.json`
- `128x128x128` paired：`results/t022_mmplus_different_k_large_20260821/benchmark_bm128_bn128_bk128_audit_shim/result.json`
- sandbox 设备权限证据：`profile_bm64_bn128_bk64/result.json`、`profile_bm64_bn128_bk64_rerun1/result.json`
- fresh launcher 编译失败证据：`profile_bm64_bn128_bk64_rerun2/result.json`、`profile_bm64_bn128_bk64_rerun3/result.json`、`profile_bm128_bn128_bk128/result.json`
- [T-020 扩展性能与内存报告](t020_mmplus_different_k_extended_benchmark_20260821.md)
- [T-021 正式接入设计](t021_mmplus_different_k_integration_design_20260821.md)
