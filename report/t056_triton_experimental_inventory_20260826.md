# T-056 triton_experimental 静态 inventory（2026-08-26）

## 结论

已为当前 2.14 源码建立独立的 experimental 路由层，不再把 default 的 251 行 verdict 直接
搬过来。静态结果分成三份：

- `experimental_pass_route_matrix.csv`：原 251 行逐条增加 experimental 路由；初始动态状态
  重置为 `not-run`，随后由生成器中的已验证结果表回填 T-057 状态；
- `experimental_config_inventory.csv`：69 个配置项的默认值、引用位置和绑定时机；
- `experimental_feature_families.csv`：把互相耦合的 config/lowering/codegen 行为合并为 35 个
  可验收功能族，明确它们不全是 FX pass。

生成器为 `t056_build_experimental_inventory.py`，只使用标准库，不导入 torch。当前结果说明：
experimental 不是“default 的 27 个 custom pass 加一个开关”，而是“203 条上游候选 + 2 个
自有 FX 变换 + 2 个明确关闭 gate + 一套独立 decomposition/lowering/codegen/autotune”。

## 源码与安装态边界

inventory 读取当前 source tree；当前安装仍是 P-013 wheel。source 现在还包含本任务 P-014、
P-017 的 `decomposition.py` 修改和 P-016 的 experimental `config.py`/注释改动；另有共享工作区其他
人的未安装 `codegen/triton.py` provenance 修改。本轮不覆盖、不归因后者，也不把这些 source
修改冒充 installed verdict。

| 产物 | SHA256 |
|---|---|
| `t056_build_experimental_inventory.py` | `ee1b73b4...88c9` |
| `experimental_pass_route_matrix.csv` | `b1c1cdf8...a46f` |
| `experimental_config_inventory.csv` | `1d804039...455` |
| `experimental_feature_families.csv` | `0c34580d...0c2a` |

这些 CSV 是静态任务地图，不是测试结果；不得把 `active` 路由写成 `supported`。

## 251 行 experimental 路由统计

| 路由 | 数量 | 含义 |
|---|---:|---|
| `inherited-upstream-needs-dynamic-validation` | 203 | 上游 Inductor 注册仍在，但 NPU lowering/codegen 可能改变结果 |
| `inactive-not-called-by-experimental-loader` | 32 | torch_npu 的 default-loader pass/pattern，未迁移就不执行 |
| `inactive-other-backend` | 8 | DVM/MLIR 专属 |
| `inherited-upstream-explicitly-disabled` | 4 | pad mm/bmm/addmm 三条与上游 addmm fusion 一条 |
| `active-experimental-disabled-gate` | 2 | 对上述 pad/addmm 行为施加 gate 的 experimental 记录 |
| `active-experimental-owned` | 1 | int→float→int 自有 post-grad pass |
| `active-experimental-owned-infrastructure` | 1 | 该 pass 的 installer，不是第二个独立 pass |

历史矩阵生成于 2026-08-20，没有包含后来加入的 Max(1) loop-merge 等当前 source 行为。因此
35 行 feature-family 表是必要补充，不能只看 251 行 overlay。

## 69 个配置项不等于 69 个 pass

引用分析得到：

- 55 个只在执行路径中 live-read；
- 12 个在模块导入时复制成全局快照；
- `all_blocks_parallel` 同时存在快照和 live-read；
- `reassociate_sum_of_add=True` 在当前 experimental 源码中零引用，属于 declared-unused。

导入时快照包括 `rsplit_outer`、`rewrite_int1_cast_as_ne`、`codegen_linearize`、
`int64_boundary_cast`、`dedup_downcast`、`group_dispatch`、`autotune_enhance`、
`elide_reduction_where`、`inject_care_padding`、`refactor_clamp_stride`、`mask_cmp_fp32` 和
`tail_bcast_min_vec`。这些开关在模块导入后用 `config.patch()` 改值，通常不会改变已绑定的
本地全局变量；A/B 必须在 fresh process、import 前设置或使用受控 source hook。

即使是 live-read，installer/monkeypatch 也可能不可逆，例如 addmm gate直接改已注册 pattern
entry，loop-merge 直接替换类方法。因此正式 pass-on/pass-off 一律用独立 fresh process，不能
假设同一进程 patch context 自动恢复产品状态。

## 35 个可验收功能族

按层分组如下；完整默认值、源码锚点和风险见 `experimental_feature_families.csv`。

| 层 | 代表功能族 | 首要验收点 |
|---|---|---|
| loader/隔离 | backend reload、default/experimental 往返 | registry、config、decomposition、pattern 不串态 |
| FX/gate | int-float-int、Max(1) merge、pad/addmm disable | 语义边界、精确触发、pass-on/off |
| decomposition | matmul fold、softmax backward、GELU、RMSNorm、dropout | eager 合同、训练、dtype/rank/特殊参数 |
| lowering/IR | fallback policy、slice/expand/permute realize、index_put、safe stride | fallback、copy/alias/layout、dynamic shape |
| codegen | rsplit、int1、extract-slice、index/range-tree folds、reduction tiling | OOB、边界、workspace、generated code、性能 |
| dispatch/autotune | group dispatch、int64 boundary、tile search、MSPTI | grid 覆盖、候选有效性、计时可靠性 |
| wrapper/compiler | NPU allocation、rsplit workspace、VF fusion | 内存规划、首次编译、opt-in 行为 |

## 静态高风险与高收益候选

### 1. int→float→int elide：已证伪并完成默认关闭 wheel 验证

T-057 已覆盖 FP16/BF16/FP32 边界。Float32 pass ON/OFF 单点证明 `±(2**24+1)` 数值差 1；
三个 dtype 的 ON 都把 eager 新 tensor 改成 input alias。P-016 已把 source 默认值改为 False，
并把 installer 改为始终、幂等注册，pass body 在 rewrite 时读取动态配置。独立 wheel 的默认
关闭路径精确一致且不 alias；导入后显式开启能真实激活旧 rewrite，并按预期复现边界错误与
alias。TE-FX-001 的 feature verdict 已回填为
`correctness-failed-default-disabled-verified-installed-wheel`，详细证据见
`t057_int_float_int_boundary_20260826.md` 和
`issues/P016_int_float_int_gate/修复验证报告.md`。错误结果不测性能。

### 2. GELU decomposition：installed 历史失败，P-017 wheel 已恢复合同

installed P-013 override 无论 `approximate="none"` 还是 `"tanh"` 都使用 sigmoid 形式的 tanh
近似，CPU reference 证明 none forward/backward 合同失败。P-017 current source 已让 none 使用
erf/CDF/PDF，tanh 保持原公式；FP32/FP16/BF16 六组 NPU source-overlay、非法参数和生成代码检查
通过。随后独立 wheel 的六组无 overlay NPU 验证、非法参数 compile 与 P-014 近邻回归也通过。
TE-DEC-003 已回填为 `upstream-contract-restored-verified-installed-wheel`；仍不对错误的历史
installed-none 做性能测试。详见
`t057_gelu_approximate_20260826.md`。

### 3. backend 隔离已动态确认失败

P-014 修复的是 default registrar 的 erfc 重复注册崩溃，但静态源码还显示：

- experimental 直接把 addmm pattern `extra_check` 替换成永久 False lambda；
- `layout_optimization`、`coordinate_descent_tuning`、`split_reductions` 和 `shape_padding` 被全局改写；
- experimental decomposition 表覆盖 GELU/dropout/softmax/RMSNorm，而
  `restore_inductor_baseline()` 只恢复 lowering/scheduler；
- int-float-int installer 会组合已有 `post_grad_custom_post_pass`。

T-057 无模型状态快照已确认：34 个 addmm check、五项 config/guard、matmul fold 与五个
decomposition 回切 default 后仍残留 experimental 状态。TE-INFRA-001 已回填
`backend-switch-isolation-failed`；P-015 设计中。跨 backend 对照必须使用不同 fresh process。

### 4. addmm gate：独立 wheel 已闭环，host tail 保留监控

default 历史中 8/8 代表配置收益超过 10%，而 experimental 默认永久关闭该 pattern。它是当前
最明确的性能提升候选。T-058 已用 fresh process 恢复激活前保存的原 check：FP16 vector-bias
从 `mm+Triton add` 变为单 extern addmm，三轮 p50 中位数改善 `22.17%`、p99 改善 `14.05%`，
峰值 allocated 不变。后续 11/11 capability 扩展、
unaligned 第二性能 cohort 和 NPU Event 分解也已完成，P-018 source 默认已改为启用，显式 opt-out
进一步修成 match-time 读取配置的幂等 live wrapper。专属 wheel 的正常用户顺序
default/late-opt-out/restore 和 11/11 capability 全部通过；installed shape-A/unaligned host p50
改善 `17.10%/13.52%`，unaligned Event device p50/p99 改善 `19.87%/19.10%`，显存不增加。
host 同步长尾原样保留为监控项；TE-GATE-002 已回填
`verified-installed-wheel-beneficial-host-tail-monitor`。详见
`t058_experimental_addmm_gate_20260826.md` 和
`../issues/P018_addmm_gate/修复验证报告.md`。

### 5. permute-gather 与 outer rsplit：代表场景均已闭环

T-059 在现有安装态完成 permute-gather 6/6 dtype/dynamic/guard 覆盖；代表 shape 的 device
P50/P99 改善 `8.14%/8.99%`。代价是一个额外 copy kernel、peak +`1,560,576 B`、首编增加和
host P99 长尾，因此保持默认 ON、不修改源码，状态为
`supported-beneficial-host-tail-memory-environment-monitor`。

T-060/P-019 只在 rsplit 局部接受 generic Inductor 的 `ReductionHint.DEFAULT`，保留单 sum、
非 Welford、唯一输出、size/free-axis 和 nested-reduction 等既有门禁，不新增 kernel。专属 wheel
target UT 5/5、安装态矩阵 9/9；三轮 Event device P50/P99 改善中位数 `32.67%/28.58%`，host P50
改善 `26.20%`。peak +`393,728 B`、首编 +`96.48%` 和 host P99 长尾继续监控。

### 6. int64 boundary downcast：正确性已由 dtype-aware fallback 恢复

T-061/P-020 证明 launcher 把 int64 数据降成 int32 会在边界算术中静默环绕。修复只对显式
int64 数据 overload 路由 NPU ATen fallback，FP32 Triton 与 embedding/gather index 路径保持。
专属 wheel target UT 6/6、安装态矩阵 8/8 exact；性能以正确 eager 基线归档且 peak 不变，状态为
`installed-wheel-verified-correctness-restored-performance-characterized`。

### 7. header tiling/odometer：有效默认策略保留，两项配置清理

T-065/TE-CG-008 在 P-022 安装态完成五项开关 10/10 正确性和三个有效开关 18/18 paired
性能。input-stride device P50/P99 改善 `27.81%/24.23%`，odometer 改善
`3.44%/5.57%`，align-8 在约 2% 内中性，三项默认值保留。`unify_block` 和
`pad_min_block_to_8` 在当前 greedy allocator 中不可生效，只登记配置清理，不恢复旧路径。
状态为 `verified-active-defaults-retained-two-configs-ineffective-cleanup`。

### 8. group/all-block dispatch：保留实际 group 路径，修复候选双门控

T-066/TE-CG-009 完成静态契约、6/6 experimental Inductor 正确性、65535/65536 边界和
三轮性能。`group_dispatch` config 无消费者，但非 A5 48-core group codegen 相对 backend
auto size-1 的 device P50/P99 改善 `3.15%/3.55%`，因此保留实际路径并仅登记死配置清理。
P-023 只有在 torch_npu config 与 `TRITON_ALL_BLOCKS_PARALLEL` 同时开启时才生成 2/4/8；
独立 wheel/venv 的 target UT 3/3、近邻 2/2、candidate 两态、dynamic smoke 与 lint 均通过。
状态为 `verified-group-retained-dead-config-recorded-p023-local-fix`。

### 9. autotune：保留公式默认路径，修复动态 gate 生命周期

T-067/TE-AUTO-001 完成候选契约、9/9 NPU E2E、12/12 三轮 paired runtime 与 P-024
安装态验证。公式路径相对 legacy 在 pointwise 上减少 `23.18%` 首编、`100%` 调优、
`92.42%` peak 和 `7.76%` device P50；reduction 首编/调优改善 `6.43%/27.25%`，选中
tile 不变。P-024 删除 `autotune_enhance` 导入快照，三类入口 live patch UT 与真实 NPU
开/关两态通过。状态为 `verified-default-formula-retained-p024-live-gate-fix`。

### 10. wrapper：保留快速 allocation，消除 memory planner 复制漂移

T-068/TE-WRAP-001 完成静态、安装态和真实 NPU wrapper 闭环。NPU direct allocation 三轮
P50 中位 `3.167990 us`，dispatcher 为 `7.156705 us`，改善 `55.73%`，因此保留。P-025
只预先移除末尾无 `.name` 的 WorkspaceArg planning line，再委托上游
`memory_plan_reuse()`；基线 2 FAIL→候选 2 PASS，allocation UT 11/11，pointwise 和真实
outer-rsplit workspace 均通过。状态为
`verified-fast-allocation-retained-p025-upstream-planner-delegation`。

## 下一步

experimental 第一批 P0 correctness/performance 闸门已完成，当前顺序为：

1. 已完成：backend 状态快照，结论 `backend-switch-isolation-failed`；
2. 已完成：int-float-int compiled 边界，P-016 独立 wheel 默认关闭与 late opt-in；
3. 已完成：GELU exact/tanh forward/backward，P-017 独立 wheel 安装态验证；
4. 已完成：addmm 11/11 扩展、动态 late opt-out、两性能 cohort 与 P-018 独立 wheel；host-tail
   转为持续监控；
5. 已完成：T-059 permute-gather 与 P-019 outer rsplit 功能、guard、资源和 paired 性能；
6. 已完成：P-020 int64 dtype-aware fallback 的正确性、近邻控制组和正确基线性能；
7. 已完成：T-062/TE-LOW-001 registry、keep-list、12/12 功能矩阵和两 cohort 三轮性能；现有
   fallback 策略通过，全局 `ceil` generate 因 standalone 回退否决；
8. 已完成 T-063/TE-CG-005：六项 AST/合成契约、NPU 14/14 配置和 18/18 对照通过；保留
   默认值并监控 device p99/host tail；
9. 已完成 T-064/TE-CG-007：P-021/P-022 安装态 UT 6/6、7/7，NPU 正确性 18/18；
   `rtree_real_block` 携 kept-axis guard 保持开启，flatten 正确性恢复但因 device P50 回退
   147.96% 保持默认关闭；
10. 已完成 T-065/TE-CG-008：五项配置正确性 10/10，三个有效开关 paired 性能 18/18；
    input-stride/odometer 保持开启，align-8 中性保留，unify/padding 两项无效配置登记清理；
11. 已完成 T-066/TE-CG-009：group config 死配置已记录，48-core group 路径保留；P-023
    双门控独立 wheel/venv 验证通过；
12. 已完成 T-067/TE-AUTO-001：默认公式路径保留，P-024 live gate 独立 wheel/venv 验证通过；
13. 已完成 T-068/TE-WRAP-001：fast allocation 保留，P-025 planner 委托独立 wheel/venv
    验证通过；
14. 已完成 T-069/TE-AUTO-002：默认 MSPTI device-time 保留，自动加载、显式预加载、Event
    opt-out 与运行态失败回退均验证通过；
15. 已完成 T-070/TE-FX-002：动态卷积 pointwise `3→1` 轴、空输出正确、关闭态隔离图
    rank-mismatch，三轮 device 性能中性且峰值相同；默认保留，无产品修改。
16. 已完成 T-071/TE-GUARD-001：当前上游默认 recursive=False；深层 NPU 训练两态正确，
    三轮 CPU guard 量化关闭 fast path 的 P50 开销中位 `36.57%`；默认安全覆盖保留。
17. 已完成 T-072/TE-DEC-001：P-026 把 rank-sorted 广泛 matmul fold 收窄到精确 seq-first
    stride；目标 P50 中位改善 `5.53%`、peak 下降 `83.09%`，三类扩大作用域回退恢复上游。
18. 当前 T-073：TE-DEC-002 softmax backward no-FMA。剩余 P1 13 个、P2 4 个。

P-014/P-016/P-017/P-018/P-019/P-020/P-021/P-022/P-023/P-024/P-025/P-026 已用 detached worktree 和独立 venv 完成 installed wheel 验证。共享
Benchmark 安装态仍是 P-013，因此在该共享环境运行的后续 worker 仍须显式
`elide_int_float_int=False`，并标明 installed/current-source-overlay 边界。不得把
audit-only launcher shim 当作正式环境成功。
