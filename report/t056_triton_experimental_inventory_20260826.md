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
| `t056_build_experimental_inventory.py` | `4cf45c67...cdc9` |
| `experimental_pass_route_matrix.csv` | `6e0ea82d...2e1d` |
| `experimental_config_inventory.csv` | `69b1d629...1a66` |
| `experimental_feature_families.csv` | `c2af2532...c1ae` |

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

### 1. int→float→int elide：已证伪并 source 默认关闭

T-057 已覆盖 FP16/BF16/FP32 边界。Float32 pass ON/OFF 单点证明 `±(2**24+1)` 数值差 1；
三个 dtype 的 ON 都把 eager 新 tensor 改成 input alias。P-016 已把 source 默认值改为 False，
并通过默认 installer no-op/显式 opt-in probe；installed wheel 尚未重建。TE-FX-001 的 feature
verdict 已回填为 `correctness-failed-source-default-disabled-wheel-pending`，详细证据见
`t057_int_float_int_boundary_20260826.md`。错误结果不测性能。

### 2. GELU decomposition：installed 失败，P-017 source 已恢复合同

installed P-013 override 无论 `approximate="none"` 还是 `"tanh"` 都使用 sigmoid 形式的 tanh
近似，CPU reference 证明 none forward/backward 合同失败。P-017 current source 已让 none 使用
erf/CDF/PDF，tanh 保持原公式；FP32/FP16/BF16 六组 NPU source-overlay、非法参数和生成代码检查
通过。TE-DEC-003 已回填为 `upstream-contract-restored-source-wheel-pending`；共享 diff 可隔离前
不构建 wheel，也不对错误 installed-none 做性能测试。详见
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

### 4. addmm gate：首个 experimental cohort 有益，覆盖待扩展

default 历史中 8/8 代表配置收益超过 10%，而 experimental 默认永久关闭该 pattern。它是当前
最明确的性能提升候选。T-058 已用 fresh process 恢复激活前保存的原 check：FP16 vector-bias
从 `mm+Triton add` 变为单 extern addmm，三轮 p50 中位数改善 `22.17%`、p99 改善 `14.05%`，
峰值 allocated 不变。后续 11/11 capability 扩展、
unaligned 第二性能 cohort 和 NPU Event 分解也已完成，P-018 source 默认已改为启用，显式 opt-out
为幂等可恢复 wrapper；TE-GATE-002 已回填
`source-verified-beneficial-wheel-pending-host-tail-monitor`。source probe 通过，installed wheel 与
host-tail 复验仍 pending；详见 `t058_experimental_addmm_gate_20260826.md`。

### 5. permute-gather 与 outer rsplit：源码自述收益大，风险面也大

`realize_permute_gather` 注释记录 T5 softmax 相对位置 bias 约 110x 的历史观察；`rsplit_outer`
把单 reduction 变为 40-core partial+combine。二者都值得优先实测，但前者会引入物化 copy 和
alias/layout 变化，后者会引入 workspace、双 kernel、padding-lane 与 provenance 问题。必须同时
记录数值、task、allocated peak、首次编译和 steady P50/P99。

### 6. int64 boundary downcast：可用性优化也是边界正确性风险

NPU AI Vector Core 不支持原生 int64 算术，experimental 在 boundary/compute type 中降成 int32。
这可能让常规索引可编译，但超出 int32 的 shape/index/值不能从小样本推断。该功能族排在 P0，
先做边界负例，再谈 dedup 的任务/带宽收益。

## 下一步

T-057 三项 correctness 已完成，后续按以下顺序执行：

1. 已完成：backend 状态快照，结论 `backend-switch-isolation-failed`；
2. 已完成：int-float-int compiled 边界，P-016 source 默认关闭、wheel pending；
3. 已完成：GELU exact/tanh forward/backward，P-017 source 验证、wheel pending；
4. 已完成：addmm 11/11 扩展、两性能 cohort 与 P-018 source gate；wheel/host-tail pending；
5. 当前：permute-gather 与 outer rsplit 各选一个源码注释对应场景做功能和 paired 性能。

P-014/P-016/P-017/P-018 installed wheel 仍待共享 diff 可隔离后构建；在此之前后续 installed worker 必须
显式 `elide_int_float_int=False`，并标明 installed/current-source-overlay 边界。不得把
audit-only launcher shim 当作正式环境成功。
