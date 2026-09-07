# T-079/T-080 NPU 功能验证与修复实施计划

> 更新时间：2026-09-07 09:32 CST（UTC+08:00）
>
> 状态：GPU reference 已冻结；NPU 功能 worker 尚未实现，本文件是实现与执行合同，不是实测结果
>
> 后端：仅 `triton_experimental`

通用环境、原生优先、0 tests 判定、最小适配和修复回归方法见
[`NPU_FUNCTION_REPAIR_WORKFLOW.md`](NPU_FUNCTION_REPAIR_WORKFLOW.md)。本文件把该流程具体化到 T-079
和 T-080 的七个 acceptance units，说明每一步应看什么、可能在哪一层分歧以及何时允许修复。

## 1. 每个单元的固定执行顺序

1. fresh process 运行原生 community nodeid，记录 tests ran/skip/return code；
2. 原生未进入 test body 时，只移植设备、测试实例化与 artifact capture；
3. 先证明 `triton_experimental` 已加载，再运行正例和全部负例；
4. 记录 target counter、readable/transformed FX、generated code、数值/梯度；
5. 用 GPU 1.3 review 对照输入、shape、正负 guard 与结构，不直接比较不同 backend 的代码字符串；
6. 找到首个分歧并分类；
7. 只有 `NPU_REGRESSION` 或单独评审通过的 capability 缺口才进入源码修复；
8. 修复后重跑本单元全部 variants 和相邻 pass；
9. 写入 NPU result/comparison，功能门禁通过后才实现性能 worker。

每个单元使用独立目录：

```text
/home/z50063656/tmp/t079-npu-results/<unit>/<run-id>/
/home/z50063656/tmp/t080-npu-results/<unit>/<run-id>/
├── environment.json
├── native.stdout.log
├── native.stderr.log
├── adapter_result.json
├── fx_before.txt
├── fx_after.txt
├── generated_code.py
└── artifact_inventory.json
```

## 2. T-079：bmm→mm

GPU 合同：batch=1 改写为 `squeeze→mm→unsqueeze`；batch=3 保留 bmm。

NPU 功能步骤：

1. 原生运行 `test_pattern_matcher.py::TestPatternMatcher.test_bmm_to_mm`；
2. 若 GPU_TYPE/HAS_GPU 阻断，只显式实例化同一方法并将 device 设为 NPU；
3. batch=1 断言 target counter、transformed FX 和 generated code 指向 mm 路径；
4. batch=3 断言 target counter 为零且保留 bmm；
5. 两路均比较 eager/compiled，必要时检查 forward/backward；
6. 若只因 joint-graph generic device guard 拒绝，标 `CAPABILITY_PENDING`，先做测试态最小探针；
7. 探针正确且性能有益前，不扩展产品 device guard。

修复边界：不能把 batch=3 放宽；不能以 NPU bmm fallback 数值正确冒充 batch=1 rewrite 已生效。

## 3. T-079：cat→slice→cat

GPU 合同包括三条路径：合法 size 折叠；越界 size 和负 end 进入 handler 后 fallback。

NPU 功能步骤：

1. 原生运行 `test_cat_slice_cat_cuda`；
2. 最小适配只处理设备和 CUDA 后缀测试实例化；
3. 三个分支分别记录 handler marker；
4. 合法分支必须证明两个 cat/slice 合同被折成目标图；
5. 越界和负 end 即使 handler 被调用，也必须证明最终保留原语义；
6. 检查输出值、shape、stride，以及是否错误复用 storage/alias。

修复边界：handler matched 不等于 rewrite applied。若 NPU 只是在 codegen 字符串上不同，先寻找等价
FX/IR/task 证据，不为通过 CUDA FileCheck 而修改产品代码。

## 4. T-079：split→cat 与 cat→split

两个方向必须分开验证：

- split→cat：完整、同序、同 dim 才可直接返回原输入；缺片、异维、重排必须拒绝；
- cat→split：split 边界等于原输入边界、dim 相同且 cat 单用户才可消除；多用户、异维、异数量、
  异边界必须拒绝。

NPU 功能步骤：

1. 分别运行两个社区方法，不把方向相反的 replacement 合为一个 counter；
2. 正例记录 replacement 节点和节点消除数量；
3. 每个负例记录其具体 guard，不只写“未命中”；
4. 比较数值、输出 tuple/list 结构、shape/stride 和 alias；
5. 若 pattern 本身设备无关、仅测试类受 GPU gate 阻断，修 tracker adapter，不修改 PyTorch pass。

## 5. T-080：常量 scatter→pointwise

GPU 合同：稀疏 scalar scatter 从 full+mutation 改为 selector broadcast 与 `where`；shape、密度、base
常量和低精度 dtype guard 必须保持，CrossEntropy backward 是端到端入口。

NPU 功能步骤：

1. 先运行八个社区入口；记录专属 target metric，而非全局 pattern 总数；
2. 3D、非末维、负 dim 正例应形成等价 `iota/eq/where` 或 NPU 对应 pointwise IR；
3. short-index、dense-selector、non-constant-base 必须保留 scatter/mutation；
4. FP16/BF16 输出 dtype 不得提升；
5. CrossEntropy backward 比较梯度并确认目标 metric；
6. 排除 CPU fallback，检查生成的 NPU Triton kernel/IR；
7. 功能通过后才能使用社区 CrossEntropy `DO_PERF_TEST` shape 做 OFF/ON。

修复边界：不能缩小 selector 或删除密度 guard；正例在 joint-graph 后的 before/after 可能同形，应结合
target metric 与 generated code 判断。

## 6. T-080：prepare-softmax→online primitive

GPU 合同：`amax/sub/exp/sum` 变为 `prepare_softmax_online`，同时保持 fast-math 后续 log 和严格
signed-zero 语义。

NPU 功能步骤：

1. 运行 fast-math、signed-zero 和 community perf 默认小 shape 三个入口；
2. 原生入口若只因上游 cuda/xpu extra-check 拒绝，记录为 `CAPABILITY_PENDING`；
3. 单独评审测试态最小适配，只扩展设备识别，不改 online-softmax 算法和数值条件；
4. 断言 transformed FX 出现 online primitive，生成代码实际使用 NPU reduction；
5. 比较 xmax、sum/log、正负零 signbit、BF16 容差；
6. 若 Triton reduction/helper 编译失败，定位到 helper/codegen 层，不回退成“pattern 不支持”；
7. 正负功能全通过后，OFF/ON 两臂使用相同 compiled `triton_experimental`，不能用 eager 充当 OFF。

修复边界：generic device guard 不是产品显式 disable，但也不能未经评审永久删除；先证明功能，再证明
性能，最后才决定是否形成产品补丁。

## 7. T-080：constructor mover

GPU 合同：安全的 CPU arange 构造器移动到设备侧并消除额外 copy；index_put 的 scalar constructor
属于受保护依赖，必须留在 CPU。

NPU 功能步骤：

1. 运行 arange 正例和 index_put 负例；
2. 正例比较输出，并记录 constructor device、device_put/copy、generated kernel/task 数；
3. 负例确认 CPU scalar constructor 和 index_put 依赖没有被错误移动；
4. 当前 FX debug 点可能早于 mover，不能只看 before/after；必须补 post-mover IR、wrapper 或 task
  证据；
5. CUDA token（例如 `empty_strided_cuda`）不能直接作为 NPU 判据，应换成等价 NPU codegen/IR 断言；
6. 若 mover 使用 generic GPU helper 没识别 NPU，先标 capability pending 并做最小设备探针。

修复边界：正例若只是 fallback copy 后数值正确，不算优化生效；负例不能为了减少 task 数破坏
index_put 的依赖、alias 或 device 语义。

## 8. 修复完成的最低文档集合

若某个单元发现并修复真实 NPU regression，必须同时形成：

```text
issues/<case-id>/复现报告.md
issues/<case-id>/根因分析.md
issues/<case-id>/修复验证报告.md
issues/<case-id>/代码合入描述.md
results/current/<unit>/npu_result.json
results/current/<unit>/comparison_result.json
```

每份文档都要记录北京时间时间戳、PyTorch/torch_npu commit、实际 backend、原生命令、最小适配、
首个失败层、补丁位置、修复前后结果、全 variants/邻近回归以及性能是否解锁。没有源码修复时也要在
复现报告明确写“只需测试入口适配”或“明确产品关闭，不修复”。

若修改产品代码，`根因分析.md` 与 `修复验证报告.md` 还必须包含可定位的代码框和必要调用栈：触发
测试、原实现决策点、首个分歧、修复实现、修复前后生成路径均给出仓库相对位置；调用栈必须从社区
test/adapter 入口连续到断言或异常点。数值错误没有 exception traceback 时，明确标注哪些节点来自
动态 artifact、哪些来自源码重建，不得只给自然语言概括。格式以
`issues/REF-decompose-mm-native/{根因分析.md,修复验证报告.md}` 为最低样例。

## 9. 当前执行优先级

1. 先实现 T-079 四单元的 NPU 功能 runner/adapter 和结构化结果；
2. 完成 T-079 comparison，只有 regression 才进入修复；
3. 实现并执行 T-079 性能 worker；
4. 再按同一流程顺序处理 T-080 三单元；
5. 两批都不得引用 default/DVM/MLIR 历史结果作为 `triton_experimental` 动态 verdict。
