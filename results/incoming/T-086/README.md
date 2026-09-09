# T-086 GPU handoff 接收入口

> 更新时间：2026-09-09 00:35 CST（UTC+08:00）

此目录只接收 T-086 实际 GPU 原生 reference 的文本 handoff，不是通过标记，也不存放 NPU 结果。

GPU 一键运行成功后，将以下单文件完整复制为本目录的 `text-handoff.json`：

```text
/data/z50063656/tmp/t086-reference-results/latest-text-handoff.json
```

若脚本提示 `handoff_upload_mode=split`，则不要上传上面的单文件；应将同一轮目录内的
`manifest.json` 和全部 `part-*.json` 一起放到本目录：

```text
/data/z50063656/tmp/t086-reference-results/latest/text-handoff-parts/
```

不得混入其他 task、旧 run 或缺失分片。控制节点拉取后必须先校验 payload、PyTorch commit、实际 CUDA
设备、2/2 cases、0 skip/xfail 和 FX artifacts，再决定是否冻结 1 个 acceptance unit。

具体功能、性能来源和边界见
[T-086 功能与性能测例讲解](../../../docs/T086_FUNCTION_PERFORMANCE_GUIDE.md)。
