# T-084 GPU 回传入口

> 更新时间：2026-09-09 00:34 CST（UTC+08:00）

GPU 一键入口完成后，将：

```text
/data/z50063656/tmp/t084-reference-results/latest-text-handoff.json
```

保存为本目录的 `text-handoff.json`。若入口提示 `handoff_upload_mode=split`，则改为上传同一轮次的
`manifest.json` 与全部 `part-*.json`，不能混入其他任务或旧轮次。

本批只执行 `REF-dedup-reduce-scatter-native`：社区原生测试使用真实 CUDA/NCCL，但
`world_size=1`，只建立功能和改图 reference，不证明两 rank 通信收益。真实性能必须等 NPU
`triton_experimental` 两 rank 功能门禁签发后再运行。

完整说明见 [T-084 功能与性能讲解](../../../docs/T084_FUNCTION_PERFORMANCE_GUIDE.md)。
