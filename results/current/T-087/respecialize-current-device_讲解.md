# T-087：设备节点消除与运行时设备编号

> 更新时间：2026-09-11 18:44 CST（UTC+08:00）；已部署Pass，原合同与近邻通过。

compile-on-one-rank 要求生成代码可在运行时解析设备。post-grad 消除 FX 中不可 lowering 的 device-valued node，同时 wrapper/kernel metadata 不能把编译时设备编号固定进去。

```python
# torch/_inductor/fx_passes/post_grad.py:145（语义节选）
nodes = graph.find_nodes(op='call_function', target=torch.ops.coor.current_device.default)
# 将这些设备节点再特化并删除，后续生成代码仍须遵守CooR设备无关合同。

# NPU 已安装修复：triton_experimental/device.py 与 codegen/triton.py
def current_device_idx_expr(self):
    return 'torch.npu.current_device()'  # 返回表达式，不是现在调用得到整数
# CooR下 DeviceProperties.create(torch.device('npu')) -> index=None
```

GPU原单设备codegen合同通过。NPU修复前已发生目标改图，但wrapper缺接口而未执行数值；第一版候选数值通过，却仍固化metadata index。最终两文件修复经授权部署Pass，安装态原合同1/1、数值与全部codegen检查通过。
另外4项新进程近邻通过：普通CooR=False的FP32/FP16编译、cpp/fx wrapper拒绝、物理NPU5/6在两配置下代码一致。原件分别归档，未回写候选结果。
尚未验证多rank通信/跨卡kernel handle复用，不能把单/双设备codegen成功扩展成完整分布式支持。
它是正确性必需的lowering前改写，没有合法OFF；功能已通过，性能仍免测，不计收益。

[完整根因、两级错误栈及修复前后FX/IR/output_code](../../../issues/REF-respecialize-current-device-native/根因分析.md)；[安装态与候选状态](npu_blocker_review.json)。

[安装态验证步骤、原例/近邻日志和备份](../../../issues/REF-respecialize-current-device-native/修复验证报告.md)是当前学习入口。正式结论NEWLY_SUPPORTED，源码局部分支提交6cff38b16；未社区合入。
