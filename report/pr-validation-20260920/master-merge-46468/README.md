# PR 46468 合并主线后的复测

2026-09-21 10:08:55（北京时间）完成。17 项通过，0 失败，0 跳过。

测试提交为 `2d4e07aa7d44d6b2ce10e49821c818743b78a477`；推送提交为 `4b7059b2dc8db2e931e0122a115d5f075d33f33d`。两者只调整了 Git 提交身份，代码树完全一致，映射见 [commit-identity-mapping.json](commit-identity-mapping.json)。

已合并 master `28a9bc52a5bf45a26b2700943cfaa13ceb8df2fc`，同时保留主线 softmax 逻辑及 addcdiv 修复。使用完整编译器 Python 副本，原生库沿用安装环境，未重建 wheel。

[结果清单](results.json) · [环境和完整编译器文件哈希](regression-identity.json)
