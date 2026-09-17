# 文档导航

本仓库将离线 Toolkit 与运行时插件作为两个独立 Python Distribution。

| 文档 | 内容 |
|---|---|
| [架构边界](architecture.md) | Toolkit、Runtime Extension 和 KV 能力边界 |
| [现有量化架构](current-vllm-ascend-quant-architecture.md) | 冻结宿主源码、模块输入输出和调用链 |
| [运行时插件设计](runtime-plugin-design.md) | vLLM 原生入口、输入输出、启动和回退 |
| [Artifact Contract](../contracts/README.md) | 格式、软件、张量和 fail-closed 规则 |
| [验收矩阵](acceptance-matrix.md) | 正确性、精度、性能、HBM 和回退门槛 |
| [W8A8 证据](w8a8-evidence.md) | NPU E2E、PPL 和 ShareGPT 记录 |
| [打包发布指南](runtime-extension-packaging-and-release.zh-CN.md) | wheel/sdist、干净安装和 PyPI 发布 |
| [发布清单](release-checklist.md) | 每个不可覆盖版本的发布门禁 |
| [0.4.1a2 NPU 验证](validation-w8a8-v0.4.1a2-20260916.md) | 原生 vLLM 插件路径端到端记录 |
| [v1 适配状态](validation-w8a8-v1-readiness-20260916.md) | 冻结 ref、Ascend 来源链、静态验证与待完成 NPU 门禁 |
| [冻结 v1 NPU E2E 证据](evidence/w8a8-frozen-v1-npu-e2e-20260917.md) | 冻结宿主正确性验证、fail-closed 项与仍未完成的门禁 |

Qwen2.5-14B W8A8 已在旧冻结宿主上完成加载和确定性推理验证。vLLM-HUST `v1`
组合也已通过正确性验证（见上面的冻结 v1 NPU E2E 证据），但仍缺 wheel 哈希级别
的 NPU 复验与 Manager 生命周期门禁，不能继承旧结果作为发布依据。

历史验证文档保留当时版本的真实结构，不回写。`0.4.1a2` 记录的"移除实验性
Manager Bundle"只描述当时状态；当前插件接口是 Manifest 0.2 双入口，见
[打包发布指南](runtime-extension-packaging-and-release.zh-CN.md)。
