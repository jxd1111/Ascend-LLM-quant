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

当前已验证 Qwen2.5-14B W8A8 在冻结 Ascend 宿主上的加载和确定性推理。
BF16/W8A8 matched benchmark 与持续 HBM 采样仍应作为独立证据补齐。

历史验证文档保留当时版本的真实结构；其中出现的实验性 Manager Bundle
只代表旧版本，不是当前插件接口。
