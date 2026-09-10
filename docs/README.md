# 文档导航

本目录记录 Ascend 模型权重与激活量化的现状分析、运行时插件设计、
Extension Manager 对接边界和验证证据。离线 Toolkit 与 Runtime Extension
是两个独立的 Python Distribution；Adaptive Quantized KV 不在本项目范围内。

## 建议阅读顺序

| 顺序 | 文档 | 用途 |
|---:|---|---|
| 1 | [中文设计概览](design-overview.zh-CN.md) | 快速理解目标、边界、W8A8 和当前状态 |
| 2 | [当前量化架构](current-vllm-ascend-quant-architecture.md) | 了解 vLLM-HUST/vLLM-Ascend 的代码流程、接口和算法 |
| 3 | [Runtime Extension 设计](runtime-plugin-design.md) | 查看插件输入、输出、注册方式和启动模式 |
| 4 | [Extension Manager 对接](extension-manager-handoff.md) | 查看 discover/check/plan/render 的交接契约 |
| 5 | [Artifact Contract](../contracts/README.md) | 查看产物格式、版本和 fail-closed 规则 |
| 6 | [验收矩阵](acceptance-matrix.md) | 查看正确性、精度、性能、HBM 和回滚门槛 |
| 7 | [W8A8 证据](w8a8-evidence.md) | 查看当前可追溯的 NPU E2E、PPL 和 ShareGPT 记录 |
| 8 | [后续路线图](runtime-plugin-roadmap.md) | 查看依赖项、优先级和开放问题 |
| 9 | [打包发布指南](runtime-extension-packaging-and-release.zh-CN.md) | 构建、隔离安装、发现、卸载与发布流程 |

## 架构决策

- [ADR 0001：Runtime Quantization Plugin 边界](adr/0001-runtime-plugin-boundary.md)

后续经 vLLM-Ascend 或 Extension Manager 团队确认的接口决策，应新增 ADR，
而不是只保留在 Issue、聊天记录或实现代码中。

## 当前完成度

| 能力 | 状态 |
|---|---|
| W8A8 离线 Recipe | 已实现 |
| 版本化 Artifact Contract | 已实现 |
| Runtime Extension 独立 wheel | 已实现 |
| 默认禁用及 fail-closed admission | 已实现 |
| Qwen2.5-14B Ascend NPU E2E | 已验证 |
| Extension Manager 正式 materializer | 待框架团队确认 |
| vLLM-Ascend 公共 Scheme SPI | 待框架团队确认 |
| BF16/W8A8 matched benchmark 与 HBM | 待补齐 |
| W4A4/W4A8 硬件验证 | 未完成 |

## 评审入口

- [Design Issue #1](https://github.com/jxd1111/Ascend-LLM-quant/issues/1)
- [Release Checklist](release-checklist.md)

状态标记必须以仓库中的可复现证据为准。本地服务器路径不能作为外部评审者
可访问的公开证据；对外发布前应提交脱敏后的命令、环境信息和结构化 Evidence。
