# Ascend W8A8 量化运行时插件设计概览

## 1. 项目目标

本项目将 Ascend 大模型量化拆分为离线开发和在线推理两个生命周期：

- **Ascend Quant Toolkit** 生成量化模型、来源信息和版本化产物契约；
- **vLLM Ascend Quant Extension** 在推理进程中只读检查产物并注册量化 Scheme；
- **vLLM-HUST Extension Manager** 发现、检查、规划和启停插件；
- **vLLM-Ascend** 继续拥有 NPU Platform、并行策略、权重加载和算子生命周期。

目标不是把整个 ModelSlim/离线量化仓库装入 vLLM，而是提供一个依赖清晰、
默认不生效、可禁用和可卸载的运行时扩展。

```mermaid
flowchart LR
    A[原始模型与校准集] --> B[Ascend Quant Toolkit]
    B --> C[W8A8 量化产物]
    C --> D[Extension Manager]
    D -->|check / plan / render| E[Runtime Extension]
    E -->|注册 namespaced Scheme| F[vLLM-Ascend]
    F --> G[torch-npu / CANN]
```

## 2. 当前 vLLM-Ascend 量化流程

现有执行链路可概括为：

```text
读取 quant_model_description.json
  → 识别每层 quant type
  → registry 选择 Linear/MoE Scheme
  → 创建参数并加载 safetensors
  → post-load transpose/flatten/NZ 转换
  → 激活量化和 NPU quant matmul
  → 浮点输出
```

详细的源代码位置、方法输入输出和 tensor 形状见
[当前 vLLM-HUST/vLLM-Ascend 量化架构](current-vllm-ascend-quant-architecture.md)。

## 3. Toolkit 与 Runtime Extension 边界

| 组件 | 负责 | 不负责 |
|---|---|---|
| Toolkit | 校准、SmoothQuant/GPTQ、转换、产物和评测 | vLLM worker 生命周期 |
| Runtime Extension | 产物 admission、量化类型映射、命名空间 Scheme 适配 | 校准、数据集、模型转换和 NPU 算子实现 |
| Extension Manager | discover/check/plan/render/start/disable/uninstall | 修改量化模型 |
| vLLM-Ascend | Platform、TP/EP、公共 Scheme SPI、CANN/torch-npu 算子 | 离线校准 |

模型权重/激活量化与 Adaptive Quantized KV 是两条独立能力线，不共享 manifest、
生命周期或兼容声明。

## 4. W8A8 量化方案

当前验证方案对权重和激活都使用 INT8。权重采用离线 per-channel 量化；激活
执行路径由 vLLM-Ascend 根据运行角色选择。具体 ModelSlim 参数属于 Toolkit
内部 recipe，不作为插件名称或公共运行时类型的一部分。

当前 vLLM-Ascend 的运行时选择为：

```text
execution_role=kv_consumer            → static W8A8
execution_role=standalone/kv_producer → dynamic W8A8
```

`execution_role` 只选择 W8A8 执行路径，也不启用 KV-cache 量化。其稳定 Host
API 仍需 vLLM-Ascend 团队确认。

## 5. 插件输入与输出

Manager 配置输入：

```json
{
  "enabled": true,
  "model": "/absolute/path/to/quantized/model"
}
```

量化模型输入至少包含 `config.json`、`quant_model_description.json`、
`ascend_quant_artifact.json` 和声明的 safetensors shards。产物契约冻结模型
身份、tensor shape/dtype、packing/layout、scale/zero-point、软件版本、
Scheme Provider 和 required operators。

插件输出为：

- admission report；
- immutable artifact identity；
- runtime plan；
- 环境变量和 vLLM plugin 增删项；
- namespaced Scheme 注册；
- 结构化且终止启动的错误。

未知字段、不兼容版本、错误 tensor、缺失算子或不支持的 shape 必须 fail
closed，不允许静默回退 BF16。完整接口见
[Runtime Extension 详细设计](runtime-plugin-design.md)。

## 6. 启动和回滚

生产目标流程：

```text
install → discover → check → plan → render → start
        → worker re-admission → Scheme registration → inference
        → disable/uninstall → rollback
```

安装 wheel 默认不生效。Manager 和 Runtime Extension 始终只读模型目录；worker
启动时再次校验 artifact identity，避免 plan/start 之间产物变化。禁用或卸载
只移除插件选择和环境状态，模型文件保持不变。若 active metadata 使用
`ASCEND_QUANT_W8A8`，卸载前后都必须显式运行 Toolkit 的 `restore-modelslim` 才能
恢复原生 `W8A8_MIX` 路径；恢复命令会校验 active metadata，发现后续修改则拒绝
覆盖。

在 Manager 正式接入前，扩展自身的 `check/plan/render` 只属于诊断模式。
详细交接方式见 [Extension Manager 对接契约](extension-manager-handoff.md)。

## 7. 当前验证状态

已经完成：

- Qwen2.5-14B W8A8 离线量化；
- 版本化 Artifact Contract 和严格 validator；
- 独立 Runtime Extension wheel；
- 默认禁用、只读 admission 和 namespaced Scheme；
- 真实 Ascend NPU 模型加载、健康检查和 Chat Completion；
- Python 3.10/3.11 CI、wheel discovery 和卸载检查。

尚未完成：

- Extension Manager 与 vLLM-Ascend 正式支持量化产物加载和算子选择协议；
- vLLM-Ascend 公共、稳定的外部 Scheme SPI；
- standalone/producer/consumer 的正式接口和完整测试；
- 匹配协议下的 BF16/W8A8 精度、性能和 HBM 对照；
- W4A4/W4A8 的真实硬件验证。

当前 `0.2-experimental` Bundle 原型采用以下暂定标识：

```text
Bundle ID:          org.vllm-hust.ascend-quant-runtime
Component ID:       ascend-quant-artifact-validator
完整 Component ID: org.vllm-hust.ascend-quant-runtime/ascend-quant-artifact-validator
Contract:           vllm.ascend.quantized-artifact-loader.v1
Execution planes:   worker, device
Status:             import_only
```

这些标识用于先完成与 BidKV 一致的打包和发现结构，正式发布前仍需与
Extension Manager 团队同步。

因此当前状态应表述为：

```text
Extension-side integration ready;
Extension Manager and public Host API confirmation pending.
```

## 8. 需要框架团队确认

1. 是否接受 Toolkit 与 Runtime Extension 拆分；
2. 是否发布量化产物加载与算子选择 Host 协议；
3. Extension Manager 的 manifest 和 typed Provider/Materializer API；
4. vLLM-Ascend 的公共 namespaced Scheme 注册接口；
5. parameter specification 和 post-load layout API；
6. W8A8 `execution_role` 的稳定来源；
7. 缺失 optimized kernel 时的 fail-closed 策略。

评审讨论集中在
[Issue #1](https://github.com/jxd1111/Ascend-LLM-quant/issues/1)。接口一旦确认，
应写入新的 ADR，再进入实现和发布阶段。

## 9. 继续阅读

- [完整文档导航](README.md)
- [Artifact Contract](../contracts/README.md)
- [验收矩阵](acceptance-matrix.md)
- [W8A8 可追溯证据](w8a8-evidence.md)
- [Runtime Plugin Roadmap](runtime-plugin-roadmap.md)
