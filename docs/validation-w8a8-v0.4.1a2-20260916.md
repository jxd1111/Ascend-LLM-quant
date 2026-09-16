# W8A8 Runtime Extension 0.4.1a2 验证记录（2026-09-16）

本记录验证移除 Extension Manager Bundle 后，最终本地 wheel 能通过 vLLM
原生 `vllm.general_plugins` 路径在冻结 Ascend 宿主上完成推理。

## 对象

- Runtime Extension：`vllm-ascend-quant-ext==0.4.1a2`
- wheel SHA-256：`a0ce18a45c7b066f0034c8a0f2798701484cbeac43b5ba898c56e32dfc90c1fc`
- sdist SHA-256：`99f9a6cff5f322eaa0e8e521d6035db945cb7515470c1d2b3eccf68ad5163e1b`
- 模型：Qwen2.5-14B-Instruct W8A8，artifact contract `1.1.0`
- vLLM-HUST：`0.17.2.post2.dev1080+g6cff12512.d20260914`
- vLLM-Ascend-HUST：`0.1.dev2798+g203a33e67.d20260914`
- torch / torch-npu：`2.9.0`
- CANN：`8.5.0`
- NPU：Ascend 910B3，device 7，TP=1

## 结果

| 检查 | 结果 |
|---|---|
| Toolkit 单元测试 | PASS（17） |
| Runtime 单元测试 | PASS（38） |
| Ruff | PASS |
| wheel/sdist + Twine | PASS |
| 隔离安装、原生入口发现、卸载 | PASS |
| `vllm_hust.extension_bundles` 不存在 | PASS |
| 全量 artifact/hash/tensor admission | PASS（2595 tensors） |
| 冻结 vLLM/vLLM-Ascend revision admission | PASS |
| NPU 权重加载 | PASS（15.2713 GB） |
| API 启动 | PASS（127.0.0.1:18002） |
| 确定性 chat completion | PASS（精确输出 `OK`） |
| 临时服务停止及 NPU 释放 | PASS |

冻结宿主的 torch-npu preflight 固定超时为 20 秒，而该机器冷启动 import
曾超过该值，因此本次沿用既有验证方法设置
`VLLM_ASCEND_TORCH_PREFLIGHT=0`。真实模型加载、KV cache 初始化及推理随后均在
NPU 7 成功完成；该设置不是对 NPU 执行的替代。

原始日志保存在验证机：

```text
/data/jxd/ascend-quant-validation/20260916-native-vllm-041a2/logs/server.log
/data/jxd/ascend-quant-validation/20260916-native-vllm-041a2/logs/chat.json
```

本记录证明正确性与原生插件接入，不构成 BF16/W8A8 matched 性能结论。
