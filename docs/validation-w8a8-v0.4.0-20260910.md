# W8A8 Runtime Extension 0.4.0 验证记录（2026-09-10）

本记录覆盖量化团队能够独立完成的发布前验证。它不是 Extension Manager
正式联调结果，也不是 BF16/W8A8 matched benchmark。

## 验证对象

- Runtime Extension：`vllm-ascend-quant-ext==0.4.0`
- 模型：Qwen2.5-14B-Instruct W8A8
- Artifact ID：`qwen2:W8A8:f54cc7465f390ad8`
- Runtime quant type：`ASCEND_QUANT_W8A8`
- NPU：Ascend 910B3，单卡，TP=1，device 7
- CANN：8.5.0
- torch-npu：2.9.0
- vLLM：0.17.2.post2.dev27+g54ee69103
- vLLM-Ascend：0.1.dev2790+g95c956c6a

## 结果

| 检查项 | 结果 | 说明 |
|---|---|---|
| Toolkit 单测 | PASS | 12 passed |
| Runtime Extension 单测 | PASS | 27 passed |
| Ruff | PASS | Toolkit 与 Runtime 源码均通过 |
| wheel/sdist 元数据 | PASS | `twine check` 全部通过 |
| 隔离安装与入口点发现 | PASS | runtime 与 Bundle 两个 entry point 均可发现 |
| 隔离卸载 | PASS | 卸载后两个 entry point 均消失 |
| 安装/卸载模型不变性 | PASS | 契约、描述、配置哈希及权重 size/mtime 未变化 |
| 旧私有 runtime type | REJECTED | 旧 `JXD_W8A8_PDMIX` 契约被 fail closed |
| 完整 Artifact admission | PASS | 2595 个张量通过格式、shape、hash 与软件版本检查 |
| 插件启用启动 | PASS | `ASCEND_QUANT_W8A8` 模型启动并完成 chat completion |
| 插件禁用对照 | PASS | 同一产物因不支持 `ASCEND_QUANT_W8A8/linear` 而拒绝启动 |
| 原生路径恢复 | PASS | 插件禁用时，原生 `W8A8_MIX` 模型启动并返回 `OK` |

启用态模型加载权重为 15.2713 GB；采样时 NPU 7 总 HBM 使用约 60829 MiB，
空闲基线约 3414 MiB。本次数值只证明单点 loaded HBM，未持续采集峰值，
因此不能填充发布验收中的 peak HBM 指标。

## 关键命令

```bash
vllm-ascend-quant-ext check \
  --model /path/to/Qwen2.5-14B-Instruct-w8a8-runtime-v040

export VLLM_ASCEND_QUANT_EXT_ENABLE=1
export VLLM_ASCEND_QUANT_EXT_ARTIFACT=/path/to/Qwen2.5-14B-Instruct-w8a8-runtime-v040
ASCEND_RT_VISIBLE_DEVICES=7 vllm-hust serve \
  /path/to/Qwen2.5-14B-Instruct-w8a8-runtime-v040 \
  --served-model-name ascend-w8a8 \
  --host 127.0.0.1 \
  --port 18002 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --enforce-eager
```

禁用对照使用相同模型与启动参数，只取消两个
`VLLM_ASCEND_QUANT_EXT_*` 环境变量。原生恢复测试使用未迁移的
`W8A8_MIX` 产物。

## 证据范围与剩余门槛

原始日志保存在验证机器的独立结果目录中，模型契约的 evidence level 已更新为
`npu_e2e`。对外发布前仍需将脱敏日志作为可访问的 CI artifact 或 Release asset
上传，并完成：

1. Extension Manager 正式 discover/check/plan/render/enable/run；
2. Manager disable/uninstall 生命周期测试；
3. 相同数据、参数、版本和启动配置下的 BF16/W8A8 精度、吞吐、TTFT 与峰值 HBM；
4. 签名 tag、GitHub Release、PyPI 发布及无缓存安装验证。

