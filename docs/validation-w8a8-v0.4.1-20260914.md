# W8A8 Runtime Extension 0.4.1 验证记录（2026-09-14）

本记录覆盖 Runtime Extension 0.4.1 与 artifact contract 1.1 的真实
Ascend NPU 端到端复验。它不代表 Extension Manager 正式激活联调，也不构成
BF16/W8A8 matched benchmark。

## 验证对象

- Git commit：`fde6dd30696950e3c6ef3b94e1cacf57023ea3b1`
- Toolkit：`ascend-quant-toolkit==0.4.1`
- Runtime Extension：`vllm-ascend-quant-ext==0.4.1`
- 模型：Qwen2.5-14B-Instruct W8A8
- Contract：`1.1.0`
- Artifact ID：`qwen2:W8A8:289309eb1322e19c`
- Artifact files digest：
  `289309eb1322e19c1008b1645eaa83bb30c83ff7ca98a93693978515e6f3f0c6`
- Runtime quant type：`ASCEND_QUANT_W8A8`
- NPU：Ascend 910B3，单卡，TP=1，device 7
- CANN：8.5.0
- torch-npu：2.9.0
- 实际导入的 vLLM：`0.17.2.post2.dev1080+g6cff12512.d20260603`
- 实际 vLLM-Ascend 源码：`0.1.dev2798+g203a33e67.d20260910`

## 结果

| 检查项 | 结果 | 说明 |
|---|---|---|
| Contract 1.1 export | PASS | config、description、index、4 个权重分片均绑定 size/SHA-256 |
| 完整 artifact admission | PASS | 2595 个张量通过哈希、SafeTensors、dtype、shape、index 与软件范围检查 |
| 插件入口点发现 | PASS | runtime 与 Extension Bundle 两个 entry point 均来自 0.4.1 |
| 插件启用启动 | PASS | `ASCEND_QUANT_W8A8` 在 NPU 7 完成加载 |
| 确定性推理 | PASS | chat completion 对约束提示精确返回 `OK` |
| 插件禁用 fail closed | PASS | 同一产物以 `ASCEND_QUANT_W8A8/linear` 不受支持为根因拒绝启动 |
| 原生路径恢复 | PASS | 未迁移的 `W8A8_MIX` 产物在插件禁用时启动并精确返回 `OK` |
| 安装/卸载模型不变性 | PASS | 模型文件 size、mtime、inode 与关键元数据 SHA-256 前后完全一致 |
| 卸载入口点清理 | PASS | 两个 extension entry point 均消失；重装后恢复 |
| Evidence 1.1 | PASS | `w8a8-npu-e2e-v041-20260914` 通过闭合 schema 与 contract 绑定校验 |

模型加载权重为 15.2713 GB。NPU 7 空闲 HBM 为 3414 MiB，加载后采样为
60830 MiB。本次只记录加载后采样值，未持续采集时间序列，因此不得将其作为
matched benchmark 的严格峰值 HBM。

## 证据位置

验证机上的原始证据保存在：

```text
/data/jxd/ascend-quant-validation/20260914-v041/logs/
/data/jxd/models/Qwen2.5-14B-Instruct-w8a8-runtime-v041/evidence/
```

最终 contract 声明：

```text
evidence.level = npu_e2e
evidence.verified_profiles = ["W8A8"]
evidence.results = ["evidence/w8a8-npu-e2e-v041.json"]
```

历史 v0.4.0 模型未被覆盖；其 contract SHA-256 在建立验证副本前后保持为
`acc030c5612aef54a5f1269f22fe1ee6c83351832a8661b333e65011ab82aa32`。

## 版本元数据加固与清理

初次验证时，当前 Conda 环境残留多个 vLLM-HUST 与 vLLM-Ascend-HUST dist-info。
`importlib.metadata.version` 按目录顺序报告 dev27/dev2790，而实际导入源码分别为
dev1080/dev2798。实际版本仍在 contract 允许范围，本次 E2E 未受影响。

本次同时完成了版本探测加固和环境清理：

- validator 现在枚举并规范化匹配发行包元数据；
- 同一发行包存在多个不同版本时直接 fail closed；
- 从当前源码重新生成两个发行包的 editable 元数据；
- 旧 dist-info、editable pth 和源码 egg-info 已移动到可恢复的
  `/data/jxd/ascend-quant-validation/20260914-v041/metadata-backup/`；
- 清理后 validator 与实际导入模块一致，分别报告
  `vllm-hust ...d20260914` 和 `vllm-ascend-hust ...d20260914`；
- 清理后的 contract admission、全量测试和真实 NPU 确定性推理均通过。

## 剩余门槛

仍未完成的联合/发布门槛：

1. Extension Manager 正式 discover/check/plan/render/enable/run；
2. Manager disable/uninstall 生命周期联调；
3. 相同协议下 BF16/W8A8 精度、吞吐、TTFT 和持续峰值 HBM 对照；
4. 签名 tag、GitHub Release、PyPI 发布及无缓存安装验证。
