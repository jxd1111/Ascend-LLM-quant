# W8A8 v1 适配状态（2026-09-16）

## 冻结输入

- vLLM-HUST ref：`v1`
- vLLM-HUST commit：`f18cf803c5f63625e2c71253ddaf8b0bad0bad1a`
- vLLM-HUST 第一父提交：`a67f6a5dda6e6b81eb42d0ef82c7fca864ca969f`
- 合入的上游 vLLM commit：`bfb443a6b6f670e68e211112a141d089f1cf956f`
- vLLM-Ascend-HUST snapshot：`74f0c0a272376412b51e1c1864803d5f3a0f1b5f`
- 该 Ascend snapshot 记录的 verified core：`a67f6a5dda6e6b81eb42d0ef82c7fca864ca969f`

`74f0c0a272` 因而直接验证了 `v1` 的第一父 core，但没有记录对最终
`f18cf803c5` merge commit 的 NPU E2E。仓库必须保留这一差异，不能把源码
邻接关系写成已经完成的端到端兼容认证。

## 已完成的源码验证

- `v1` 提供标准 `vllm.general_plugins` entry-point loader；
- general plugin 回调可能在多个进程加载，入口必须可重入；
- Ascend snapshot 提供 `register_scheme`、`get_scheme_class` 和公开的
  `AscendW8A8PDMixLinearMethod`；
- 该 snapshot 不提供 PDMix fused-MoE class，因此 `0.4.1a3` 仅注册
  `ASCEND_QUANT_W8A8/linear`，MoE 请求必须 fail closed；
- 当前 admission 仅接受已验证范围内的 `qwen2/Qwen2ForCausalLM`，其他
  模型家族在新增独立证据前 fail closed；
- 对应构建依赖是 torch `2.13.0`、torch-npu `2.13.0rc1` 和 CANN
  `9.1.x`。

## 发布前仍需完成

1. 在干净环境安装由 `v1` 和 `74f0c0a272` 构建的宿主发行包；
2. 运行 `pip check` 并记录所有实际发行包版本；
3. 从 wheel 安装 `vllm-ascend-quant-ext==0.4.1a3`；
4. 验证 entry point 发现、默认关闭和错误 revision fail-closed；
5. 使用真实 W8A8 模型完成 NPU 加载和确定性推理；
6. 禁用/卸载插件后验证原生路径恢复且模型文件哈希不变；
7. 将原始日志和精确硬件/软件信息写入新的证据记录。

完成以上门禁前，`0.4.1a3` 只能标记为 v1 source-compatible candidate，
不能标记为 v1 NPU-verified release。
