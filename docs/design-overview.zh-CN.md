# Ascend W8A8 量化设计概览

## 两个独立制品

本仓库不是把离线量化工具整体塞入 vLLM，而是发布两个独立制品：

- `ascend-quant-toolkit`：离线校准、ModelSlim 转换、契约生成和评测证据；
- `vllm-ascend-quant-ext`：只读的 W8A8 运行时插件。

```text
原始模型 + 校准数据
  -> Offline Toolkit / ModelSlim
  -> W8A8 权重 + quant_model_description.json
  -> ascend_quant_artifact.json
  -> vLLM 原生插件校验并注册 ASCEND_QUANT_W8A8
  -> vLLM-Ascend 加载权重并执行 NPU 算子
```

## 插件接口

插件通过标准 Python distribution metadata 注册：

```text
vllm.general_plugins/vllm_ascend_quant
```

安装只产生 `installed + discoverable` 状态，不会注册 scheme、访问设备或修改
模型。只有新启动的 vLLM 进程同时设置以下变量时才启用：

```bash
export VLLM_ASCEND_QUANT_EXT_ENABLE=1
export VLLM_ASCEND_QUANT_EXT_ARTIFACT=/path/to/w8a8-model
```

扩展不提供 Extension Manager manifest，不修改 vLLM 源码，也不 monkey patch。
进程生命周期由 vLLM 管理；当前不承诺热卸载。

冻结宿主会自动发现已安装的 general plugin。不要把 `VLLM_PLUGINS` 仅设置为
量化插件名，因为它会同时过滤 platform/general 入口并破坏 Ascend 平台加载。
已有全局 allowlist 的部署必须保留冻结宿主所需的全部入口。

## 冻结兼容基线

```text
vLLM-HUST        6cff125127bac512488dc90a9812dcafddb65298
vLLM-Ascend-HUST 203a33e677ac6728108473e749069244bea00373
torch            2.9.0
torch-npu        2.9.0
CANN             >=8.5,<8.6
```

插件同时检查发行包版本和版本字符串中的 Git revision。相同版本区间内的其他
提交不会被自动视为兼容。

## 输入与输出

输入是只读模型目录，其中 contract 1.1 绑定：

- `config.json`；
- `quant_model_description.json`；
- safetensors 索引及所有权重分片；
- dtype、shape、packing、scale、zero-point 和所需算子；
- 软件兼容声明与评测证据。

成功输出是两个幂等 registry 映射：

```text
ASCEND_QUANT_W8A8/linear
ASCEND_QUANT_W8A8/moe
```

具体 W8A8 算法、权重加载、布局转换、并行通信和 NPU 算子仍由冻结的
vLLM-Ascend-HUST 实现。插件只负责 admission 和命名空间别名。

## 失败与回退

未知字段、错误哈希、缺失张量、不支持的 shape、软件/提交不匹配、算子缺失或
registry 冲突都会中止启动，不静默回退到 BF16。

禁用时停止旧进程、清除三个环境变量并启动新进程。卸载插件不修改量化模型；
若要让 `ASCEND_QUANT_W8A8` 产物回到原生 `W8A8_MIX` 路径，必须显式使用离线
Toolkit 恢复保存的 ModelSlim 描述。

## 不在范围内

- KV cache 量化、压缩、驱逐或调度；
- 数据集下载和在线校准；
- vLLM/vLLM-Ascend 平台实现；
- 未经真实 NPU 验证的 W4A4/W4A8 性能声明。
