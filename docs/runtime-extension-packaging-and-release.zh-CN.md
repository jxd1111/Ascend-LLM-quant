# Runtime Extension 打包、发布与安装指南

本文说明 `vllm-ascend-quant-ext` 从版本管理、构建、检查到安装的流程。
它参照 vLLM-HUST Extension Bundle 0.2 的打包方式，但不把离线校准和模型
转换工具带入 vLLM 进程。

## 1. 包与标识

| 项目 | 当前值 |
|---|---|
| Distribution | `vllm-ascend-quant-ext` |
| Python package | `vllm_ascend_quant_ext` |
| Bundle ID | `org.vllm-hust.ascend-quant-runtime` |
| Component ID | `ascend-quant-artifact-validator` |
| Contract | `vllm.ascend.quantized-artifact-loader.v1` |
| Execution planes | `worker`, `device` |
| Lifecycle owner | `vllm` |
| Implementation status | `import_only` |

以上名称暂按当前设计使用。Manager 或 vLLM-Ascend 团队确认正式命名后，必须
同时修改 manifest、entry point、测试和文档，不能只修改其中一处。

## 2. 单一版本源

版本只在下列文件维护：

```text
runtime-extension/src/vllm_ascend_quant_ext/_version.py
```

`pyproject.toml` 从该模块读取版本。发布前还需保证 Bundle manifest 的
`extension_version` 与它一致；CI 会在隔离环境中检查两者。

## 3. Bundle 发现与运行时边界

安装 wheel 后，Manager 通过以下 entry point 发现静态 manifest：

```text
vllm_hust.extension_bundles:
  org.vllm-hust.ascend-quant-runtime = vllm_ascend_quant_ext.manifests
```

manifest 位于：

```text
vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json
```

Bundle activation 保持为空，implementation 为 `import_only`。Manager 可以发现、
检查，但启用时必须拒绝 plan/render；它不会自动设置环境变量、修改模型或注入
未知 vLLM 参数。`vllm.general_plugins/vllm_ascend_quant` 仅用于直接诊断，不是
正式 Manager activation。正式启动依赖 vLLM-Ascend 发布版本化量化产物加载与
算子选择协议。

`ASCEND_QUANT_W8A8` 只作为命名空间别名委托给选定的 vLLM-Ascend W8A8
linear/MoE Scheme。权重加载、参数布局、运行角色选择和 NPU 算子仍由
vLLM-Ascend 维护，扩展包不复制这些实现。

## 4. 本地构建

在仓库根目录执行：

```bash
python -m pip install build pytest packaging
python -m pytest -q
python -m pytest -q runtime-extension/tests

BUILD_DIR="$(mktemp -d /tmp/ascend-quant-build.XXXXXX)"
python -m build --wheel --sdist --no-isolation \
  --outdir "$BUILD_DIR" runtime-extension
```

构建结果必须同时包含 wheel 和 sdist。

## 5. 发布前验证

```bash
python runtime-extension/tools/verify_release.py \
  --wheel "$BUILD_DIR"/vllm_ascend_quant_ext-*.whl \
  --sdist "$BUILD_DIR"/vllm_ascend_quant_ext-*.tar.gz
```

验证器会检查：

- wheel/sdist 包含 manifest、typed carrier、W8A8 scheme 和版本文件；
- 两类 entry point 均可被发现；
- Distribution 版本与 manifest 版本一致；
- 在全新虚拟环境中可以安装和卸载；
- 卸载后 entry point 消失；
- 传入 `--model /path/to/model` 时，检查期间模型文件不被修改。

正式发布前还必须执行真实 Ascend 环境测试：

```bash
vllm-ascend-quant-ext check --model /path/to/quantized-model

python runtime-extension/tools/verify_release.py \
  --wheel "$BUILD_DIR"/vllm_ascend_quant_ext-*.whl \
  --sdist "$BUILD_DIR"/vllm_ascend_quant_ext-*.tar.gz \
  --model /path/to/quantized-model
```

第一条命令执行完整 artifact/software/tensor 准入；第二条命令中的 `--model`
只负责证明安装和卸载过程没有修改模型产物，不能代替前者。

## 6. 安装与发现检查

```bash
python -m pip install --no-deps \
  "$BUILD_DIR"/vllm_ascend_quant_ext-*.whl

python - <<'PY'
from importlib.metadata import entry_points

for ep in entry_points(group="vllm_hust.extension_bundles"):
    if ep.name == "org.vllm-hust.ascend-quant-runtime":
        print(ep.name, "->", ep.value)
PY
```

预期输出：

```text
org.vllm-hust.ascend-quant-runtime -> vllm_ascend_quant_ext.manifests
```

仅进行旧路径诊断时，必须显式提供开关与量化模型路径。生产接入不应依赖此
环境变量协议：

```bash
export VLLM_ASCEND_QUANT_EXT_ENABLE=1
export VLLM_ASCEND_QUANT_EXT_ARTIFACT=/path/to/quantized-model
```

## 7. Manager 联调门槛

当前仓库能完成 Bundle 的构建、静态发现、隔离安装和卸载验证，但不能替代
Manager/Host 的正式联调。公开发布前需要 Manager 团队确认：

1. manifest 0.2 的严格 schema 与 Bundle identity；
2. `vllm.ascend.quantized-artifact-loader.v1`；
3. `vllm.ascend.quantized-operator-selection.v1`；
4. Host 如何把模型路径及不可变 artifact identity 传给 worker；
5. typed carrier 的实例化、错误传播、disable/uninstall 生命周期。

任何一项未匹配都必须拒绝启用，不允许自动回退到未声明的量化算子。

## 8. 发布顺序

1. 更新 `_version.py`、CHANGELOG 和 manifest 版本；
2. 运行单测、wheel/sdist 构建及隔离安装验证；
3. 在匹配软件栈的 Ascend 机器完成 W8A8 正确性与性能回归；
4. 提交 Manager discover/check 与 import-only 拒绝启用证据；
5. 创建 Git tag 和 GitHub Release；
6. Manager 接口稳定后再发布 PyPI alpha。

W4A4/W4A8 可保留契约与测试骨架，但当前发布验收只以已验证的 W8A8 为准。

卸载 wheel 只移除 entry point，不改模型元数据。若模型 active metadata 为
`ASCEND_QUANT_W8A8`，需要在离线 Toolkit 中显式执行 `restore-modelslim` 后才能
回到原生 `W8A8_MIX` 路径；恢复前后都应记录模型描述文件哈希。
