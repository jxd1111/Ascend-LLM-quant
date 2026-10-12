# W8A8 Runtime Extension 打包与发布

## 发布边界

发布物 `vllm-ascend-quant-ext` 只包含运行时 contract 校验、两个插件入口和
W8A8 scheme 注册。离线 Toolkit、数据集和模型权重不进入该 wheel；Extension
Manager 的 Manifest 0.2 是随包发布的静态描述，不含任何校准或转换代码。

```text
vllm.general_plugins/vllm_ascend_quant                          运行时激活入口
vllm_hust.extension_bundles/org.vllm-hust.ascend-quant-runtime   Manifest 0.2 定位符
```

第二个入口解析到 `vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json`；
注册名必须与 manifest 的 `extension_id` 完全一致，且不得占用新的 `vllm.*`
命名空间。

## 本地门禁

```bash
python -m pytest -q runtime-extension/tests
python -m ruff check runtime-extension/src runtime-extension/tests
python -m build --wheel --sdist --no-isolation \
  --outdir runtime-extension/dist runtime-extension
python -m twine check runtime-extension/dist/*
python runtime-extension/tools/verify_release.py \
  --wheel runtime-extension/dist/vllm_ascend_quant_ext-*.whl \
  --sdist runtime-extension/dist/vllm_ascend_quant_ext-*.tar.gz
```

`verify_release.py` 会在隔离 venv 中验证安装、两个入口的发现、Manifest 0.2
可从 distribution metadata 定位、`extension_id` 与注册名一致、卸载清理及可选
的模型目录不变性。

`vllm-hust-ext` 尚未发布，但可按 README 从源码安装。本轮已用 Manager
`cf1ea71`（`0.2.0.dev0`）在隔离客户端环境（`VLLM_HUST_EXT_CONFIG` 指向 `/tmp`）
跑通 `extension list/inspect/validate/status/check/plan/render/env/configure/
enable/run --dry-run/disable/forget`，并据此修正 manifest 的三处不兼容（见
ADR 0002 与冻结 v1 NPU 证据记录）。Manager 正式发布后需要用发布版本复跑一次。

## 正式发布

仓库使用 GitHub Actions + PyPI Trusted Publisher，不手工保存或上传 token。
版本 `0.5.0` 对应 Tag：

```text
runtime-v0.5.0
```

发布顺序：

1. 将版本修改和证据通过 PR 合入 `main`；
2. 确认 CI 全部通过；
3. 先在 TestPyPI 预演：以 `workflow_dispatch` 运行
   `rehearse-runtime-testpypi.yml`，再从未发布版本安装该精确版本并重复
   discovery/validate/enable/dry-run/disable/forget/uninstall 门禁；
4. 创建指向该合入提交的 GitHub Release/Tag；
5. `publish-runtime.yml` 校验 Tag、测试、构建 wheel/sdist；
6. OIDC 发布到 PyPI；
7. 从 PyPI `--no-cache-dir` 安装精确版本，并完成 wheel 哈希级别的 NPU E2E。

不要覆盖已存在版本。若发布内容错误，提升版本号后重新发布，必要时 yank
错误版本。

## Toolkit 打包与发布

离线 Toolkit 是与运行时插件并列的独立 Distribution，使用独立的 Tag 命名空间
`toolkit-v<version>`；版本来自 `ascend_quant_toolkit.__version__`，
`pyproject.toml` 使用动态版本，不再重复声明：

```text
toolkit-v0.4.2
```

本地门禁：

```bash
python -m pytest -q
python -m ruff check .
python -m build --wheel --sdist --no-isolation --outdir dist .
python -m twine check dist/*
```

发布流程与运行时一致：PR 合入 `main` → CI 通过 → 以 `workflow_dispatch` 运行
`rehearse-toolkit-testpypi.yml`（构建、校验 wheel/sdist 内容、上传 TestPyPI，
再从 TestPyPI 安装该精确版本并运行 `ascend-quant-toolkit list-recipes`）→ 创建指向该
合入提交的 `toolkit-v<version>` Tag → `publish-toolkit.yml` 校验 Tag、跑单测与
ruff、构建、校验内容并在隔离 venv 中以 `--no-deps` 安装后运行 CLI → OIDC 发布
到 PyPI。

两个项目的 PyPI Trusted Publisher 需要分别登记。Owner `jxd1111`、仓库
`Ascend-LLM-quant`，首次发布前在 PyPI/TestPyPI 以 Pending Publisher 形式登记：

| 项目 | Workflow | Environment |
|---|---|---|
| `vllm-ascend-quant-ext` | `publish-runtime.yml` | `pypi` |
| `vllm-ascend-quant-ext`（预演） | `rehearse-runtime-testpypi.yml` | `testpypi` |
| `ascend-quant-toolkit` | `publish-toolkit.yml` | `pypi` |
| `ascend-quant-toolkit`（预演） | `rehearse-toolkit-testpypi.yml` | `testpypi` |

## 启停验证

```bash
export VLLM_ASCEND_QUANT_EXT_ENABLE=1
export VLLM_ASCEND_QUANT_EXT_ARTIFACT=/path/to/w8a8-model
vllm-hust serve /path/to/w8a8-model --host 0.0.0.0 --port 18000
```

禁用需要停止旧进程并在新进程中清除三个环境变量。当前不承诺热卸载。
