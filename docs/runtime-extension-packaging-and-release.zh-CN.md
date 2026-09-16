# W8A8 Runtime Extension 打包与发布

## 发布边界

发布物 `vllm-ascend-quant-ext` 只包含运行时 contract 校验、vLLM 原生插件
入口和 W8A8 scheme 注册。离线 Toolkit、数据集、模型权重及 Extension Manager
manifest 不进入该 wheel。

唯一运行时入口：

```text
vllm.general_plugins/vllm_ascend_quant
```

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

`verify_release.py` 会在隔离 venv 中验证安装、原生入口发现、Manager 入口
不存在、卸载清理及可选的模型目录不变性。

## 正式发布

仓库使用 GitHub Actions + PyPI Trusted Publisher，不手工保存或上传 token。
版本 `0.4.1a2` 对应 Tag：

```text
runtime-v0.4.1a2
```

发布顺序：

1. 将版本修改和证据通过 PR 合入 `main`；
2. 确认 CI 全部通过；
3. 创建指向该合入提交的 GitHub Release/Tag；
4. `publish-runtime.yml` 校验 Tag、测试、构建 wheel/sdist；
5. OIDC 发布到 PyPI；
6. 从 PyPI `--no-cache-dir` 安装精确版本并进行 NPU E2E。

不要覆盖已存在版本。若发布内容错误，提升版本号后重新发布，必要时 yank
错误版本。

## 启停验证

```bash
export VLLM_ASCEND_QUANT_EXT_ENABLE=1
export VLLM_ASCEND_QUANT_EXT_ARTIFACT=/path/to/w8a8-model
vllm-hust serve /path/to/w8a8-model --host 0.0.0.0 --port 18000
```

禁用需要停止旧进程并在新进程中清除三个环境变量。当前不承诺热卸载。
