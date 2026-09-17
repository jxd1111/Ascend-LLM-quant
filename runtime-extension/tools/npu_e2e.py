#!/usr/bin/env python3
"""Run the frozen-host NPU end-to-end gate against a built wheel.

The gate proves that the *released wheel*, not the source tree, serves the
reference artifact on the frozen host:

1. hash the wheel and every file of the artifact;
2. install the wheel non-editable and prove that the package resolves from
   ``site-packages`` and that both entry-point groups are discoverable;
3. activate it through vLLM's own ``load_general_plugins()`` and capture the
   admission record plus the registered scheme classes;
4. contrast that with the extension disabled and with a rejected legacy
   artifact;
5. serve the reference artifact, send two identical requests, compare them;
6. stop the server, uninstall, and prove that every entry point vanishes;
7. restore the previous install and re-hash the artifact.

Host paths are never hard-coded: pass the frozen host's CANN/ATB environment
scripts with ``--env-script`` and its source trees with ``--host-source``.  The
extension source tree must *not* be passed, so the run can only exercise the
wheel.  ``--dry-run`` prints the plan and touches nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DIST_NAME = "vllm-ascend-quant-ext"
ENABLE_ENV = "VLLM_ASCEND_QUANT_EXT_ENABLE"
ARTIFACT_ENV = "VLLM_ASCEND_QUANT_EXT_ARTIFACT"
ENTRY_GROUP = "vllm.general_plugins"
ENTRY_NAME = "vllm_ascend_quant"
ENTRY_VALUE = "vllm_ascend_quant_ext.plugin:register"
BUNDLE_GROUP = "vllm_hust.extension_bundles"
BUNDLE_ID = "org.vllm-hust.ascend-quant-runtime"
BUNDLE_VALUE = "vllm_ascend_quant_ext.manifests"
MANIFEST = "vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json"
SCHEME = "ASCEND_QUANT_W8A8"
LINEAR_SCHEME_CLASS = "AscendQuantW8A8LinearMethod"
SERVED_NAME = "w8a8-plugin-v1"
PROMPT = "Reply with exactly the two letters OK and nothing else."
EXPECTED_CONTENT = "OK"
SHIM = """import logging
import sys

# This host configures only the "vllm" logger tree, so the extension's own
# records are dropped at the default root level.  Raising the level here keeps
# the wheel and the host untouched while making activation observable.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
"""

DIST_PROBE = """\
import json
from importlib.metadata import distribution

try:
    dist = distribution("vllm-ascend-quant-ext")
except Exception:
    print("PROBE " + json.dumps({"installed": False}))
else:
    editable = None
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
    except Exception:
        direct = {}
    if direct.get("dir_info", {}).get("editable"):
        editable = direct.get("url", "").removeprefix("file://")
    print(
        "PROBE "
        + json.dumps(
            {
                "installed": True,
                "version": dist.version,
                "editable": editable,
                "module_file": __import__("vllm_ascend_quant_ext").__file__,
            }
        )
    )
"""

RESOLUTION_PROBE = """\
import json
from importlib.metadata import distribution, entry_points

dist = distribution("vllm-ascend-quant-ext")
manifest = dist.locate_file("vllm_ascend_quant_ext/manifests/vllm-hust-extension-v0.2.json")
document = json.loads(manifest.read_text(encoding="utf-8"))
general = sorted(
    (ep.name, ep.value) for ep in entry_points(group="vllm.general_plugins") if ep.name == "vllm_ascend_quant"
)
bundles = sorted((ep.name, ep.value) for ep in entry_points(group="vllm_hust.extension_bundles"))
import vllm_ascend_quant_ext

print(
    "PROBE "
    + json.dumps(
        {
            "version": dist.version,
            "module_file": vllm_ascend_quant_ext.__file__,
            "from_site_packages": "/site-packages/" in vllm_ascend_quant_ext.__file__,
            "manifest_exists": manifest.is_file(),
            "manifest_extension_id": document["extension_id"],
            "general": general,
            "bundles": bundles,
            "record_lists_manifest": any(
                "manifests/vllm-hust-extension-v0.2.json" in str(item) for item in (dist.files or [])
            ),
        }
    )
)
"""

LOADER_PROBE = """\
import json

from vllm.plugins import load_general_plugins

load_general_plugins()
from vllm_ascend.quantization.methods import get_scheme_class

linear = get_scheme_class("ASCEND_QUANT_W8A8", "linear")
moe = get_scheme_class("ASCEND_QUANT_W8A8", "moe")
print(
    "PROBE "
    + json.dumps(
        {
            "linear": getattr(linear, "__name__", str(linear)),
            "linear_module": getattr(linear, "__module__", None),
            "moe": getattr(moe, "__name__", str(moe)) if moe is not None else None,
        }
    )
)
"""

LEGACY_PROBE = """\
import json

from vllm.plugins import load_general_plugins

try:
    load_general_plugins()
except RuntimeError as exc:
    print("PROBE " + json.dumps({"rejected": True, "reason": str(exc)}))
else:
    print("PROBE " + json.dumps({"rejected": False}))
"""

UNINSTALL_PROBE = """\
import json
from importlib.metadata import entry_points

leftovers = []
try:
    import vllm_ascend_quant_ext  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    leftovers.append("module")

leftovers += [
    ep.name
    for ep in entry_points(group="vllm.general_plugins")
    if ep.name == "vllm_ascend_quant"
]
leftovers += [ep.name for ep in entry_points(group="vllm_hust.extension_bundles")]
print("PROBE " + json.dumps({"leftovers": leftovers}))
"""


def _probe(text: str) -> str:
    """Return a probe program, rejecting one that cannot report a record."""

    if "print(\"PROBE " not in text and "'PROBE '" not in text:
        raise ValueError("probe does not emit a PROBE record")
    return text


def _quote(parts: list[str]) -> str:
    """Quote a command so it can be pasted into a shell verbatim."""

    return " ".join(shlex.quote(part) for part in parts)


def _shell(env_scripts: list[str], command: str, exports: dict[str, str]) -> str:
    lines = [f"source {script}" for script in env_scripts]
    lines += [f"export {name}={shlex.quote(value)}" for name, value in exports.items()]
    lines.append(f"exec {command}")
    return "; ".join(lines)


def _run(command: str, env: dict[str, str], dry_run: bool) -> subprocess.CompletedProcess[str]:
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _probe_result(
    completed: subprocess.CompletedProcess[str],
    dry_run: bool,
) -> dict[str, object]:
    """Return the last PROBE record, failing closed when it is missing."""

    if dry_run:
        return {}
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"probe failed with exit {completed.returncode}: " + " | ".join(tail))
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("PROBE "):
            return json.loads(line[len("PROBE ") :])
    tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
    raise RuntimeError("probe produced no record: " + " | ".join(tail))


def _file_hashes(root: Path) -> list[str]:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(root)}")
    return lines


def _tree_hash(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _opener() -> urllib.request.OpenerDirector:
    # Bypass ambient proxies: the gate always talks to the local server.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _health(port: int, timeout: float = 5.0) -> int | None:
    try:
        with _opener().open(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            return response.status
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _chat(port: int, timeout: float) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": SERVED_NAME,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": 16,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with _opener().open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _content(reply: dict[str, object]) -> str:
    choices = reply["choices"]  # type: ignore[index]
    message = choices[0]["message"]  # type: ignore[index]
    return str(message["content"])  # type: ignore[index]


def _stop(process: subprocess.Popen[bytes] | None) -> int:
    if process is None or process.poll() is not None:
        return 0
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return 0
    for _ in range(30):
        if process.poll() is not None:
            return 0
        time.sleep(1)
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    for _ in range(10):
        if process.poll() is not None:
            return 0
        time.sleep(1)
    return 1



def _extension_exports(args: argparse.Namespace) -> dict[str, str]:
    return {
        "VLLM_ASCEND_TORCH_PREFLIGHT": "0",
        "ASCEND_RT_VISIBLE_DEVICES": str(args.device),
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "COMPILE_CUSTOM_KERNELS": "1",
        "PYTHONUNBUFFERED": "1",
    }


def _probe_env(args: argparse.Namespace, enabled: bool, shim: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    paths = [*args.host_source]
    if shim is not None:
        paths.append(str(shim))
    if paths:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = ":".join([*paths, existing]) if existing else ":".join(paths)
    env.pop(ENABLE_ENV, None)
    env.pop(ARTIFACT_ENV, None)
    env.update(_extension_exports(args))
    if enabled:
        env[ENABLE_ENV] = "1"
        env[ARTIFACT_ENV] = str(args.artifact)
    return env


def _probe_command(
    args: argparse.Namespace,
    python: str,
    program: str,
    *,
    enabled: bool,
    artifact: Path | None = None,
) -> str:
    exports = _extension_exports(args)
    if enabled:
        exports[ENABLE_ENV] = "1"
        exports[ARTIFACT_ENV] = str(artifact or args.artifact)
    return _shell(args.env_script, _quote([python, "-c", program]), exports)


def _install_command(python: str, wheel: Path) -> str:
    return _quote(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            "--force-reinstall",
            str(wheel),
        ]
    )


def _restore_command(python: str, editable: str) -> str:
    return _quote(
        [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", editable]
    )


def _server_command(args: argparse.Namespace, python: str, log: Path) -> str:
    vllm = args.vllm or str(Path(python).with_name("vllm"))
    exports = _extension_exports(args)
    exports[ENABLE_ENV] = "1"
    exports[ARTIFACT_ENV] = str(args.artifact)
    serve = _quote(
        [
            vllm,
            "serve",
            str(args.artifact),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--served-model-name",
            SERVED_NAME,
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--enforce-eager",
        ]
    )
    return _shell(args.env_script, f"{serve} > {shlex.quote(str(log))} 2>&1", exports)



def describe_plan(args: argparse.Namespace, python: str, log: Path) -> list[str]:
    """Return the plan text; ``--dry-run`` prints exactly these commands."""

    steps = [
        "1. record the wheel hash and the artifact hashes",
        f"   sha256sum {args.wheel}",
        f"   hash every file under {args.artifact}",
        "   host sources on PYTHONPATH: "
        + (", ".join(args.host_source) if args.host_source else "none"),
        f"   host environment scripts: {', '.join(args.env_script) if args.env_script else 'none'}",
        "2. install the wheel non-editable; the extension source tree is never on sys.path",
        f"   {_install_command(python, args.wheel)}",
        "   probe: version, module file, site-packages resolution, both entry-point groups,",
        "          manifest path, manifest in RECORD, manifest extension_id",
        f"   {_probe_command(args, python, RESOLUTION_PROBE, enabled=False)}",
        "3. activate through vLLM's own general-plugin loader and capture the record",
        f"   {_probe_command(args, python, LOADER_PROBE, enabled=True)}",
        f"   expect an admission record for {args.artifact} and the scheme class",
        f"   {LINEAR_SCHEME_CLASS}",
        "4. contrast: extension disabled leaves the scheme unregistered",
        f"   {_probe_command(args, python, LOADER_PROBE, enabled=False)}",
    ]
    if args.legacy_artifact:
        steps += [
            "   contrast: the legacy artifact is rejected at plugin-load time",
            "   "
            + _probe_command(
                args, python, LEGACY_PROBE, enabled=True, artifact=args.legacy_artifact
            ),
        ]
    steps += [
        "5. serve the reference artifact from the wheel",
        f"   {_server_command(args, python, log)}",
        f"   poll http://127.0.0.1:{args.port}/health for {args.health_timeout}s",
        "   send two identical chat completions and compare the content",
        "6. stop the server and prove every entry point vanished",
        f"   {_quote([python, '-m', 'pip', 'uninstall', '-y', DIST_NAME])}",
        f"   {_probe_command(args, python, UNINSTALL_PROBE, enabled=False)}",
        "7. restore the previous install and re-hash the artifact",
        "   reinstall the editable checkout recorded in phase 1 when there was one",
        "   " + _restore_command(python, "<editable-dir-recorded-in-phase-1>"),
        "   compare every artifact file hash with phase 1",
    ]
    return steps



def _gate(args: argparse.Namespace, python: str, log: Path) -> dict[str, object]:
    """Run every phase; the caller prints the summary and chooses the exit code."""

    summary: dict[str, object] = {
        "wheel": str(args.wheel),
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "artifact": str(args.artifact),
        "device": args.device,
        "port": args.port,
        "served_model_name": SERVED_NAME,
        "phases": [],
        "ok": False,
    }
    phases: list[dict[str, object]] = summary["phases"]  # type: ignore[assignment]

    def record(name: str, ok: bool, **detail: object) -> None:
        phases.append({"phase": name, "ok": ok, **detail})

    def probe(program: str, *, enabled: bool, shim: Path | None, artifact: Path | None = None):
        return _probe_result(
            _run(
                _probe_command(args, python, program, enabled=enabled, artifact=artifact),
                _probe_env(args, enabled=enabled, shim=shim),
                False,
            ),
            False,
        )

    cwd = Path(tempfile.mkdtemp(prefix="npu-e2e-"))
    shim = cwd / "logging-shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(SHIM, encoding="utf-8")

    before_lines: list[str] = []
    previous: dict[str, object] = {"installed": False, "editable": None}
    server: subprocess.Popen[bytes] | None = None
    interrupted: BaseException | None = None

    try:
        if not args.skip_hash:
            before_lines = _file_hashes(args.artifact)
            summary["artifact_tree_sha256_before"] = _tree_hash(before_lines)
            record(
                "artifact-hash-before",
                True,
                files=len(before_lines),
                tree_sha256=summary["artifact_tree_sha256_before"],
            )

        previous = probe(DIST_PROBE, enabled=False, shim=None)
        record("previous-install", True, **previous)

        _run(_install_command(python, args.wheel), _probe_env(args, False, None), False)
        resolution = probe(RESOLUTION_PROBE, enabled=False, shim=None)
        module_file = str(resolution.get("module_file", ""))
        record(
            "wheel-resolution",
            resolution.get("from_site_packages") is True
            and resolution.get("manifest_exists") is True
            and resolution.get("record_lists_manifest") is True
            and resolution.get("manifest_extension_id") == BUNDLE_ID
            and resolution.get("general") == [[ENTRY_NAME, ENTRY_VALUE]]
            and resolution.get("bundles") == [[BUNDLE_ID, BUNDLE_VALUE]],
            **resolution,
        )
        if "site-packages" not in module_file:
            raise RuntimeError(f"the wheel did not take over the import path: {module_file}")

        enabled = _run(
            _probe_command(args, python, LOADER_PROBE, enabled=True),
            _probe_env(args, True, shim),
            False,
        )
        payload = _probe_result(enabled, False)
        admitted = "admitted" in enabled.stderr and SCHEME in enabled.stderr
        record(
            "loader-enabled",
            admitted,
            admitted_record=admitted,
            admission_line=next(
                (line.strip() for line in enabled.stderr.splitlines() if "admitted" in line),
                "",
            ),
            **payload,
        )

        disabled = probe(LOADER_PROBE, enabled=False, shim=shim)
        record("loader-disabled", disabled.get("linear") == "None", **disabled)

        if args.legacy_artifact:
            legacy = probe(LEGACY_PROBE, enabled=True, shim=shim, artifact=args.legacy_artifact)
            record(
                "loader-legacy-artifact",
                legacy.get("rejected") is True,
                legacy_artifact=str(args.legacy_artifact),
                **legacy,
            )

        server = subprocess.Popen(
            ["bash", "-c", _server_command(args, python, log)],
            env=_probe_env(args, enabled=True, shim=None),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + args.health_timeout
        status = None
        while time.monotonic() < deadline:
            if server.poll() is not None:
                break
            status = _health(args.port)
            if status == 200:
                break
            time.sleep(args.poll_interval)
        ready = status == 200
        record("server-health", ready, status=status, exit_code=server.poll(), log=str(log))

        if ready:
            first = _chat(args.port, args.request_timeout)
            second = _chat(args.port, args.request_timeout)
            content_first = _content(first)
            content_second = _content(second)
            record(
                "deterministic-inference",
                content_first == EXPECTED_CONTENT
                and content_second == EXPECTED_CONTENT
                and content_first == content_second,
                content=content_first,
                identical=content_first == content_second,
                system_fingerprint=first.get("system_fingerprint"),
                prompt=PROMPT,
            )
        else:
            record("deterministic-inference", False, skipped="the server never became healthy")

        record("server-stop", _stop(server) == 0 and _health(args.port) is None)
        server = None

        if not args.keep_installed:
            _run(
                _quote([python, "-m", "pip", "uninstall", "-y", DIST_NAME]),
                _probe_env(args, False, None),
                False,
            )
            leftovers = probe(UNINSTALL_PROBE, enabled=False, shim=None)
            record("uninstall", leftovers.get("leftovers") == [], **leftovers)

        editable = previous.get("editable")
        if isinstance(editable, str) and editable:
            _run(_restore_command(python, editable), _probe_env(args, False, None), False)
            restored = probe(DIST_PROBE, enabled=False, shim=None)
            record(
                "restore-editable",
                restored.get("editable") == editable,
                expected=editable,
                **restored,
            )
        elif previous.get("installed") is True and not args.keep_installed:
            record(
                "restore-editable",
                False,
                error=(
                    "the extension was installed non-editable (version "
                    f"{previous.get('version')}); reinstall that build by hand"
                ),
            )
        else:
            record("restore-editable", True, note="no previous install to restore")

        if not args.skip_hash:
            after_lines = _file_hashes(args.artifact)
            summary["artifact_tree_sha256_after"] = _tree_hash(after_lines)
            record(
                "artifact-hash-after",
                before_lines == after_lines,
                files=len(after_lines),
                tree_sha256=summary["artifact_tree_sha256_after"],
                changed=sum(1 for a, b in zip(before_lines, after_lines) if a != b),
            )
    except BaseException as exc:
        interrupted = exc
        record("aborted", False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if server is not None:
            _stop(server)
        if interrupted is None and not args.keep_installed:
            if not any(phase["phase"] == "restore-editable" for phase in phases):
                editable = previous.get("editable")
                if isinstance(editable, str) and editable:
                    try:
                        _run(_restore_command(python, editable), _probe_env(args, False, None), False)
                        record("restore-editable", True, recovered=True)
                    except Exception as exc:  # pragma: no cover - best effort
                        record("restore-editable", False, error=str(exc))

    failed = [phase["phase"] for phase in phases if phase["ok"] is False]
    summary["failed_phases"] = failed
    summary["ok"] = not failed and interrupted is None
    if interrupted is not None:
        raise interrupted
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-host NPU end-to-end gate for a built runtime wheel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wheel", required=True, type=Path, help="wheel to install and serve")
    parser.add_argument("--artifact", required=True, type=Path, help="reference model artifact")
    parser.add_argument("--legacy-artifact", type=Path, help="artifact that must be rejected")
    parser.add_argument("--python", type=Path, help="interpreter of the frozen host")
    parser.add_argument("--vllm", type=Path, help="vLLM executable; defaults to <python dir>/vllm")
    parser.add_argument(
        "--env-script",
        action="append",
        default=[],
        help="host environment script to source; repeat for CANN, ATB and custom ops",
    )
    parser.add_argument(
        "--host-source",
        action="append",
        default=[],
        help="frozen host source tree for PYTHONPATH; never pass the extension source tree",
    )
    parser.add_argument("--device", type=int, default=0, help="ASCEND_RT_VISIBLE_DEVICES value")
    parser.add_argument("--port", type=int, default=18003, help="serving port")
    parser.add_argument("--log", type=Path, help="serving log path")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--health-timeout", type=float, default=1800.0, help="seconds")
    parser.add_argument("--poll-interval", type=float, default=20.0, help="seconds")
    parser.add_argument("--request-timeout", type=float, default=180.0, help="seconds")
    parser.add_argument("--skip-hash", action="store_true", help="skip the 16 GB artifact hashing")
    parser.add_argument(
        "--keep-installed",
        action="store_true",
        help="leave the wheel installed instead of uninstalling and restoring",
    )
    parser.add_argument("--summary", type=Path, help="write the JSON summary here")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return parser


def _validate(args: argparse.Namespace, python: str) -> None:
    if not args.wheel.is_file():
        raise SystemExit(f"wheel not found: {args.wheel}")
    if not args.artifact.is_dir():
        raise SystemExit(f"artifact directory not found: {args.artifact}")
    if args.legacy_artifact is not None and not args.legacy_artifact.is_dir():
        raise SystemExit(f"legacy artifact directory not found: {args.legacy_artifact}")
    if not Path(python).is_file():
        raise SystemExit(f"python not found: {python}")
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    offenders = [item for item in args.host_source if source_root in str(Path(item).resolve())]
    if offenders:
        raise SystemExit(
            "the extension source tree must not be on PYTHONPATH: " + ", ".join(offenders)
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    python = str(args.python.resolve()) if args.python else sys.executable
    _validate(args, python)

    log = args.log or (args.artifact.parent / f"npu-e2e-{args.wheel.stem}-port{args.port}.log")
    if args.dry_run:
        for line in describe_plan(args, python, log):
            print(line)
        return 0

    summary = _gate(args, python, log)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
