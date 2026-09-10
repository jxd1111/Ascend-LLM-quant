"""Adapter for invoking the msModelSlim CLI without shell interpolation."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Recipe


@dataclass(frozen=True)
class QuantizeRequest:
    recipe: Recipe
    model_path: Path
    output_path: Path
    device: str
    trust_remote_code: bool = False
    executable: str = "msmodelslim"

    def command(self) -> list[str]:
        return [
            self.executable,
            "quant",
            "--model_path",
            str(self.model_path),
            "--save_path",
            str(self.output_path),
            "--device",
            self.device,
            "--model_type",
            self.recipe.model_type,
            "--config_path",
            str(self.recipe.config_path()),
            "--trust_remote_code",
            str(self.trust_remote_code),
        ]

    def rendered_command(self) -> str:
        return shlex.join(self.command())


def validate_request(request: QuantizeRequest) -> None:
    if not request.model_path.is_dir():
        raise ValueError(f"Model directory does not exist: {request.model_path}")
    if not request.recipe.config_path().is_file():
        raise ValueError(f"Recipe YAML does not exist: {request.recipe.config_path()}")
    if request.output_path == request.model_path or request.model_path in request.output_path.parents:
        raise ValueError("Output directory must not be the input model or a child of it")
    if request.output_path.exists():
        if not request.output_path.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {request.output_path}")
        if any(request.output_path.iterdir()):
            raise ValueError(
                f"Output directory is not empty: {request.output_path}. "
                "Choose a new output path to avoid overwriting a model."
            )
    if shutil.which(request.executable) is None:
        raise ValueError(f"Executable is not available on PATH: {request.executable}")


def run_quantization(request: QuantizeRequest) -> int:
    validate_request(request)
    request.output_path.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(request.command(), check=False)
    return completed.returncode
