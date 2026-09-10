"""Stable, dependency-free data contracts used by the quantization plugin."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    """Description of one reproducible offline quantization recipe."""

    name: str
    description: str
    model_type: str
    config_resource: str
    producer_quant_type: str
    runtime_quant_type: str
    calibration_dataset: str
    weight_dtype: str
    activation_dtype: str
    quant_scheme: str

    def config_path(self) -> Path:
        resource = files("ascend_quant_toolkit").joinpath(self.config_resource)
        return Path(str(resource))

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "config_resource": self.config_resource,
            "producer_quant_type": self.producer_quant_type,
            "runtime_quant_type": self.runtime_quant_type,
            "calibration_dataset": self.calibration_dataset,
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "quant_scheme": self.quant_scheme,
        }
