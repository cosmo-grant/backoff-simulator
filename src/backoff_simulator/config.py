"""Load simulation configuration from a toml file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackoffConfig:
    type: str
    params: dict[str, float]


@dataclass(frozen=True)
class Spec:
    """A specification of all simulation parameters."""

    max_clients: int
    repeat: int
    network_mu: float
    network_sigma: float
    work_to_duration: float
    control: str
    control_params: dict[str, float]
    strategies: list[BackoffConfig]


def load_config(path: Path) -> list[Spec]:
    with open(path, "rb") as f:
        config = tomllib.load(f)

    specs = config["simulation"]
    return [parse_spec(s) for s in specs]


def parse_spec(spec: dict) -> Spec:
    # These keys are required.
    max_clients = spec.pop("max_clients")
    repeat = spec.pop("repeat")
    network_mu = spec.pop("network_mu")
    network_sigma = spec.pop("network_sigma")
    work_to_duration = spec.pop("work_to_duration")
    control = spec.pop("control")
    strategies = spec.pop("strategies")

    # Any remaining keys should be control params.
    control_params = spec

    strategies = [parse_strategy(s) for s in strategies]

    return Spec(
        max_clients=max_clients,
        repeat=repeat,
        network_mu=network_mu,
        network_sigma=network_sigma,
        work_to_duration=work_to_duration,
        control=control,
        control_params=control_params,
        strategies=strategies,
    )


def parse_strategy(strategy: dict) -> BackoffConfig:
    type_ = strategy.pop("type")
    # Any remaining keys should be strategy params.
    params = strategy

    return BackoffConfig(type=type_, params=params)
