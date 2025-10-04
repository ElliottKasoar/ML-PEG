"""Get MLIPs to be used for calculations or analysis."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import yaml

from ml_peg.models import MODELS_ROOT

if TYPE_CHECKING:
    from mlipx import GenericASECalculator as MlipxGenericASECalc


def get_subset(
    all_models: dict[str, MlipxGenericASECalc], models: None | str | Iterable = None
):
    if models is None:
        return all_models

    if isinstance(models, str):
        models = models.split(",")

    return {k: all_models[k] for k in models}


def load_models(models: None | str | Iterable = None) -> dict[str, MlipxGenericASECalc]:
    """Load models for use in calculations."""
    from ml_peg.models.models import FairChemCalc, GenericASECalc, OrbCalc

    loaded_models = {}

    print("MODELS loading...", models)

    # Load models from registry YAML: models.yml
    with open(MODELS_ROOT / "models.yml") as file:
        all_models = yaml.safe_load(file)

    for name, cfg in get_subset(all_models, models).items():
        print(f"Loading model from models.yml: {name}")

        if cfg["class_name"] == "FAIRChemCalculator":
            kwargs = cfg.get("kwargs", {})
            loaded_models[name] = FairChemCalc(
                model_name=kwargs["model_name"],
                task_name=kwargs.get("task_name", "omat"),
                device=cfg.get("device", "cpu"),
                overrides=kwargs.get("overrides", {}),
            )
        elif cfg["class_name"] == "OrbCalc":
            kwargs = cfg.get("kwargs", {})
            loaded_models[name] = OrbCalc(
                name=kwargs["name"],
                device=cfg.get("device", "cpu"),
            )
        else:
            loaded_models[name] = GenericASECalc(
                module=cfg["module"],
                class_name=cfg["class_name"],
                device=cfg.get("device", "auto"),
                kwargs=cfg.get("kwargs", {}),
            )

    return loaded_models


def get_model_names(models: None | Iterable = None) -> list[str]:
    """Load models names for use in analysis."""
    # Load models from registry YAML: models.yml
    with open(MODELS_ROOT / "models.yml") as file:
        all_models = yaml.safe_load(file)

    model_names = []
    for name in get_subset(all_models, models):
        model_names.append(name)

    return model_names
