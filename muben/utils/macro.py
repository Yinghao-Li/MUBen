"""
# Author: Yinghao Li
# Modified: November 30th, 2023
# ---------------------------------------
# Description: Constants
"""


import logging
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = [
    "DATASET_NAMES",
    "CLASSIFICATION_DATASET",
    "REGRESSION_DATASET",
    "dataset_mapping",
    "EVAL_METRICS",
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "metrics_mapping",
    "MODEL_NAMES",
    "UncertaintyMethods",
    "FINGERPRINT_FEATURE_TYPES",
    "StrEnum",
    "QM_DATASET",
    "PC_DATASET",
    "BIO_DATASET",
    "PHY_DATASET",
]


class StrEnum(str, Enum):
    @classmethod
    def options(cls):
        opts = list()
        for k, v in cls.__dict__.items():
            if not k.startswith("_") and k != "options":
                opts.append(v.value)
        return opts


DATASET_NAMES = [
    "tox21",
    "esol",
    "freesolv",
    "lipo",
    "muv",
    "hiv",
    "bace",
    "bbbp",
    "toxcast",
    "sider",
    "clintox",
    "qm7",
    "qm8",
    "qm9",
]

CLASSIFICATION_DATASET = [
    "bace",
    "bbbp",
    "clintox",
    "tox21",
    "toxcast",
    "sider",
    "hiv",
    "muv",
]

REGRESSION_DATASET = ["esol", "freesolv", "lipo", "qm7", "qm8", "qm9"]

# Quantum Mechanics
QM_DATASET = ["qm7", "qm8", "qm9"]

# Physical Chemistry
PC_DATASET = ["esol", "freesolv", "lipo"]

# Biophysics
BIO_DATASET = ["bace", "hiv", "muv"]

# Plysiology
PHY_DATASET = [
    "bbbp",
    "clintox",
    "tox21",
    "toxcast",
    "sider",
]


dataset_mapping = {
    "tox21": "Tox21",
    "esol": "ESOL",
    "freesolv": "FreeSolv",
    "lipo": "Lipophilicity",
    "muv": "MUV",
    "hiv": "HIV",
    "bace": "BACE",
    "bbbp": "BBBP",
    "toxcast": "ToxCast",
    "sider": "SIDER",
    "clintox": "ClinTox",
    "qm7": "QM7",
    "qm8": "QM8",
    "qm9": "QM9",
}

EVAL_METRICS = {
    "esol": "rmse",
    "freesolv": "rmse",
    "lipo": "rmse",
    "pcba": "prc-auc",
    "muv": "prc-auc",
    "hiv": "roc-auc",
    "bace": "roc-auc",
    "bbbp": "roc-auc",
    "tox21": "roc-auc",
    "toxcast": "roc-auc",
    "sider": "roc-auc",
    "clintox": "roc-auc",
    "qm7": "mae",
    "qm8": "mae",
    "qm9": "mae",
}

MODEL_NAMES = ["DNN", "ChemBERTa", "GROVER", "Uni-Mol", "TorchMD-NET", "GIN"]

CLASSIFICATION_METRICS = ["roc-auc", "ece", "nll", "brier"]
REGRESSION_METRICS = ["rmse", "mae", "nll", "ce"]
metrics_mapping = {
    "roc-auc": "ROC-AUC",
    "ece": "ECE",
    "nll": "NLL",
    "brier": "BS",
    "rmse": "RMSE",
    "mae": "MAE",
    "ce": "CE",
}


class UncertaintyMethods(StrEnum):
    none = "none"
    mc_dropout = "MCDropout"
    temperature = "TemperatureScaling"
    swag = "SWAG"
    bbp = "BBP"
    sgld = "SGLD"
    ensembles = "DeepEnsembles"
    conformal = "ConformalPrediction"
    focal = "FocalLoss"
    iso = "IsotonicCalibration"
    evidential = "Evidential"

    @classmethod
    def options(cls, classification_only=False, regression_only=False):
        if not classification_only and not regression_only:
            return super().options()
        elif classification_only:
            return [m for m in super().options() if m not in ["ConformalPrediction", "IsotonicCalibration"]]
        elif regression_only:
            return [m for m in super().options() if m not in ["TemperatureScaling", "FocalLoss"]]
        else:
            raise ValueError("Invalid arguments")


FINGERPRINT_FEATURE_TYPES = ["none", "rdkit", "morgan"]
