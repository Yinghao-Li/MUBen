"""
# Author: Yinghao Li
# Modified: December 1st, 2023
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
    "DatasetNames",
    "ModelNames",
    "FINGERPRINT_FEATURE_TYPES",
    "StrEnum",
    "QM_DATASET",
    "PC_DATASET",
    "BIO_DATASET",
    "PHY_DATASET",
]

is_callable = lambda x: hasattr(x, "__call__")


class StrEnum(str, Enum):
    @classmethod
    def options(cls):
        opts = list()
        for k, v in cls.__dict__.items():
            if not k.startswith("_") and not isinstance(v, classmethod) and not is_callable(v):
                try:
                    opts.append(v.value)
                except AttributeError:
                    pass
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
    def options(cls, constraint: str = None):
        """
        Parameters
        ----------
        constraint : str
            The constraint for the options. If None, return all options.
            If "classification", return classification options.
            If "regression", return regression options.

        Returns
        -------
        options : list
            The options for the constraint.
        """
        if constraint is None:
            return super().options()
        elif constraint == "classification":
            return [m for m in super().options() if m not in ["ConformalPrediction", "IsotonicCalibration"]]
        elif constraint == "regression":
            return [m for m in super().options() if m not in ["TemperatureScaling", "FocalLoss"]]
        else:
            raise ValueError("Invalid constraint")


class DatasetNames(StrEnum):
    esol = "esol"
    freesolv = "freesolv"
    lipo = "lipo"
    muv = "muv"
    hiv = "hiv"
    bace = "bace"
    bbbp = "bbbp"
    tox21 = "tox21"
    toxcast = "toxcast"
    sider = "sider"
    clintox = "clintox"
    qm7 = "qm7"
    qm8 = "qm8"
    qm9 = "qm9"

    @classmethod
    def options(cls, task_type: str = None):
        """
        Parameters
        ----------
        task_type : str
            The task type for the options. If None, return all options.
            If "classification", return classification options.
            If "regression", return regression options.
            If "qm", return quantum mechanics options.
            If "pc", return physical chemistry options.
            If "bio", return biophysics options.
            If "phy", return physiology options.

        Returns
        -------
        options : list
            The options for the task type.
        """
        if task_type is None:
            return super().options()
        elif task_type == "classification":
            return ["muv", "hiv", "bace", "bbbp", "tox21", "toxcast", "sider", "clintox"]
        elif task_type == "regression":
            return ["esol", "freesolv", "lipo", "qm7", "qm8", "qm9"]
        elif task_type in ["qm", "quantum mechanics"]:
            return ["qm7", "qm8", "qm9"]
        elif task_type in ["pc", "physical chemistry"]:
            return ["esol", "freesolv", "lipo"]
        elif task_type in ["bio", "biophysics"]:
            return ["bace", "hiv", "muv"]
        elif task_type in ["phy", "physiology"]:
            return ["bbbp", "clintox", "tox21", "toxcast", "sider"]
        else:
            raise ValueError("Invalid task type")


class ModelNames(StrEnum):
    dnn = "DNN"
    chemberta = "ChemBERTa"
    grover = "GROVER"
    unimol = "Uni-Mol"
    torchmdnet = "TorchMD-NET"
    gin = "GIN"


FINGERPRINT_FEATURE_TYPES = ["none", "rdkit", "morgan"]
