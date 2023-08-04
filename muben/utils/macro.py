"""
# Author: Yinghao Li
# Modified: August 4th, 2023
# ---------------------------------------
# Description: Constants
"""


import logging
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = ['DATASET_NAMES', 'CLASSIFICATION_DATASET', 'REGRESSION_DATASET', 'dataset_mapping',
           'EVAL_METRICS', 'CLASSIFICATION_METRICS', 'REGRESSION_METRICS', 'metrics_mapping',
           'MODEL_NAMES', 'UncertaintyMethods', 'FINGERPRINT_FEATURE_TYPES',
           'StrEnum']


class StrEnum(str, Enum):
    @classmethod
    def options(cls):
        opts = list()
        for k, v in cls.__dict__.items():
            if not k.startswith('_') and k != 'options':
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
    "qm9"
]

CLASSIFICATION_DATASET = [
    'bace',
    'bbbp',
    'clintox',
    'tox21',
    'toxcast',
    'sider',
    'hiv',
    'muv'
]

REGRESSION_DATASET = [
    'esol',
    'freesolv',
    'lipo',
    'qm7',
    'qm8',
    'qm9'
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
    "qm9": "QM9"
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
    "qm9": "mae"
}

MODEL_NAMES = [
    "DNN",
    "ChemBERTa",
    "GROVER",
    "Uni-Mol",
    "TorchMD-NET"
]

CLASSIFICATION_METRICS = ['roc-auc', 'ece', 'nll', 'brier']
REGRESSION_METRICS = ['rmse', 'mae', 'nll', 'ce']
metrics_mapping = {
    'roc-auc': 'ROC-AUC',
    'ece': 'ECE',
    'nll': 'NLL',
    'brier': 'BS',
    'rmse': 'RMSE',
    'mae': 'MAE',
    'ce': 'CE'
}


class UncertaintyMethods(StrEnum):
    none = 'none'
    mc_dropout = 'MCDropout'
    temperature = 'TemperatureScaling'
    swag = 'SWAG'
    bbp = 'BBP'
    sgld = 'SGLD'
    ensembles = 'DeepEnsembles'
    conformal = 'ConformalPrediction'
    focal = 'FocalLoss'
    iso = 'IsotonicCalibration',
    evidential = 'Evidential'


FINGERPRINT_FEATURE_TYPES = [
    "none",
    "rdkit",
    "morgan"
]
