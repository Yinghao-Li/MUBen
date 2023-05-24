import logging
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = ['DATASET_NAMES', 'CLASSIFICATION_DATASET', 'REGRESSION_DATASET', 'EVAL_METRICS',
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
    "pcba",
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
    'hiv',
    'muv',
    'pcba',
    'sider',
    'tox21',
    'toxcast'
]

REGRESSION_DATASET = [
    'qm7',
    'qm8',
    'qm9',
    'esol',
    'freesolv',
    'lipo',
]

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
    "Uni-Mol"
]


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


FINGERPRINT_FEATURE_TYPES = [
    "none",
    "rdkit",
    "morgan"
]
