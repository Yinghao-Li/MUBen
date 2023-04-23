import logging
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = ['DATASET_NAMES', 'SPLITTING', 'EVAL_METRICS',
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
    "Tox21",
    "ESOL",
    "FreeSolv",
    "Lipophilicity",
    "PCBA",
    "MUV",
    "HIV",
    "BACE",
    "BBBP",
    "ToxCast",
    "SIDER",
    "ClinTox",
]

SPLITTING = {
    "Tox21": "random",
    "ESOL": "random",
    "FreeSolv": "random",
    "Lipophilicity": "random",
    "PCBA": "random",
    "MUV": "random",
    "HIV": "scaffold",
    "BACE": "scaffold",
    "BBBP": "scaffold",
    "ToxCast": "random",
    "SIDER": "random",
    "ClinTox": "random",
}

EVAL_METRICS = {
    "ESOL": "RMSE",
    "FreeSolv": "RMSE",
    "Lipophilicity": "RMSE",
    "PCBA": "PRC-AUC",
    "MUV": "PRC-AUC",
    "HIV": "ROC-AUC",
    "BACE": "ROC-AUC",
    "BBBP": "ROC-AUC",
    "Tox21": "ROC-AUC",
    "ToxCast": "ROC-AUC",
    "SIDER": "ROC-AUC",
    "ClinTox": "ROC-AUC",
}

MODEL_NAMES = [
    "DNN",
    "ChemBERTa",
    "GROVER",
    "Uni-Mol"
]


class UncertaintyMethods(StrEnum):
    none = 'none'
    mc_dropout = 'MC-Dropout'
    temperature = 'Temperature-Scaling'
    swag = 'SWAG'
    bbp = 'BBP'
    sgld = 'SGLD'
    laplace = 'Laplace-Approximation'
    ensembles = 'Deep-Ensembles'
    conformal = 'Conformal-Prediction'
    focal = 'Focal-Loss'


FINGERPRINT_FEATURE_TYPES = [
    "none",
    "rdkit",
    "morgan"
]
