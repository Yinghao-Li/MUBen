import logging
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = ['DATASET_NAMES', 'SPLITTING', 'EVAL_METRICS',
           'MODEL_NAMES', 'UncertaintyMethods', 'FINGERPRINT_FEATURE_TYPES',
           'StrEnum', 'QM9_PROPERTIES']


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
    "QM9"
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
    "QM9": "random"
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
    "QM9": "MAE"
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

QM9_PROPERTIES = [
    # |Dipole moment|
    'mu',
    # |Isotropic polarizability|
    'alpha',
    # |Energy of Highest occupied molecular orbital (HOMO)|
    'homo',
    # |Energy of Lowest unoccupied molecular orbital (LUMO)|
    'lumo',
    # |Gap, difference between LUMO and HOMO|
    'gap',
    # |Electronic spatial extent|
    'r2',
    # |Zero point vibrational energy|
    'zpve',
    # |Internal energy at 0 K|
    'u0',
    # |Internal energy at 298.15 K|
    'u298',
    # |Enthalpy at 298.15 K|
    'h298',
    # |Free energy at 298.15 K|
    'g298',
    # |Heat capacity at 298.15 K|
    'cv'
]
