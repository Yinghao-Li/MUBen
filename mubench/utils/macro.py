
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

UNCERTAINTY_METHODS = [
    "none",
    "MCDropout",
    "TemperatureScaling",
    "SWAG",
    "BBB",
    "SGLD",
    "LaplaceApproximation",
    "DeepEnsembles",
    "ConformalPrediction",
    "FocalLoss",
    "GaussianProcess",
]
