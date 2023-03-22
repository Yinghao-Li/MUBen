import torch


def logit_to_var(logit, minimum_variance=1e-6):
    """
    Convert the network output logit to valid variance values.
    Refer to "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" for details
    """
    return torch.log(1 + torch.exp(logit)) + minimum_variance
