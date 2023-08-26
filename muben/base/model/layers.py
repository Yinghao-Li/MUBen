"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description: 

Versatile Output Layer for Backbone Models

This module implements an output layer that can be applied to various
backbone models. It provides the option of utilizing different uncertainty 
methods for model's output and supports both classification and regression tasks.

"""


import torch
import torch.nn as nn
from ..uncertainty.bbp import BBPOutputLayer
from ..uncertainty.evidential import NIGOutputLayer
from muben.utils.macro import UncertaintyMethods


class OutputLayer(nn.Module):
    """
    Customizable output layer for various backbone models.

    This class provides an interface to add an output layer with or without uncertainty methods.
    """

    def __init__(
        self,
        last_hidden_dim,
        n_output_heads,
        uncertainty_method=UncertaintyMethods.none,
        task_type="classification",
        **kwargs
    ):
        """
        Initialize the model output layer.

        Parameters
        ----------
        last_hidden_dim : int
            Dimensionality of the last hidden state from the backbone model.
        n_output_heads : int
            Number of output heads (e.g., number of classes for classification).
        uncertainty_method : str, optional
            Method to introduce uncertainty in the output layer.
            Available methods are defined in UncertaintyMethods.
        task_type : str, optional
            Type of task - "classification" or "regression".
        kwargs : dict
            Additional keyword arguments to be passed to the specific output layers.
        """

        super().__init__()

        self._uncertainty_method = uncertainty_method
        self._task_type = task_type

        # Choose the output layer based on the uncertainty method and task type
        if uncertainty_method == UncertaintyMethods.bbp:
            self.output_layer = BBPOutputLayer(
                last_hidden_dim, n_output_heads, **kwargs
            )
        elif (
            uncertainty_method == UncertaintyMethods.evidential
            and task_type == "regression"
        ):
            self.output_layer = NIGOutputLayer(
                last_hidden_dim, n_output_heads, **kwargs
            )
        else:
            self.output_layer = nn.Linear(last_hidden_dim, n_output_heads)

        # Kullback-Leibler Divergence for Bayesian methods
        self.kld = None
        self.initialize()

    def initialize(self) -> "OutputLayer":
        """
        Initialize the weights of the output layer.

        Different initializations are applied based on the uncertainty method and task type.

        Returns
        -------
        OutputLayer
            Initialized model instance.
        """
        if self._uncertainty_method == UncertaintyMethods.bbp:
            self.output_layer.initialize()
            self.kld = None
        elif (
            self._uncertainty_method == UncertaintyMethods.evidential
            and self._task_type == "regression"
        ):
            self.output_layer.initialize()
        else:
            nn.init.xavier_uniform_(self.output_layer.weight)
            self.output_layer.bias.data.fill_(0.01)
        return self

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the output layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the output layer.

        Returns
        -------
        logits : torch.Tensor
            The output logits or values of the model.
        """
        if self._uncertainty_method == UncertaintyMethods.bbp:
            logits, self.kld = self.output_layer(x)
        else:
            logits = self.output_layer(x)

        return logits
