"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: A simple deep neural network with customizable activation function.
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import OutputLayer

__all__ = ["DNN"]


class DNN(nn.Module):
    """
    Simple Deep Neural Network (DNN) with customizable layers and activation functions.

    Extends PyTorch's Module to provide additional functionality for constructing
    and initializing a deep neural network.

    Attributes
    ----------
    input_layer : torch.nn.Sequential
        The initial input layer.
    hidden_layers : torch.nn.Sequential
        Hidden layers in the neural network.
    output_layer : muben.base.model.OutputLayer
        The final output layer.
    """

    def __init__(
        self,
        d_feature: int,
        n_lbs: int,
        n_tasks: int,
        n_hidden_layers: Optional[int] = 4,
        d_hidden: Optional[int] = 128,
        p_dropout: Optional[float] = 0.1,
        hidden_dims: Optional[list] = None,
        activation: Optional[str] = "ReLU",
        uncertainty_method: Optional[int] = "none",
        **kwargs
    ):
        """
        Initialize the DNN.

        Parameters
        ----------
        d_feature : int
            Input feature dimension.
        n_lbs : int
            Number of labels.
        n_tasks : int
            Number of tasks.
        n_hidden_layers : Optional[int]
            Number of hidden layers. Defaults to 4.
        d_hidden : Optional[int]
            Dimension of each hidden layer. Defaults to 128.
        p_dropout : Optional[float]
            Dropout probability. Defaults to 0.1.
        hidden_dims : Optional[List[int]]
            List specifying dimensions of each hidden layer.
            Overrides `n_hidden_layers` and `d_hidden` if provided.
        activation : Optional[str]
            Activation function to use. Must be an attribute of torch.nn. Defaults to 'ReLU'.
        uncertainty_method : Optional[str]
            Method for calculating uncertainty. Defaults to 'none'.
        **kwargs
            Additional keyword arguments for the output layer.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [d_hidden] * (n_hidden_layers + 1)
        else:
            n_hidden_layers = len(hidden_dims)

        self.input_layer = nn.Sequential(
            nn.Linear(d_feature, hidden_dims[0]),
            getattr(nn, activation)(),
            nn.Dropout(p_dropout),
        )

        hidden_layers = [
            nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                getattr(nn, activation)(),
                nn.Dropout(p_dropout),
            )
            for i in range(n_hidden_layers)
        ]
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = OutputLayer(hidden_dims[-1], n_lbs * n_tasks, uncertainty_method, **kwargs)

        self.initialize()

    def initialize(self):
        """
        Initialize the weights of the DNN using Xavier uniform initialization.

        Returns
        -------
        DNN
            Returns the initialized DNN model.
        """

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_weights)
        self.output_layer.initialize()
        return self

    def forward(self, batch, **kwargs):
        """
        Perform a forward pass on the input batch.

        Parameters
        ----------
        batch : object
            Input batch which must have attribute 'features' representing input features.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Output logits.
        """
        features = batch.features

        x = self.input_layer(features)
        x = self.hidden_layers(x)

        logits = self.output_layer(x)

        return logits
