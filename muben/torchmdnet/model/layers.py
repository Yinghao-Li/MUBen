"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description: Equivariant Transformer Layers.
# Reference: Modified from https://github.com/shehzaidi/pre-training-via-denoising.
"""
from torch import nn
from .modules import GatedEquivariantBlock


class EquivariantScalar(nn.Module):
    """
    A module to process input features in an equivariant manner.

    Parameters
    ----------
    hidden_channels : int
        Number of channels in the hidden layers.
    activation : str, optional
        Activation function to be used. Default is "silu".
    external_output_layer : bool, optional
        If True, only the first layer of the output network is used. Default is False.

    Attributes
    ----------
    output_network : nn.ModuleList
        List containing the Gated Equivariant Blocks of the network.
    external_output_layer : bool
        Indicates whether the output layer is external or not.
    """

    def __init__(
        self, hidden_channels, activation="silu", external_output_layer=False
    ):
        super().__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    1,
                    activation=activation,
                ),
            ]
        )
        self.external_output_layer = external_output_layer

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters of the layers in the output network.
        """
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        """
        Forward propagate the input features through the network.

        Parameters
        ----------
        x : torch.Tensor
            Atom feature.
        v : torch.Tensor
            Edge feature.
        Other Parameters are unused.

        Returns
        -------
        torch.Tensor
            Processed tensor after propagation.
        """
        if self.external_output_layer:
            x, v = self.output_network[0](x, v)
            # include v in output to make sure all parameters have a gradient
            return x + v.sum() * 0

        else:
            for layer in self.output_network:
                x, v = layer(x, v)
            # include v in output to make sure all parameters have a gradient
            return x + v.sum() * 0


class EquivariantVectorOutput(EquivariantScalar):
    """
    Module to produce vector outputs in an equivariant manner.

    Parameters
    ----------
    hidden_channels : int
        Number of channels in the hidden layers.
    activation : str, optional
        Activation function to be used. Default is "silu".
    """

    def __init__(self, hidden_channels, activation="silu"):
        super().__init__(hidden_channels, activation)

    def pre_reduce(self, x, v, z, pos, batch):
        """
        Forward propagate the input features through the network.

        Parameters
        ----------
        x : torch.Tensor
            Atom feature.
        v : torch.Tensor
            Edge feature.
        Other Parameters are unused.

        Returns
        -------
        torch.Tensor
            Vector output after propagation.
        """
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()
