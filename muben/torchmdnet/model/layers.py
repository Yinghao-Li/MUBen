from torch import nn
from .modules import GatedEquivariantBlock


class EquivariantScalar(nn.Module):
    def __init__(self, hidden_channels, activation="silu", external_output_layer=False):
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
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
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
    def __init__(self, hidden_channels, activation="silu"):
        super().__init__(hidden_channels, activation)

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()
