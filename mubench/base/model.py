import torch
import torch.nn as nn
from typing import Optional

from seqlbtoolkit.training.dataset import Batch


class DNN(nn.Module):

    def __init__(self,
                 d_feature: int,
                 n_lbs: int,
                 n_hidden_layers: Optional[int] = 4,
                 d_hidden: Optional[int] = 128,
                 p_dropout: Optional[float] = 0.1,
                 hidden_dims: Optional[list] = None,
                 activation: Optional[str] = 'ReLU',
                 **kwargs):

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [d_hidden] * (n_hidden_layers + 1)
        else:
            n_hidden_layers = len(hidden_dims)

        self.input_layer = nn.Sequential(
            nn.Linear(d_feature, hidden_dims[0]),
            getattr(nn, activation)(),
            nn.Dropout(p_dropout)
        )

        hidden_layers = [nn.Sequential(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
            getattr(nn, activation)(),
            nn.Dropout(p_dropout)
        ) for i in range(n_hidden_layers)]

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(hidden_dims[-1], n_lbs)

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_weights)
        return self

    def forward(self, batch: Batch, **kwargs):
        features = batch.features

        x = self.input_layer(features)
        x = self.hidden_layers(x)

        logits = self.output_layer(x).squeeze()
        if not logits.shape:
            logits = logits.unsqueeze(-1)

        return logits
