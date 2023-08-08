"""
# Author: Yinghao Li
# Modified: August 8th, 2023
# ---------------------------------------
# Description: A simple GIN, modified from torch_geometric example.
"""


import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from typing import Optional

from muben.base.model import OutputLayer


class GIN(nn.Module):
    def __init__(self,
                 n_lbs: int,
                 n_tasks: int,
                 max_atomic_num: int = 100,
                 d_hidden: int = 64,
                 n_layers: int = 3,
                 uncertainty_method: Optional[int] = 'none',
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.emb = nn.Embedding(max_atomic_num, d_hidden)
        self.gnn = pygnn.GIN(d_hidden, d_hidden, n_layers, dropout=dropout, jk='cat')
        self.ffn = pygnn.MLP(
            [d_hidden, d_hidden, d_hidden], norm="batch_norm", dropout=dropout
        )
        self.output_layer = OutputLayer(d_hidden, n_lbs * n_tasks, uncertainty_method, **kwargs)

    def forward(self, batch, **kwargs):
        atoms_ids = batch.atom_ids
        edge_indices = batch.edge_indices
        mol_ids = batch.mol_ids

        embs = self.emb(atoms_ids)
        x = self.gnn(embs, edge_indices)
        x = pygnn.global_add_pool(x, mol_ids)
        # x = self.ffn(x)
        logits = self.output_layer(x)
        return logits

    def training_step(self, data):
        y_hat = self(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(y_hat, data.y)
        self.train_acc(y_hat.softmax(dim=-1), data.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss
