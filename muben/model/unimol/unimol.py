"""
# Author: Yinghao Li
# Modified: March 4th, 2024
# ---------------------------------------
# Description: The Uni-Mol model
# Reference: Modified from https://github.com/dptech-corp/Uni-Mol/tree/main/unimol
"""

# Original copyright:
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn

from muben.layers import OutputLayer
from .layers import init_bert_params
from .encoder import TransformerEncoderWithPair
from .module import NonLinearHead, DistanceHead, GaussianLayer


logger = logging.getLogger(__name__)


class UniMol(nn.Module):
    """
    Uni-Mol model for molecular structure analysis.
    """

    def __init__(self, config, dictionary):
        """
        Initialize the UniMol model.

        Parameters
        ----------
        config : Configuration
            The configuration object containing model hyperparameters.
        dictionary : Dictionary
            The dictionary object mapping tokens to integers.
        """
        super().__init__()
        self.config = config
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), config.encoder_embed_dim, self.padding_idx)
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=config.encoder_layers,
            embed_dim=config.encoder_embed_dim,
            ffn_embed_dim=config.encoder_ffn_embed_dim,
            attention_heads=config.encoder_attention_heads,
            emb_dropout=config.emb_dropout,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            max_seq_len=config.max_seq_len,
            activation_fn=config.activation_fn,
            no_final_head_layer_norm=config.delta_pair_repr_norm_loss < 0,
        )

        k = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(k, config.encoder_attention_heads, config.activation_fn)
        self.gbf = GaussianLayer(k, n_edge_type)

        if config.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(config.encoder_attention_heads, 1, config.activation_fn)
        if config.masked_dist_loss > 0:
            self.dist_head = DistanceHead(config.encoder_attention_heads, config.activation_fn)

        self.apply(init_bert_params)

        self.hidden_layer = nn.Sequential(
            nn.Dropout(config.pooler_dropout),
            nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim),
            getattr(nn, config.pooler_activation_fn)(),
            nn.Dropout(config.pooler_dropout),
        )
        self.output_layer = OutputLayer(
            config.encoder_embed_dim,
            config.n_lbs * config.n_tasks,
            config.uncertainty_method,
            task_type=config.task_type,
            bbp_prior_sigma=config.bbp_prior_sigma,
        )

        if not config.disable_checkpoint_loading:
            state = self.load_checkpoint(self.config.checkpoint_path)
            self.load_state_dict(state["model"], strict=False)

    def load_checkpoint(self, path, arg_overrides=None):
        """
        Load a checkpoint to CPU.

        If present, the function also applies overrides to arguments present
        in the checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        arg_overrides : dict, optional
            Dictionary of arguments to be overridden in the loaded state.

        Returns
        -------
        dict
            Loaded state dictionary.
        """
        local_path = path
        with open(local_path, "rb") as f:
            state = torch.load(f, map_location=torch.device("cpu"))

        if "args" in state and state["args"] is not None and arg_overrides is not None:
            args = state["args"]
            for arg_name, arg_val in arg_overrides.items():
                setattr(args, arg_name, arg_val)

        return state

    def forward(self, batch, **kwargs):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        batch : Batch
            A batch of input data.

        Returns
        -------
        torch.Tensor
            Logits produced by the model.
        """
        src_tokens, src_distance, src_edge_type = (
            batch.atoms,
            batch.distances,
            batch.edge_types,
        )

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias_inner = gbf_result
            graph_attn_bias_inner = graph_attn_bias_inner.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias_inner = graph_attn_bias_inner.view(-1, n_node, n_node)
            return graph_attn_bias_inner

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        encoder_rep, _, _, _, _ = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)

        hidden_state = self.hidden_layer(encoder_rep[:, 0, :])  # take <s> token (equiv. to [CLS])
        logits = self.output_layer(hidden_state)
        return logits
