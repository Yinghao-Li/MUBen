# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch.nn as nn
from .layers import init_bert_params
from .encoder import TransformerEncoderWithPair
from .module import (
    ClassificationHead,
    NonLinearHead,
    DistanceHead,
    GaussianLayer
)


logger = logging.getLogger(__name__)


class UniMol(nn.Module):
    def __init__(self, config, dictionary):
        super().__init__()
        base_architecture(config)
        self.config = config
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), config.encoder_embed_dim, self.padding_idx
        )
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
        self.gbf_proj = NonLinearHead(
            k, config.encoder_attention_heads, config.activation_fn
        )
        self.gbf = GaussianLayer(k, n_edge_type)

        if config.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                config.encoder_attention_heads, 1, config.activation_fn
            )
        if config.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                config.encoder_attention_heads, config.activation_fn
            )
        self.classification_heads = nn.ModuleDict()

        self.apply(init_bert_params)

        self.output_layer = ClassificationHead(
            input_dim=self.config.encoder_embed_dim,
            inner_dim=self.config.encoder_embed_dim,
            num_classes=self.config.n_lbs * self.config.n_tasks,
            activation_fn=self.config.pooler_activation_fn,
            pooler_dropout=self.config.pooler_dropout,
        )

    def forward(self, batch, **kwargs):
        src_tokens, src_distance, src_edge_type = batch.atoms, batch.distances, batch.edge_types

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

        logits = self.output_layer(encoder_rep)
        return logits

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


def base_architecture(config):

    # Model architecture, should not be changed
    config.encoder_layers = 15
    config.encoder_embed_dim = 512
    config.encoder_ffn_embed_dim = 2048
    config.encoder_attention_heads = 64

    # Fix the dropout ratio to the original implementation
    config.dropout = 0.1
    config.emb_dropout = 0.1
    config.attention_dropout = 0.1
    config.activation_dropout = 0.0
    config.pooler_dropout = getattr(config, "pooler_dropout", 0.0)

    config.max_seq_len = getattr(config, "max_seq_len", 512)

    config.activation_fn = getattr(config, "activation_fn", "gelu")
    config.pooler_activation_fn = getattr(config, "pooler_activation_fn", "tanh")

    config.post_ln = getattr(config, "post_ln", False)
    config.masked_token_loss = getattr(config, "masked_token_loss", -1.0)
    config.masked_coord_loss = getattr(config, "masked_coord_loss", -1.0)
    config.masked_dist_loss = getattr(config, "masked_dist_loss", -1.0)
    config.x_norm_loss = getattr(config, "x_norm_loss", -1.0)
    config.delta_pair_repr_norm_loss = getattr(config, "delta_pair_repr_norm_loss", -1.0)
