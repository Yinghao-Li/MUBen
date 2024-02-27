"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: ChemBERTa model.

This module implements the ChemBERTa model which is tailored for chemical informatics tasks. 
It leverages the pretrained BERT architecture from the HuggingFace `transformers` library 
and adds an output layer tailored for multi-label and multi-task prediction.
"""

import torch.nn as nn
from transformers import AutoModel

from muben.dataset.dataset import Batch
from .layers import OutputLayer


class ChemBERTa(nn.Module):
    """
    The ChemBERTa model, a modified version of BERT for chemical informatics tasks.

    Attributes:
        bert_model (transformers.PreTrainedModel): The pretrained BERT model.
        output_layer (OutputLayer): The output layer for multi-label and multi-task classification.

    Args:
        bert_model_name_or_path (str): Identifier or path for the BERT model to be loaded.
        n_lbs (int): Number of labels per task.
        n_tasks (int): Number of tasks.
        uncertainty_method (str): Method for modeling uncertainty.
        **kwargs: Additional keyword arguments passed to the output layer's initialization.
    """

    def __init__(self, bert_model_name_or_path: str, n_lbs: int, n_tasks: int, uncertainty_method: str, **kwargs):
        super().__init__()

        self.bert_model = AutoModel.from_pretrained(bert_model_name_or_path)
        dim_bert_last_hidden = list(self.bert_model.parameters())[-1].shape[-1]

        self.output_layer = OutputLayer(
            dim_bert_last_hidden, n_lbs * n_tasks, uncertainty_method, **kwargs
        ).initialize()

    def forward(self, batch: Batch, **kwargs):
        """
        Forward pass for the ChemBERTa model.

        Args:
            batch (Batch): A batch of data containing atom_ids and attn_masks.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The logits for each label and task.
        """
        bert_hidden = self.bert_model(input_ids=batch.atom_ids, attention_mask=batch.attn_masks)
        bert_features = bert_hidden.pooler_output

        logits = self.output_layer(bert_features)

        return logits
