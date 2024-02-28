"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: ChemBERTa model.

This module implements the ChemBERTa model which is tailored for chemical informatics tasks. 
It leverages the pretrained BERT architecture from the HuggingFace `transformers` library 
and adds an output layer tailored for multi-label and multi-task prediction.
"""

import torch.nn as nn
from transformers import AutoModel

from muben.dataset import Batch
from muben.layers import OutputLayer


class LinearTransformer(nn.Module):
    """
    The ChemBERTa model, a modified version of BERT for chemical informatics tasks.

    Attributes
    ----------
    bert_model : transformers.PreTrainedModel
        The pretrained BERT model.
    output_layer : OutputLayer
        The output layer for multi-label and multi-task classification.

    Args:
    """

    def __init__(self, config, **kwargs):
        """
        Initialize the Huggingface model with Linear Sequential Input.

        Parameters
        ----------
        bert_model_name_or_path : str, required
            Identifier or path for the BERT model to be loaded.
        n_lbs : int, required
            Number of labels per task.
        n_tasks : int, required
            Number of tasks.
        uncertainty_method : str, required
            Method for modeling uncertainty.
        task_type : str, required
            Type of task (classification or regression), used to initialize the output layer.
        bbp_prior_sigma : float, optional
            The prior sigma for BBP layers when BBP is adopted as the uncertainty estimator.
        """
        super().__init__()

        bert_model_name_or_path = config.pretrained_model_name_or_path
        n_lbs = config.n_lbs
        n_tasks = config.n_tasks
        uncertainty_method = config.uncertainty_method
        task_type = config.task_type
        bbp_prior_sigma = config.bbp_prior_sigma

        self.bert_model = AutoModel.from_pretrained(bert_model_name_or_path)
        dim_bert_last_hidden = list(self.bert_model.parameters())[-1].shape[-1]

        self.output_layer = OutputLayer(
            dim_bert_last_hidden,
            n_lbs * n_tasks,
            uncertainty_method,
            task_type=task_type,
            bbp_prior_sigma=bbp_prior_sigma,
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
