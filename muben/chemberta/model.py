import torch.nn as nn
from transformers import AutoModel

from muben.base.dataset import Batch
from muben.base.model import OutputLayer


class ChemBERTa(nn.Module):

    def __init__(self,
                 bert_model_name_or_path: str,
                 n_lbs: int,
                 n_tasks: int,
                 apply_bbp: bool = False,
                 **kwargs):

        super().__init__()

        self.bert_model = AutoModel.from_pretrained(bert_model_name_or_path)
        dim_bert_last_hidden = list(self.bert_model.parameters())[-1].shape[-1]

        self.output_layer = OutputLayer(
            dim_bert_last_hidden,
            n_lbs * n_tasks,
            apply_bbp,
            **kwargs
        ).initialize()

    def forward(self, batch: Batch, **kwargs):
        bert_hidden = self.bert_model(input_ids=batch.atom_ids, attention_mask=batch.attn_masks)
        bert_features = bert_hidden.pooler_output

        logits = self.output_layer(bert_features)

        return logits
