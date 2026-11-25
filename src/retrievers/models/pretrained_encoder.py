import torch
import torch.nn as nn
from transformers import AutoModel


class PretrainedEncoder(nn.Module):
    def __init__(self, pretrained_name, pooling_type="mean", attn_implementation="sdpa"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name, attn_implementation=attn_implementation)
        self.pooling_type = pooling_type

    def forward(self, input_ids, attention_mask, return_attention=False):
        """Run BERT and pool its token vectors into one vector per text."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=return_attention)
        x = out.last_hidden_state                          

        if self.pooling_type == "cls":
            pooled = x[:, 0]                                  
        else:                                                 
            mask = attention_mask.unsqueeze(-1).float()       
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1)   

        if return_attention:
            return pooled, list(out.attentions)               
        return pooled