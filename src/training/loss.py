import torch
import torch.nn as nn


class DPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim

        Computes cross entropy between the query 

        Returns: A 1d torch tensor 
        """
        scores = queries @ documents.T
        labels = torch.arange(queries.size(dim=0), device=queries.device)
        loss = self.cross_entropy(scores, labels)
        return loss