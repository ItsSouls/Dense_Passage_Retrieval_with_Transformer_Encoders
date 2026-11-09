import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import math

# 1. attention function    
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -math.inf)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# 2. Multi-head attention         
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None # store the attention maps
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)          

# 3. Feed-forward                 
class FeedForward(nn.Module):
    def __init__(self, emb_dim, feed_forward_dim, dropout): 
        super(FeedForward, self).__init__()  
        self.linears = nn.ModuleList([nn.Linear(emb_dim,feed_forward_dim), nn.Linear(feed_forward_dim,emb_dim)])
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linears[0](x)
        x = self.activation(x)
        x = self.linears[1](x)
        x = self.dropout(x)
        return x

# 4. Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, feed_forward_dim, dropout):
        super().__init__()
        self.attention =  MultiHeadedAttention(h=n_heads, d_model=emb_dim, dropout=dropout)
        self.ffnn = FeedForward(emb_dim = emb_dim, feed_forward_dim = feed_forward_dim, dropout = dropout)
        self.norm_attention = nn.LayerNorm(emb_dim)
        self.norm_ffnn = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, mask): 
        a = self.attention(x, x, x, mask)
        a = self.dropout(a)
        x1 = self.norm_attention(x + a)
        f = self.ffnn(x1)
        y = self.norm_ffnn(x1 + f)
        return y

# 5. Positional Encoding
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, emb_dim, dropout, max_length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_length, emb_dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, pooling_type, vocab_size, max_length, dropout = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.pooling_type = pooling_type
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dropout = dropout
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.token_embeddings = torch.nn.Embedding(self.vocab_size, self.emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim=self.emb_dim, dropout=self.dropout, max_length=self.max_length)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(emb_dim=self.emb_dim, n_heads=self.n_heads, feed_forward_dim=self.feed_forward_dim, dropout=self.dropout) for _ in range(self.n_layers)])
    
    def pool(self, x, attention_mask):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.pooling_type == 'cls':
            return x[:, 0, :]
        
        elif self.pooling_type == 'max':
            mask = attention_mask.unsqueeze(-1).float() 
            x_masked = x.masked_fill(mask == 0, float('-inf'))
            return torch.max(x_masked, dim=1)[0]
        
        elif self.pooling_type == 'mean':
            mask = attention_mask.unsqueeze(-1).float()      
            summed = (x * mask).sum(dim=1)                  
            n_real = mask.sum(dim=1)           
            return summed / n_real  
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

    #def forward(self, x: torch.tensor) -> torch.tensor:
    def forward(self, input_ids, attention_mask, return_attention=False):
        """
        Run a forward pass on the model
        """
        x = self.token_embeddings(input_ids)    
        x = self.positional_encoding(x)

        attentions = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, attention_mask)
            attentions.append(encoder_block.attention.attn)

        pooled_output = self.pool(x, attention_mask)

        if return_attention:                       # ← 2. devolver los pesos solo si se piden
            return pooled_output, attentions
        return pooled_output