import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -math.inf)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """A simple Multi-head attention layer."""
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        y = self.norm1(x)
        x = x + self.dropout(self.self_attn(y, y, y, mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.max_seq_length = max_seq_length
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        self.decoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    @staticmethod
    def generate_mask(x):
        T = x.size(1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))  # (T, T)
        key_keep = (x != -1).unsqueeze(1)                                         # (batch, 1, T)
        return causal.unsqueeze(0) & key_keep

    def forward(self, x):
        assert x.size(1) <= self.max_seq_length, "sequence longer than max_seq_length"
        mask = self.generate_mask(x)
        x = x.masked_fill(x == -1, 0)
        pos = torch.arange(x.size(1), device=x.device)
        h = self.dropout(self.decoder_embedding(x) + self.pos_embedding(pos))
        for layer in self.decoder_layers:
            h = layer(h, mask)
        return self.fc(self.norm(h))
