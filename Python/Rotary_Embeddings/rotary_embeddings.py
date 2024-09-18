import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowGroupedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, group_size, max_seq_len=512):
        super(SlidingWindowGroupedAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.group_size = group_size
        self.max_seq_len = max_seq_len
        
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        
        self.out_projection = nn.Linear(embed_dim, embed_dim)

        # Rotary Embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()
        
        # Project queries, keys, and values
        query = self.query_projection(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        key = self.key_projection(key).view(batch_size, key_len, self.num_heads, self.head_dim)
        value = self.value_projection(value).view(batch_size, key_len, self.num_heads, self.head_dim)
        
        # Transpose to get dimensions (batch_size, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply Rotary Embeddings
        query = self.rotary_emb(query, query_len)
        key = self.rotary_emb(key, key_len)

        # Initialize attention scores with zeros
        attention_scores = torch.zeros(batch_size, self.num_heads, query_len, self.window_size).to(query.device)
        
        # Calculate attention scores for each group in a sliding window fashion
        for idx in range(0, query_len, self.group_size):
            end_idx = min(idx + self.window_size, key_len)
            scores = torch.matmul(query[:, :, idx:idx+self.group_size], key[:, :, idx:end_idx].transpose(-2, -1))
            scores = scores / (self.head_dim ** 0.5)  # Scale scores
            attention_scores[:, :, idx:idx+self.group_size, :end_idx-idx] = scores
        
        # Apply mask if provided (mask should match the attention_scores shape)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Calculate attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value[:, :, :self.window_size])
        
        # Concatenate heads and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        output = self.out_projection(context)
        
        return output

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, seq_len):
        batch_size, num_heads, seq_len, head_dim = x.size()

        # Precompute the sinusoidal positional embeddings
        pos = torch.arange(seq_len).unsqueeze(1)
        inv_freq = 1. / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pe = torch.zeros(seq_len, head_dim, dtype=torch.float)
        pe[:, 0::2] = torch.sin(pos * inv_freq)
        pe[:, 1::2] = torch.cos(pos * inv_freq)
        pe = pe.to(x.device)

        # Apply rotary embeddings
        x_even = x[:, :, :, 0::2]
        x_odd = x[:, :, :, 1::2]
        x_even_rotated = x_even * pe[:, :seq_len, 0::2] + x_odd * pe[:, :seq_len, 1::2]
        x_odd_rotated = -x_even * pe[:, :seq_len, 1::2] + x_odd * pe[:, :seq_len, 0::2]
        x = torch.cat((x_even_rotated, x_odd_rotated), dim=-1)
        return x