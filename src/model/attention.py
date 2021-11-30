import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 p_dropout: float):
        """
        :param d_model: model dimension
        :param n_heads: number of heads
        :param p_dropout: dropout probability (in calculating energy)
        """
        super().__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.scale = torch.tensor(self.d_k ** (-0.5))

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, mask=None):
        """
        :param x: input sequence of shape [B, N, d_model] (we only need one sequence for self-attention)
        :param mask: boolean padding mask of shape [B, N] (True where to mask)
        :return: sequence of the same shape as x after self-attention
        """
        bsize = x.size(0)
        seq_len = x.size(1)

        q = self.Q(x).view(bsize, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.K(x).view(bsize, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.V(x).view(bsize, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # we transpose to get [B, n_heads, seq_len, d_k] so that BMM returns [B, n_heads, seq_len, seq_len]
        # we also use boolean mask here (-inf on masked positions)

        energy = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            energy.masked_fill_(mask[:, None, None, :], float('-inf'))

        weights = F.softmax(energy, dim=-1)
        attention = torch.matmul(self.drop(weights), v)\
                         .transpose(1, 2)\
                         .reshape(bsize, seq_len, self.d_model)

        return self.out(attention)
