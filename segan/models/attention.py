import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.nn.utils as nnu
import math


class Attn(nn.Module):
    def __init__(self, hidden_size, cuda=False, snorm=False):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.do_cuda = cuda

        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.randn(1, hidden_size))
        if snorm:
            nnu.spectral_norm(self.attn)
            nnu.spectral_norm(self, name='v')

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        #print('[attn] seq len', seq_len)
        #print('[attn] encoder_outputs', encoder_outputs.size()) # B x S x N
        #print('[attn] hidden', hidden.size()) # B x S=1 x N

        # Create variable to store attention energies
        attn_energies = torch.zeros(batch_size, seq_len) # B x S

        if self.do_cuda:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(seq_len):
                attn_energies[b, i] = self.score(hidden[b, :], encoder_outputs[b, i].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        #print('[attn] attn_energies', attn_energies.size())
        #print('[attn] energies: ', attn_energies)
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        energy = self.attn(torch.cat((hidden, encoder_output), 1))
        #print('energy: ', energy)
        energy = torch.bmm(self.v.unsqueeze(1), 
                           energy.unsqueeze(2))
        return energy

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    #print(scores.size())
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
