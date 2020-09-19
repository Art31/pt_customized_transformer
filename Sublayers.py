import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, heads, d_model, dropout=.1):
#         d_k = d_model // heads
#         super(ScaledDotProductAttention, self).__init__()
#         self.scale_factor = np.sqrt(d_k)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, q, k, v, attn_mask=None):
#         # q: [b_size x n_heads x len_q x d_k]
#         # k: [b_size x n_heads x len_k x d_k]
#         # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

#         # attn: [b_size x n_heads x len_q x len_k]
#         scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
#         if attn_mask is not None:
#             assert attn_mask.size() == scores.size()
#             scores.masked_fill_(attn_mask.cuda(), -1e9)
#         attn = self.dropout(self.softmax(scores))

#         # outputs: [b_size x n_heads x len_q x d_v]
#         context = torch.matmul(attn, v)

#         return context, attn

# class MultiBranchAttentionBase(nn.Module): #  same as multi head attention
#     # def __init__(self, d_k, d_v, d_model, n_heads, dropout):
#     def __init__(self, heads, d_model, dropout = 0.1):
#         super(MultiBranchAttentionBase, self).__init__()
#         d_k = d_model // heads
#         self.d_k = d_model // heads
#         d_v = d_k
#         self.d_v = d_model // heads
#         self.d_model = d_model
#         self.heads = heads

#         self.w_q = Linear(d_model, d_k * heads)
#         self.w_k = Linear(d_model, d_k * heads)
#         self.w_v = Linear(d_model, d_v * heads)

#         self.attention = ScaledDotProductAttention(d_k, dropout)

#     def forward(self, q, k, v, attn_mask=None):
#         # q: [b_size x len_q x d_model]
#         # k: [b_size x len_k x d_model]
#         # v: [b_size x len_k x d_model]
#         b_size = q.size(0)

#         # q_s: [b_size x heads x len_q x d_k]
#         # k_s: [b_size x heads x len_k x d_k]
#         # v_s: [b_size x heads x len_k x d_v]
#         q_s = self.w_q(q).view(b_size, -1, self.heads, self.d_k).transpose(1, 2)
#         k_s = self.w_k(k).view(b_size, -1, self.heads, self.d_k).transpose(1, 2)
#         v_s = self.w_v(v).view(b_size, -1, self.heads, self.d_v).transpose(1, 2)

#         if attn_mask != None:  # attn_mask: [b_size x len_q x len_k]
#             attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
#         # context: [b_size x heads x len_q x d_v], attn: [b_size x heads x len_q x len_k]
#         context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
#         # context: [b_size x len_q x heads * d_v]
#         context = context.transpose(1, 2).contiguous().view(b_size, -1, self.heads * self.d_v)

#         # return the context and attention weights
#         return context, attn

# class MultiBranchAttention(nn.Module):
#     def __init__(self, heads, d_model, d_ff, n_branches, dropout):
#         super(MultiBranchAttention, self).__init__()
#         d_k = d_model // heads
#         d_v = d_k
#         self.d_k = d_k
#         self.d_v = d_v
#         self.d_model = d_model
#         self.d_ff = d_ff
#         self.n_branches = n_branches

#         self.multihead_attn = MultiBranchAttentionBase(d_k, d_v, d_model, n_branches, dropout)
#         # additional parameters for BranchedAttention
#         self.w_o = nn.ModuleList([Linear(d_v, d_model) for _ in range(n_branches)])
#         self.w_kp = torch.rand(n_branches)
#         self.w_kp = nn.Parameter(self.w_kp/self.w_kp.sum())
#         self.w_a = torch.rand(n_branches)
#         self.w_a = nn.Parameter(self.w_a/self.w_a.sum())

#         self.pos_ffn = nn.ModuleList([
#             PoswiseFeedForwardNet(d_model, d_ff//n_branches, dropout) for _ in range(n_branches)])
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = LayerNormalization(d_model)

#         init.xavier_normal(self.w_o)

#     def forward(self, q, k, v, attn_mask):
#         # q: [b_size x len_q x d_model]
#         # k: [b_size x len_k x d_model]
#         # v: [b_size x len_v x d_model] note (len_k == len_v)
#         residual = q

#         # context: a tensor of shape [b_size x len_q x n_branches * d_v]
#         context, attn = self.multih_attn(q, k, v, attn_mask=attn_mask)

#         # context: a list of tensors of shape [b_size x len_q x d_v] len: n_branches
#         context = context.split(self.d_v, dim=-1)

#         # outputs: a list of tensors of shape [b_size x len_q x d_model] len: n_branches
#         outputs = [self.w_o[i](context[i]) for i in range(self.n_branches)]
#         outputs = [kappa * output for kappa, output in zip(self.w_kp, outputs)]
#         outputs = [pos_ffn(output) for pos_ffn, output in zip(self.pos_ffn, outputs)]
#         outputs = [alpha * output for alpha, output in zip(self.w_a, outputs)]

#         # output: [b_size x len_q x d_model]
#         output = self.dropout(torch.stack(outputs).sum(dim=0))
#         return self.layer_norm(residual + output), attn