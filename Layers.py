import torch
import torch.nn as nn
from Sublayers import FeedForward, MultiHeadAttention, Norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, gpt_inspired_model, dropout=0.1):
        super().__init__()
        self.gpt_inspired_model = gpt_inspired_model
        self.norm_list = [Norm(d_model) for i in range(12)]
        self.dropout_list = [nn.Dropout(dropout) for i in range(12)]
        
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        if gpt_inspired_model == True:
            self.attn_1_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.attn_1_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.ff_1 = FeedForward(d_model, dropout=dropout)
            self.attn_2_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.attn_2_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.ff_2 = FeedForward(d_model, dropout=dropout)
            self.attn_3_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.attn_3_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.ff_3 = FeedForward(d_model, dropout=dropout)
            self.attn_4_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.attn_4_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.ff_4 = FeedForward(d_model, dropout=dropout) 
        else: 
            self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        if self.gpt_inspired_model == True:
            x2 = self.norm_list[0](x)
            x = x + self.dropout_list[0](self.attn_1_1(x2, x2, x2, trg_mask))
            x2 = self.norm_list[1](x)
            x = x + self.dropout_list[1](self.attn_1_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_list[2](x)
            x = x + self.dropout_list[2](self.ff_1(x2))

            x2 = self.norm_list[3](x)
            x = x + self.dropout_list[3](self.attn_2_1(x2, x2, x2, trg_mask))
            x2 = self.norm_list[4](x)
            x = x + self.dropout_list[4](self.attn_2_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_list[5](x)
            x = x + self.dropout_list[5](self.ff_2(x2))

            x2 = self.norm_list[6](x)
            x = x + self.dropout_list[6](self.attn_3_1(x2, x2, x2, trg_mask))
            x2 = self.norm_list[7](x)
            x = x + self.dropout_list[7](self.attn_3_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_list[8](x)
            x = x + self.dropout_list[8](self.ff_3(x2))

            x2 = self.norm_list[9](x)
            x = x + self.dropout_list[9](self.attn_4_1(x2, x2, x2, trg_mask))
            x2 = self.norm_list[10](x)
            x = x + self.dropout_list[10](self.attn_4_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_list[11](x)
            x = x + self.dropout_list[11](self.ff_4(x2))
        else: 
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
        return x