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
        # self.norm_list = [Norm(d_model) for i in range(12)]
        # self.dropout_list = [nn.Dropout(dropout) for i in range(12)]
        
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.norm_4 = Norm(d_model)
        self.norm_5 = Norm(d_model)
        self.norm_6 = Norm(d_model)
        self.norm_7 = Norm(d_model)
        self.norm_8 = Norm(d_model)
        self.norm_9 = Norm(d_model)
        self.norm_10 = Norm(d_model)
        self.norm_11 = Norm(d_model)
        self.norm_12 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        self.dropout_5 = nn.Dropout(dropout)
        self.dropout_6 = nn.Dropout(dropout)
        self.dropout_7 = nn.Dropout(dropout)
        self.dropout_8 = nn.Dropout(dropout)
        self.dropout_9 = nn.Dropout(dropout)
        self.dropout_10 = nn.Dropout(dropout)
        self.dropout_11 = nn.Dropout(dropout)
        self.dropout_12 = nn.Dropout(dropout)
        
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
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_1_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff_1(x2))

            x2 = self.norm_4(x)
            x = x + self.dropout_4(self.attn_2_1(x2, x2, x2, trg_mask))
            x2 = self.norm_5(x)
            x = x + self.dropout_5(self.attn_2_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_6(x)
            x = x + self.dropout_6(self.ff_2(x2))

            x2 = self.norm_7(x)
            x = x + self.dropout_7(self.attn_3_1(x2, x2, x2, trg_mask))
            x2 = self.norm_8(x)
            x = x + self.dropout_8(self.attn_3_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_9(x)
            x = x + self.dropout_9(self.ff_3(x2))

            x2 = self.norm_10(x)
            x = x + self.dropout_10(self.attn_4_1(x2, x2, x2, trg_mask))
            x2 = self.norm_11(x)
            x = x + self.dropout_11(self.attn_4_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_12(x)
            x = x + self.dropout_12(self.ff_4(x2))
        else: 
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
        return x