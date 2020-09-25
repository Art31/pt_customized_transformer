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
    def __init__(self, d_model, heads, decoder_extra_layers, dropout=0.1):
        super().__init__()
        self.decoder_extra_layers = decoder_extra_layers
        # self.norm_list = [Norm(d_model) for i in range(12)]
        # self.dropout_list = [nn.Dropout(dropout) for i in range(12)]
        
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.norm_4 = Norm(d_model)
        self.norm_5 = Norm(d_model)
        self.norm_6 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        self.dropout_5 = nn.Dropout(dropout)
        self.dropout_6 = nn.Dropout(dropout)
        
        # if decoder_extra_layers == True:
        #     self.attn_1_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        #     self.attn_1_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        #     self.ff_1 = FeedForward(d_model, dropout=dropout)
        #     self.attn_2_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        #     self.attn_2_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        #     self.ff_2 = FeedForward(d_model, dropout=dropout)
        # else: 
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        # if self.decoder_extra_layers == True:
        #     x2 = self.norm_1(x)
        #     x = x + self.dropout_1(self.attn_1_1(x2, x2, x2, trg_mask))
        #     x2 = self.norm_2(x)
        #     x = x + self.dropout_2(self.attn_1_2(x2, e_outputs, e_outputs, \
        #     src_mask))
        #     x2 = self.norm_3(x)
        #     x = x + self.dropout_3(self.ff_1(x2))

        #     x2 = self.norm_4(x)
        #     x = x + self.dropout_4(self.attn_2_1(x2, x2, x2, trg_mask))
        #     x2 = self.norm_5(x)
        #     x = x + self.dropout_5(self.attn_2_2(x2, e_outputs, e_outputs, \
        #     src_mask))
        #     x2 = self.norm_6(x)
        #     x = x + self.dropout_6(self.ff_2(x2))
        # else: 
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x