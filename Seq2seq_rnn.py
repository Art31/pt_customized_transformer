import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import dill as pickle

from torchtext import data

from tqdm import tqdm 
import pandas as pd
import random, os
import math, time

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 emb_dim: int, 
                 hid_dim: int, 
                 field, 
                 word_emb_model, 
                 opt):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.opt = opt
        
        if opt.word_embedding_type is None:
            # self.embedding = nn.Embedding(input_size, d_model)
            self.embedding = nn.Embedding(input_dim, emb_dim) #no dropout as only one layer!
        else:
            word_embeddings = torch.FloatTensor(field.vocab.vectors)
            self.embedding = nn.Embedding.from_pretrained(word_embeddings) # https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings
        
        if opt.nmt_model_type == 'rnn_naive_model':
            self.rnn = nn.GRU(emb_dim, hid_dim)
            self.dropout = nn.Dropout(dropout)
        # elif opt.nmt_model_type == 'align_and_translate':
        #     self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional = True)
        #     self.fc = nn.Linear(d_model * 2, d_model)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src sent len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded) #no cell state!
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        if self.opt.nmt_model_type == 'rnn_naive_model':
            return hidden 
        # elif self.opt.nmt_model_type == 'align_and_translate':
        #     hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        #     return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim: int, 
                 emb_dim: int, 
                 hid_dim: int, 
                 field, 
                 word_emb_model, 
                 opt, 
                 attention=None):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        
        if opt.word_embedding_type is None:
            # self.embedding = nn.Embedding(output_size, d_model)
            self.embedding = nn.Embedding(output_dim, emb_dim)
        else:
            word_embeddings = torch.FloatTensor(field.vocab.vectors)
            self.embedding = nn.Embedding.from_pretrained(word_embeddings) # https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings
        
        if self.opt.nmt_model_type == 'rnn_naive_model':
            self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
            self.out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
            self.dropout = nn.Dropout(dropout)
        # elif self.opt.nmt_model_type == 'align_and_translate':
        
    def forward(self, 
                input: Tensor, 
                hidden: Tensor, 
                context: Tensor):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]
        
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        if self.opt.nmt_model_type == 'rnn_naive_model':
            #embedded = [1, batch size, emb dim]
                    
            emb_con = torch.cat((embedded, context), dim = 2)
                
            #emb_con = [1, batch size, emb dim + hid dim]
                
            output, hidden = self.rnn(emb_con, hidden)
            
            #output = [sent len, batch size, hid dim * n directions]
            #hidden = [n layers * n directions, batch size, hid dim]
            
            #sent len, n layers and n directions will always be 1 in the decoder, therefore:
            #output = [1, batch size, hid dim]
            #hidden = [1, batch size, hid dim]
            
            output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), 
                            dim = 1)
            
            #output = [batch size, emb dim + hid dim * 2]
            
            prediction = self.out(output)
            
            #prediction = [batch size, output dim]
            
            return prediction, hidden
        # elif self.opt.nmt_model_type == 'align_and_translate':

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 device: torch.device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, 
                src: Tensor, 
                trg: Tensor, 
                teacher_forcing_ratio = 0):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is the context
        context = self.encoder(src)
        
        #context also used as the initial hidden state of the decoder
        hidden = context
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden = self.decoder(input, hidden, context) # input and hidden are new, context is fixed
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs