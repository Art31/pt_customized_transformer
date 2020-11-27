import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import torch.nn.functional as F
import copy

def get_clones(module, N, decoder_extra_layers=None):
    if decoder_extra_layers is None:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N+decoder_extra_layers)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src) # [128, 5]
        x = self.pe(x) # [128, 5, 512]
        for i in range(self.N):
            x = self.layers[i](x, mask) # [128, 5, 512], [128, 1, 5]
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, decoder_extra_layers):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, decoder_extra_layers, dropout), N, decoder_extra_layers)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, decoder_extra_layers):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, decoder_extra_layers)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask) # [128, 5], [128, 1, 5] -> [128, 5, 512]
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask) # [128, 7], [128, 5, 512], [128, 1, 5], [128, 7, 7] -> [128, 7, 512]
        output = self.out(d_output) # [128, 7, 512] -> [128, 7, 11441])
        return output

class EncoderRNN(nn.Module):
    def __init__(self, input_size, d_model, device):
        # positional encoding is unnecessary for RNNs!
        super(EncoderRNN, self).__init__()
        self.device = device
        self.d_model = d_model

        self.embedding = nn.Embedding(input_size, d_model)
        self.rnn = nn.RNN(d_model, d_model)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.d_model, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, d_model, output_size, device):
        # positional encoding is unnecessary for RNNs!
        super(DecoderRNN, self).__init__()
        self.device = device
        self.d_model = d_model

        self.embedding = nn.Embedding(output_size, d_model)
        self.rnn = nn.RNN(d_model, d_model)
        self.out = nn.Linear(d_model, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.d_model, device=self.device)

class NaiveModel(nn.Module):
    # hidden_size = d_model !
    # entender se precisa colocar input_lang.n_words
    # encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    def __init__(self, src_vocab, trg_vocab, d_model, dropout, device, max_length):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        self.device = device
        self.encoder = EncoderRNN(src_vocab, d_model, device)
        self.decoder = DecoderRNN(d_model, trg_vocab, device)
    def forward(self, src, trg):
        encoder_hidden = self.encoder.initHidden()
        input_length = src.size(0)
        target_length = trg.size(0)
        encoder_outputs = torch.zeros(self.max_length, self.encoder.d_model, device=self.device)
        for ei in range(input_length):
            encoder_output, enc_hidden = self.encoder(src[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_hidden = enc_hidden
        decoder_input = torch.tensor([[SOS_token]], device=device)
        ##### naive encoder #####
        for di in range(target_length):
            d_output, dec_hidden = self.decoder(trg, decoder_hidden)

        ##### encoder with attention #####
        for di in range(target_length):
            d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        
        return d_output 

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)

#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))

#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)

#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)

#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    if opt.naive_model_type == 0:
        model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout, opt.decoder_extra_layers)
    else: 
        model = NaiveModel(src_vocab, trg_vocab, opt.d_model, opt.dropout, opt.device, opt.max_strlen)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        # model = nn.DataParallel(model)
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.no_cuda is False:
        model = model.cuda()
    
    return model
    
