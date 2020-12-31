import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
from gensim.models import KeyedVectors
import torch.nn.functional as F
import copy, time, math

def get_clones(module, N, decoder_extra_layers=None):
    if decoder_extra_layers is None:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N+decoder_extra_layers)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, field, word_emb, opt):
        super().__init__()
        self.N = N
        self.word_emb = word_emb; self.opt = opt # unused, just for querying
        self.embed = Embedder(vocab_size, d_model, word_emb, field)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N) # attention
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src) # [128, 5]
        x = self.pe(x) # [128, 5, 512]
        for i in range(self.N):
            x = self.layers[i](x, mask) # [128, 5, 512], [128, 1, 5]
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, decoder_extra_layers, field, word_emb, opt):
        super().__init__()
        self.N = N
        self.opt = opt; self.word_emb = word_emb # unused, just for querying
        self.embed = Embedder(vocab_size, d_model, word_emb, field)
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
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, decoder_extra_layers, fields, word_emb, opt):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout, fields['SRC'], word_emb, opt)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, decoder_extra_layers, fields['TRG'], word_emb, opt)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask) # [128, 5], [128, 1, 5] -> [128, 5, 512]
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask) # [128, 7], [128, 5, 512], [128, 1, 5], [128, 7, 7] -> [128, 7, 512]
        # d_output = self.decoder(trg, torch.randn(e_outputs.shape), torch.randn(src_mask.shape), trg_mask) # [128, 7], [128, 5, 512], [128, 1, 5], [128, 7, 7] -> [128, 7, 512]
        output = self.out(d_output) # [128, 7, 512] -> [128, 7, 11441])
        return output

class EncoderRNN(nn.Module):
    def __init__(self, input_size, d_model, field, word_emb_model, opt): # emb_dim = hid_dim = d_model
        # positional encoding is unnecessary for RNNs!
        super(EncoderRNN, self).__init__()
        # self.device = device
        self.d_model = d_model

        if opt.word_embedding_type is None:
            self.embedding = nn.Embedding(input_size, d_model)
        else:
            word_embeddings = torch.FloatTensor(field.vocab.vectors)
            self.embedding = nn.Embedding.from_pretrained(word_embeddings) # https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings
        self.rnn = nn.RNN(d_model, d_model)

    # def forward(self, input, hidden): # no initialization for hidden layer in first paper
    def forward(self, input):
        #input = [input_len, batch_size]
        embedded = self.embedding(input)#.view(1, 1, -1)
        #embedded = [input_len, batch_size, d_model]
        output = embedded
        # output, hidden = self.rnn(output, hidden) # no initialization for hidden layer in first paper
        output, hidden = self.rnn(output) #no cell state!
        #outputs = [input_len, batch_size, d_model * n directions]
        #hidden = [n_layers * n directions, batch_size, d_model]
        # return output, hidden # return only context (hidden state) in first paper
        return hidden 

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.d_model, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, d_model, output_size, field, word_emb_model, opt): # emb_dim = hid_dim = d_model
        # positional encoding is unnecessary for RNNs!
        super(DecoderRNN, self).__init__()
        # self.device = device
        self.output_size = output_size
        self.d_model = d_model

        if opt.word_embedding_type is None:
            self.embedding = nn.Embedding(output_size, d_model)
        else:
            word_embeddings = torch.FloatTensor(field.vocab.vectors)
            self.embedding = nn.Embedding.from_pretrained(word_embeddings) # https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings
        self.rnn = nn.RNN(d_model + d_model, d_model)
        self.fc_out = nn.Linear(d_model + d_model * 2, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    # def forward(self, input, hidden): # context vector is passed in first paper
    def forward(self, input, hidden, context):
        # output = self.embedding(input).view(1, 1, -1) # original application
        # output = F.relu(output) # original application
        # # output, hidden = self.rnn(output, hidden) # original application
        # output = self.softmax(self.out(output[0])) # original application
        
        #input = [batch_size]
        #hidden = [n_layers * n_directions, batch_size, d_model]
        #context = [n_layers * n_directions, batch_size, d_model]
        
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch_size, d_model]
        #context = [1, batch_size, d_model]
        input = input.unsqueeze(0)
        #input = [1, batch_size]
        embedded = self.embedding(input)
        #embedded = [1, batch_size, d_model]
                
        emb_con = torch.cat((embedded, context), dim = 2)
            
        #emb_con = [1, batch_size, d_model + d_model]
            
        output, hidden = self.rnn(emb_con, hidden)

        #output = [seq len, batch_size, d_model * n_directions]
        #hidden = [n_layers * n_directions, batch_size, d_model]
        
        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch_size, d_model]
        #hidden = [1, batch_size, d_model]
        
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), 
                           dim = 1)
        
        #output = [batch_size, d_model + d_model * 2]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch_size, output dim]
        return prediction, hidden

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.d_model, device=self.device)

# usar https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb
class NaiveModel(nn.Module):
    # hidden_size = d_model !
    # entender se precisa colocar input_lang.n_words
    # encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    # def __init__(self, src_vocab, trg_vocab, d_model, dropout, device, max_length):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # encoder_hidden = self.encoder.initHidden()
        # input_length = src.size(0)
        # target_length = trg.size(0)
        # encoder_outputs = torch.zeros(self.max_length, self.d_model, device=self.device)
        # for ei in range(input_length):
        #     encoder_output, enc_hidden = self.encoder(src[ei], encoder_hidden)
        #     encoder_outputs[ei] = encoder_output[0, 0]

        # decoder_hidden = enc_hidden
        # decoder_input = torch.tensor(opt.trg_sos, device=device)
        # ##### naive encoder #####
        # for di in range(target_length):
        #     d_output, dec_hidden = self.decoder(trg, decoder_hidden)

        # ##### encoder with attention #####
        # for di in range(target_length):
        #     d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is the context
        context = self.encoder(src)
        
        #context also used as the initial hidden state of the decoder
        hidden = context
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            # teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            # input = trg[t] if teacher_force else top1
            input = top1
        
        return outputs 

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

def get_model(opt, src_vocab, trg_vocab, word_emb):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    if opt.naive_model_type == 'transformer':
        fields = {'SRC': opt.SRC, 'TRG': opt.TRG}
        model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout, opt.decoder_extra_layers, fields, word_emb, opt)
    elif opt.naive_model_type == 'rnn_naive_model': 
        encoder = EncoderRNN(src_vocab, opt.d_model, opt.SRC, word_emb, opt)
        decoder = DecoderRNN(opt.d_model, trg_vocab, opt.TRG, word_emb, opt)
        model = NaiveModel(encoder, decoder, opt.device) # (opt.d_model, opt.dropout, opt.device, opt.max_strlen)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        # model = nn.DataParallel(model)
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    elif opt.word_embedding_type is None:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.no_cuda is False:
        model = model.cuda()
    
    return model
    
