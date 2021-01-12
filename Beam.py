import torch, random
from Batch import nopeak_mask
import torch.nn.functional as F
import math


def init_vars(src, model, SRC, TRG, opt):
    
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)

    if opt.nmt_model_type == 'transformer':
        outputs = torch.LongTensor([[init_tok]], device=opt.device)
        e_output = model.encoder(src, src_mask) # [[1, 7], [1, 1, 7]] -> [1, 7, 300 (d_model)]
        trg_mask = nopeak_mask(1, opt)
        out = model.out(model.decoder(outputs,
                e_output, src_mask, trg_mask)) # [[1, 1], [1, 7, 300]] -> [1, 1, 300] -> [1, 1, 11436 (out_features)]
        out = F.softmax(out, dim=-1) # [1, 1, 11436])
    elif opt.nmt_model_type == 'rnn_naive_model':
        outputs = torch.LongTensor([init_tok for i in range(src.shape[1])], device=opt.device)
        e_output = model.encoder(src) # [1, 7] -> [1, 7, 300 (d_model)]
        encoder_hidden = e_output
        out, decoder_hidden = model.decoder(
                outputs.unsqueeze(0), encoder_hidden, e_output)
        out = out.unsqueeze(0) # [7, 11436] -> [1, 7, 11436]
    # elif opt.nmt_model_type == 'allign_and_translate':
    #     e_output, hidden = model.encoder(src)
    
    
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_len).long()
    if opt.no_cuda is False:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))
    if opt.no_cuda is False:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(src, model, SRC, TRG, opt):
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    if opt.nmt_model_type == 'rnn_naive_model':
        encoder_outputs = model.encoder(src)
        encoder_hidden = encoder_outputs
        decoder_hidden = encoder_hidden

    for i in range(2, opt.max_len):
    
        if opt.nmt_model_type == 'transformer': # keep increasing size of sentence
            trg_mask = nopeak_mask(i, opt)
            out = model.out(model.decoder(outputs[:,:i],
                    e_outputs, src_mask, trg_mask)) # [[3, 2], [3, 7, 300], [1, 1, 7], [1, 2, 2]] -> [3, 2, 300] -> [3, 2, 11436]
            out = F.softmax(out, dim=-1)
        # elif opt.nmt_model_type == 'rnn_naive_model':
            # for j in range(outputs[:,:i].shape[0]):
            #     input_tensor = torch.cat([outputs[j,:i].unsqueeze(0)]*src.shape[1], dim=0) # outputs[:,:i].shape = [3, 2]
                # input_tensor = torch.LongTensor([outputs[:,:i] for i in range(src.shape[1])]).to(opt.device) # outputs[:,:i].shape = [3, 2]
            # out, decoder_hidden = model.decoder(
            #         outputs[:,:i].unsqueeze(0), decoder_hidden, encoder_outputs) # [[3, 2], [1, 7, 300], [1, 7, 300]] # ENTENDER PQ ISSO NAO FUNCIONA !!!! -> [1, 7, 3, 2, 300]
                # best_guess = decoder_output.argmax() % decoder_output.shape[1]
                # best_guess = best_guess.item()
                # outputs.append(best_guess)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k) # (torch.Size([3, 100]), torch.Size([3, 2, 11436]))
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long, device=opt.device)
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        try:
            length = (outputs[0]==eos_tok).nonzero()[0]
        except:
            import ipdb; ipdb.set_trace()
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

def generate_rnn_translations(src, model, TRG, opt):
    max_length = opt.max_len
    with torch.no_grad():
        input_length = src.size()[0]

        if opt.nmt_model_type == 'rnn_naive_model':
            encoder_outputs = model.encoder(src)
            encoder_hidden = encoder_outputs
        elif opt.nmt_model_type == 'allign_and_translate':
            encoder_outputs, encoder_hidden = model.encoder(src)

        decoder_hidden = encoder_hidden
        outputs = [TRG.vocab.stoi["<sos>"]]
        word_outputs = ["<sos>"]

        # query input words: [opt.SRC.vocab.itos[ind] for ind in src[0, :]]
        for _ in range(max_length):
            # previous_word = torch.LongTensor([outputs[-1]]).to(opt.device)
            previous_word = torch.LongTensor([[outputs[-1] for i in range(src.size()[1])]]).to(opt.device)
            # OPTION 1 - input one word but has to filter the encoder info (not desired)
            # decoder_output, decoder_hidden = model.decoder(
            #     previous_word, decoder_hidden[:, 0, :].unsqueeze(0), encoder_outputs[:, 0, :].unsqueeze(0)) # [[1, 1], [1, 7, 300], [1, 7, 300]]
            # OPTION 2 - input one word times input size but has to create another way of choosing the best translation
            decoder_output, decoder_hidden = model.decoder(
                previous_word, decoder_hidden, encoder_outputs)
            # pytorch seq2seq desired [[1, 1], [1, 1, 256], [10, 256]] -> [[1, 2711], [1, 1, 256], [1, 10]]
            # rnn naive model 
            # topv, topi = decoder_output.data.topk(1)
            # best_guess = decoder_output.argmax(1).item() # old
            best_guess = decoder_output.argmax() % decoder_output.shape[1]
            best_guess = best_guess.item()
            outputs.append(best_guess)
            if best_guess == TRG.vocab.stoi['<eos>']:
                word_outputs.append('<eos>')
                break
            else:
                word_outputs.append(TRG.vocab.itos[best_guess])

            # previous_word = topi.squeeze().detach()

        return ' '.join(word_outputs)

# import operator
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from queue import PriorityQueue

# class BeamSearchNode(object):
#     def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
#         '''
#         :param hiddenstate:
#         :param previousNode:
#         :param wordId:
#         :param logProb:
#         :param length:
#         '''
#         self.h = hiddenstate
#         self.prevNode = previousNode
#         self.wordid = wordId
#         self.logp = logProb
#         self.leng = length

#     def eval(self, alpha=1.0):
#         reward = 0
#         # Add here a function for shaping a reward

#         return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

# def beam_decode(target_tensor, decoder_hiddens, opt, decoder, encoder_outputs=None):
#     '''
#     :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
#     :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
#     :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
#     :return: decoded_batch
#     '''

#     beam_width = opt.k
#     topk = 1  # how many sentence do you want to generate
#     decoded_batch = []

#     # decoding goes sentence by sentence
#     for idx in range(target_tensor.size(0)):
#         if isinstance(decoder_hiddens, tuple):  # LSTM case
#             decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
#         else:
#             decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
#         encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

#         # Start with the start of the sentence token
#         decoder_input = torch.LongTensor([[opt.TRG.vocab.stoi["<sos>"]]], device=opt.device)

#         # Number of sentence to generate
#         endnodes = []
#         number_required = min((topk + 1), topk - len(endnodes))

#         # starting node -  hidden vector, previous node, word id, logp, length
#         node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
#         nodes = PriorityQueue()

#         # start the queue
#         nodes.put((-node.eval(), node))
#         qsize = 1

#         # start beam search
#         while True:
#             # give up when decoding takes too long
#             if qsize > 2000: break

#             # fetch the best node
#             score, n = nodes.get()
#             decoder_input = n.wordid
#             decoder_hidden = n.h

#             if n.wordid.item() == opt.TRG.vocab.stoi["<eos>"] and n.prevNode != None:
#                 endnodes.append((score, n))
#                 # if we reached maximum # of sentences required
#                 if len(endnodes) >= number_required:
#                     break
#                 else:
#                     continue

#             # decode for one step using decoder
#             decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

#             # PUT HERE REAL BEAM SEARCH OF TOP
#             log_prob, indexes = torch.topk(decoder_output, beam_width)
#             nextnodes = []

#             for new_k in range(beam_width):
#                 decoded_t = indexes[0][new_k].view(1, -1)
#                 log_p = log_prob[0][new_k].item()

#                 node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
#                 score = -node.eval()
#                 nextnodes.append((score, node))

#             # put them into queue
#             for i in range(len(nextnodes)):
#                 score, nn = nextnodes[i]
#                 nodes.put((score, nn))
#                 # increase qsize
#             qsize += len(nextnodes) - 1

#         # choose nbest paths, back trace them
#         if len(endnodes) == 0:
#             endnodes = [nodes.get() for _ in range(topk)]

#         utterances = []
#         for score, n in sorted(endnodes, key=operator.itemgetter(0)):
#             utterance = []
#             utterance.append(n.wordid)
#             # back trace
#             while n.prevNode != None:
#                 n = n.prevNode
#                 utterance.append(n.wordid)

#             utterance = utterance[::-1]
#             utterances.append(utterance)

#         decoded_batch.append(utterances)

#     return decoded_batch

# def rnn_beam_search(src, model, TRG, opt):
#     max_length = opt.max_len
#     with torch.no_grad():
#         input_length = src.size()[0]

#         if opt.nmt_model_type == 'rnn_naive_model':
#             encoder_outputs = model.encoder(src)
#             encoder_hidden = encoder_outputs
#         elif opt.nmt_model_type == 'allign_and_translate':
#             encoder_outputs, encoder_hidden = model.encoder(src)

#         decoder_hidden = encoder_hidden
#         outputs = [TRG.vocab.stoi["<sos>"]]
#         word_outputs = ["<sos>"]

#         target_tensor = torch.ones([src.shape[1], max_length], device=opt.device)
#         decoded_batch = beam_decode(target_tensor, encoder_hidden, opt, model.decoder, encoder_outputs)

#         # TODO SOLVE THIS
#         for word in decoded_batch:
#             word_outputs.append(TRG.vocab.itos[word])

#         return ' '.join(word_outputs)
