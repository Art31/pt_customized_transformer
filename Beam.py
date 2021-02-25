import torch, random
from Batch import nopeak_mask
import torch.nn.functional as F
import math


def init_vars(src, model, SRC, TRG, opt):
    
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)

    if opt.nmt_model_type == 'transformer':
        if opt.no_cuda is False:
            outputs = torch.LongTensor([[init_tok]]).to(opt.device)
        else: 
            outputs = torch.LongTensor([[init_tok]])
        # if opt.no_cuda is False:
        #     outputs = outputs.cuda()
        e_output = model.encoder(src, src_mask) # [[1, 7], [1, 1, 7]] -> [1, 7, 300 (d_model)]
        trg_mask = nopeak_mask(1, opt)
        out = model.out(model.decoder(outputs,
                e_output, src_mask, trg_mask)) # [[1, 1], [1, 7, 300]] -> [1, 1, 300] -> [1, 1, 11436 (out_features)]
        out = F.softmax(out, dim=-1) # [1, 1, 11436])
        probs, ix = out[:, -1].data.topk(opt.k)
    elif opt.nmt_model_type == 'rnn_naive_model':
        outputs = torch.zeros(1, src.shape[1]).long().to(opt.device)
        outputs[0, 0] = init_tok
        # ---------- OLD RNN ---------- #
        e_output = model.encoder(src) # [1, 7] -> [1, 7, 300 (d_model)]
        encoder_hidden = e_output
        out, decoder_hidden = model.decoder(
                outputs[0, :], encoder_hidden, e_output) # ([1, 7], [1, 7, 300], [1, 7, 300]) -> [[7, 11436], [1, 7, 300]]
        out = out[0, :] # [7, 11436] -> [1, 11436]
        out = F.softmax(out, dim=-1)
        probs, ix = out.unsqueeze(0).data.topk(opt.k) # -> [[1, 3], [1, 3]]

    # elif opt.nmt_model_type == 'align_and_translate':
    #     e_output, hidden = model.encoder(src)
    
    
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_len).long().to(opt.device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0] # 3 sets of sentences with init_tok and 1st, 2nd and 3rd most probable translations
    
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1)).to(opt.device)
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k, TRG, SRC):
    
    probs, ix = out[:, -1].data.topk(k) # get most probable in softmax output
    try:
        log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1) # calculate log of probable translations
    except:
        a = 1
        # import ipdb; ipdb.set_trace()
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k # get rows of most probable translations
    col = k_ix % k # get columns of most probable translations

    next_config = torch.cat([outputs[row, :i], ix[row, col].unsqueeze(1)], dim=1)
    # print(f"Outputs now are\n{outputs}\n will become\n{next_config}")
    # print(f"Outputs now are\n{[[TRG.vocab.itos[ind] for ind in sent] for sent in outputs.tolist()]}\n will become\n{[[TRG.vocab.itos[ind] for ind in sent] for sent in next_config.tolist()]}")
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(src, model, SRC, TRG, opt):
    eos_tok = TRG.vocab.stoi['<eos>']
    pad_tok = SRC.vocab.stoi['<pad>']
    ind = None
    if opt.nmt_model_type == 'rnn_naive_model':
        tensor_to_fill_max_len = torch.full((1, opt.max_len - src.shape[1]), pad_tok).to(opt.device)
        src = torch.cat((src, tensor_to_fill_max_len), dim=1)
        outputs, encoder_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
        encoder_hidden = encoder_outputs
        decoder_hidden = encoder_hidden
    elif opt.nmt_model_type == 'transformer':
        src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
        outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt) #  [SRC.vocab.itos[i] for i in src.tolist()[0]] to debug

    for i in range(2, opt.max_len): # we already filled init_tok and some of most probable translations
    
        if opt.nmt_model_type == 'transformer': # keep increasing size of sentence
            trg_mask = nopeak_mask(i, opt)
            out = model.out(model.decoder(outputs[:,:i],
                    e_outputs, src_mask, trg_mask)) # [[3, 2], [3, 7, 300], [1, 1, 7], [1, 2, 2]] -> [3, 2, 300] -> [3, 2, 11436]
            out = F.softmax(out, dim=-1)
        elif opt.nmt_model_type == 'rnn_naive_model':
            decoder_input = torch.zeros(opt.k, src.shape[1]).long().to(opt.device) # TODO change to opt.max_len
            decoder_input[:, :i] = outputs[:, :i]
            # OPTION 2 - input a tensor of size src.shape[1] and fill up the other numbers with <unk>
            for j in range(opt.k):
                out_piece, decoder_hidden_piece = model.decoder(decoder_input[j, :], 
                                            decoder_hidden[j, :].unsqueeze(0), encoder_outputs[j, :].unsqueeze(0))
                if j == 0:
                    out = out_piece[:i, :].unsqueeze(0)
                    decoder_hidden_carry = decoder_hidden_piece[:i, :]
                else:
                    out = torch.cat([out, out_piece[:i, :].unsqueeze(0)], dim=0) # final shape: [src.shape[1]*3, vocab_size]
                    decoder_hidden_carry = torch.cat([decoder_hidden_carry, decoder_hidden_piece[:i, :]], dim=0)
            decoder_hidden = decoder_hidden_carry
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k, TRG, SRC) # (torch.Size([3, 100]), torch.Size([3, 2, 11436]))
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(opt.device)
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
        length = (outputs[0]==eos_tok).nonzero()
        if len(length) != 0:
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length[0]]])
        else:
            eos_3_hypothesis = torch.ones([3, 1], dtype=torch.int64).to(opt.device)*eos_tok
            outputs = torch.cat((outputs, eos_3_hypothesis), dim=1)
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()
        if len(length) != 0:
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length[0]]])
        else:
            eos_3_hypothesis = torch.ones([3, 1], dtype=torch.int64).to(opt.device)*eos_tok
            outputs = torch.cat((outputs, eos_3_hypothesis), dim=1)
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:]])

# def get_winners_from_candidates(word_vector, k):
#     probs, ix = word_vector[:, -1].data.topk(k) # get most probable in softmax output
#     log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1) # calculate log of probable translations
#     k_probs, k_ix = log_probs.view(-1).topk(k)
#     return 

# def generate_rnn_translations(src, model, SRC, TRG, opt):
#     with torch.no_grad():
#         # outputs = [TRG.vocab.stoi["<sos>"]]
#         # word_outputs = ["<sos>"]
#         eos_tok = TRG.vocab.stoi['<eos>']
#         if opt.nmt_model_type == 'rnn_naive_model':
#         #     encoder_outputs = model.encoder(src)
#             word_vector, encoder_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)         
#             encoder_hidden = encoder_outputs
#             decoder_hidden = encoder_hidden
#         # elif opt.nmt_model_type == 'align_and_translate':
#         #     encoder_outputs, encoder_hidden = model.encoder(src)
#         # # TODO implement for transformer as a sanity check for this code
#         # elif opt.nmt_model_type == 'transformer':
#         #     outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)


#         # query input words: [opt.SRC.vocab.itos[ind] for ind in src[0, :]]
#         for i in range(2, opt.max_len):
#             # previous_word = torch.LongTensor([outputs[-1]]).to(opt.device)
#             # word_vector = torch.LongTensor([word_vector[:, :i]]).to(opt.device)
#             # OPTION 1 - input one word but has to filter the encoder info (not desired)
#             # decoder_output, decoder_hidden = model.decoder(
#             #     previous_word, decoder_hidden[:, 0, :].unsqueeze(0), encoder_outputs[:, 0, :].unsqueeze(0)) # [[1, 1], [1, 7, 300], [1, 7, 300]]
#             decoder_input = torch.zeros(opt.k, src.shape[1]).long().to(opt.device) # TODO change to opt.max_len
#             decoder_input[:, :i] = word_vector[:, :i]
#             # OPTION 2 - input a tensor of size src.shape[1] and fill up the other numbers with <unk>
#             decoder_hidden_copy = decoder_hidden.detach().clone()
#             for j in range(opt.k):
#                 out, decoder_hidden = model.decoder(decoder_input[i, :].unsqueeze(0), 
#                                             decoder_hidden_copy[i, :].unsqueeze(0), encoder_outputs[i, :].unsqueeze(0))
#                 if j == 0:
#                     out_agg = out[:i, :].unsqueeze(0)
#                 else:
#                     out_agg = torch.cat([out_agg, out[:i, :].unsqueeze(0)], dim=0) # final shape: [src.shape[1]*3, vocab_size]
#             # pytorch seq2seq desired [[1, 1], [1, 1, 256], [10, 256]] -> [[1, 2711], [1, 1, 256], [1, 10]]

#             word_vector, log_scores = k_best_outputs(word_vector, out_agg, log_scores, i, opt.k, TRG)
#             # best_guess = decoder_output.argmax() % decoder_output.shape[1]
#             # best_guess = best_guess.item()
#             # outputs.append(best_guess)
#             if best_guess == TRG.vocab.stoi['<eos>']:
#                 word_outputs.append('<eos>')
#                 break
#             else:
#                 word_outputs.append(TRG.vocab.itos[best_guess])

#             # previous_word = topi.squeeze().detach()

#         return ' '.join(word_outputs)

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
#         decoder_input = torch.LongTensor([[opt.TRG.vocab.stoi["<sos>"]]]).to(opt.device)

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
