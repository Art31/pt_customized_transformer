import argparse
import time
import torch
from tqdm import tqdm
from Models import get_model
from Process import create_dataset, create_fields, read_data
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam import beam_search, generate_rnn_translations#, rnn_beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
from gensim.models import KeyedVectors
import re, math

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate_sentence(sentence, model, opt, SRC, TRG, counter):
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    # import ipdb; ipdb.set_trace()
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(0)
        #     indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.no_cuda is False:
        sentence = sentence.cuda()
    # try: 
        # import ipdb; ipdb.set_trace()
    if opt.nmt_model_type == 'transformer':
        sentence = beam_search(sentence, model, SRC, TRG, opt)
    else:
        # sentence = rnn_beam_search(sentence, model, TRG, opt)
        sentence = generate_rnn_translations(sentence, model, TRG, opt)
    # except:
    #     sentence = ''
    #     print(f'Error happened at sentence {counter}!')
        # import ipdb; ipdb.set_trace()
        
    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)

def translate(opt, model, SRC, TRG):
    sentences = [text.lower() for text in opt.text]
    translated = []

    print(f"We have {len(sentences)} to process!")
    for i, sentence in tqdm(enumerate(sentences)):
        if sentence.__contains__('.') == False:
            sentence = sentence + '.'
        translated.append(translate_sentence(sentence, model, opt, SRC, TRG, i).capitalize())

    return ('\n'.join(translated))


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-translate_file', required=True)
    parser.add_argument('-output', required=True)
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-nmt_model_type', type=str, default='transformer')
    parser.add_argument('-decoder_extra_layers', type=int, default=0)
    parser.add_argument('-word_embedding_type', type=str, default=None)
    
    opt = parser.parse_args()
    print(opt)

    # class InputArgs():
    #     def __init__(self):
    #         self.translate_file = 'data/port_test.txt'
    #         self.output = 'test_translations.txt' # 'rnn_naive_model_translations.txt' # 'vanilla_transformer.txt' 
    #         self.load_weights = 'weights_test' # 'rnn_naive_model' # 'vanilla_transformer'
    #         self.src_lang = 'pt'
    #         self.trg_lang = 'en'
    #         self.no_cuda = True
    #         self.d_model = 300 
    #         self.heads = 6
    #         self.nmt_model_type = 'rnn_naive_model' # 'transformer', 'rnn_naive_model', 'allign_and_translate' ...
    #         self.word_embedding_type = None # None, 'glove' or 'fast_text'
    #         self.k = 3
    #         self.max_len = 100
    #         self.dropout = 0.1
    #         self.n_layers = 6
    #         self.decoder_extra_layers = 0
    #         self.floyd = False
    #         # self.use_dynamic_batch = None
    # opt = InputArgs()
    # print(opt.__dict__)

    if opt.no_cuda is False:
        assert torch.cuda.is_available()
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")
 
    assert opt.k > 0
    assert opt.max_len > 10

    i_t = time.time()
    if opt.word_embedding_type in ['glove', 'fast_text']:
        if opt.word_embedding_type == 'glove':
            word_emb = KeyedVectors.load_word2vec_format('word_embeddings/glove_s300.txt')
        elif opt.word_embedding_type == 'fast_text':
            word_emb = KeyedVectors.load_word2vec_format('word_embeddings/ftext_skip_s300.txt')
        now = time.time()
        minutes = math.floor((now - i_t)/60)
        print(f'\nWord embeddding of type {str(opt.word_embedding_type)} took {minutes} minutes \
            and {now - i_t - minutes*60:.2f} seconds to load.\n')
    elif opt.word_embedding_type is None:
        word_emb = opt.word_embedding_type

    SRC, TRG = create_fields(opt)
    opt.SRC = SRC; opt.TRG = TRG # important, these are used to input embeddings
    opt.word_emb = word_emb # just for querying vocabulary
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab), word_emb)
    
    try:
        opt.text = open(opt.translate_file, encoding='utf-8').read().split('\n')
    except:
        print("error opening or reading text file")
    phrase = translate(opt, model, SRC, TRG)
    f = open(opt.output, "w+")
    f.write(phrase)
    f.close()

    print('Sample >'+ phrase[:300] + '\n')

if __name__ == '__main__':
    main()
