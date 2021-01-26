import pandas as pd
import torchtext, logging, torch
import torch.nn as nn
from torchtext import data
from Tokenize import tokenize
from Batch import MyIterator, batch_size_fn
import os, time, math, shutil
import dill as pickle
from tqdm import tqdm

class Timer():
    def __init__(self):
        self.i_t = time.time()
    def reset(self):
        self.i_t = time.time()
    def print_time(self, function_name):
        now = time.time()
        minutes = math.floor((now - self.i_t)/60)
        print(f"\n|{function_name}| Total Elapsed Time: {minutes} minutes and {now - self.i_t - minutes*60:.2f} seconds.\n")

def embedding_to_torchtext_vocab_translator(field, model):
    embedding_vectors = []
    for token, idx in tqdm(field.vocab.stoi.items()):
        if token in model.wv.vocab.keys():
            embedding_vectors.append(torch.FloatTensor(model[token].copy()))
        else:
            random_vector = nn.Embedding(1, model.vector_size)(torch.LongTensor([0]))
            embedding_vectors.append(random_vector)
        if token == '.':
            print(1)
    return embedding_vectors

def read_data(opt):
    
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
            opt.src_val_data = open(opt.src_val_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()
    
    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
            opt.trg_val_data = open(opt.trg_val_data).read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def create_fields(opt):
    timer = Timer()
    
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)  
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)
    
    lang_compatibility = {
        'en': 'en_core_web_md',
        'pt': 'pt_core_news_md',
        'fr': 'fr_core_news_md',
        'de': 'de_core_news_md',
        'es': 'es_core_news_md'
    }
    
    src_lang = lang_compatibility[opt.src_lang]
    trg_lang = lang_compatibility[opt.trg_lang]

    print("loading spacy tokenizers...")
    
    try:
        t_src = tokenize(src_lang)
        t_trg = tokenize(trg_lang)
    except Exception as e: 
        print(f'Reached exception {e}.\n Please download model using python -m spacy download.')
        
    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            # import ipdb; ipdb.set_trace()
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
        
    timer.print_time('create_fields')

    return(SRC, TRG)

def generate_tabular_dataset(opt):
    raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen) # filtering senteces with more words than max_strlen
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)
    
    data_fields = [('src', opt.SRC), ('trg', opt.TRG)]
    dataset = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    return dataset 

def create_dataset(opt, SRC, TRG, word_emb):
    timer = Timer()
    print("creating dataset and iterator... ")
    train = generate_tabular_dataset(opt)
    valid = generate_tabular_dataset(opt)    

    if opt.use_dynamic_batch == True:
        train_iter, valid_iter = MyIterator.splits((train, valid), batch_size=opt.batchsize, device=opt.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, shuffle=True) # batch_size_fn = dynamic batching
    else:
        train_iter, valid_iter = MyIterator.splits((train, valid), batch_size=opt.batchsize, device=opt.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), shuffle=True)

    os.remove('translate_transformer_temp.csv')

    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if word_emb is not None:
            SRC_vectors = embedding_to_torchtext_vocab_translator(SRC, word_emb)
            TRG_vectors = embedding_to_torchtext_vocab_translator(TRG, word_emb)
            # put word embedding vectors inside vocab class
            SRC.vocab.set_vectors(SRC.vocab.stoi, SRC_vectors, opt.d_model)
            TRG.vocab.set_vectors(TRG.vocab.stoi, TRG_vectors, opt.d_model) 

        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("Deleting and creating new weights folder. If this was unexpected, run program before 5s with -load_weights weights to load them")
                time.sleep(5)
                shutil.rmtree("weights")
                os.mkdir("weights")
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>'] # get number of pad token, used for masking 
    opt.trg_pad = TRG.vocab.stoi['<pad>'] # Pad the text so that all the sequences are the same length, so you can process them in batch
    opt.trg_sos = TRG.vocab.stoi['<sos>']

    opt.train_len = get_len(train_iter)
    timer.print_time('create_dataset')

    return train_iter, valid_iter, SRC, TRG

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i
