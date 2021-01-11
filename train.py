import argparse, time
import torch
import torch.nn as nn
from Models import get_model
from Process import *
from tqdm import tqdm
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
from gensim.models import KeyedVectors
import dill as pickle

def train_model(model, opt): # model = NaiveModel, Transformer or Seq2Seq
    
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
                 
    criterion = nn.CrossEntropyLoss(ignore_index = opt.trg_pad) # optional (new way)
    for epoch in range(opt.epochs):

        total_loss = 0
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
                    
        print(f"Epoch has {get_len(opt.train)} batches.")
        for i, batch in tqdm(enumerate(opt.train)): # opt.train = MyIterator

            # [opt.SRC.vocab.itos[i] for i in batch.src[:, 0]] # to query batch words from field
            if opt.nmt_model_type == 'transformer':
                src = batch.src.transpose(0,1)
                trg = batch.trg.transpose(0,1)
                trg_input = trg[:, :-1]
                src_mask, trg_mask = create_masks(src, trg_input, opt)
                preds = model(src, trg_input, src_mask, trg_mask) # -> [batch_size, sent_len, emb_dim]
                ys = trg[:, 1:].contiguous().view(-1) # [batch_size * sent_len]
                opt.optimizer.zero_grad()
                loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            else:
                src = batch.src
                trg = batch.trg
                # ----- OLD WAY ------ #
                # preds = model(src, trg)
                # ys = trg.contiguous().view(-1)
                # -------------------- #
                # NEW WAY
                opt.optimizer.zero_grad()
                output = model(src, trg)
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
            loss.backward()
            opt.optimizer.step()

            if opt.SGDR == True: 
                opt.sched.step()
            
            total_loss += loss.item()
            
            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / opt.train_len)
                 avg_loss = total_loss/opt.printevery
                 if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='\r')
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                 total_loss = 0
            
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()
   
   
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))

def main():

    ############################
    ### OPTIONAL 4 THe FUTURE ##
    # DO WEIGHT DECAY BASED ON #
    # ATTENTION PAPER !!!!! ####
    ############################
    # step_list = [i*500 for i in range(2000)]
    # for step in step_list:
    #     lrate = (1/np.sqrt(512)) * min(1/np.sqrt(step), step*4000**-1.5 )
    #     print(f'{step}: lrate {lrate}')

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-trg_data', required=True)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512) # hidden size for models using RNN
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.00015)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=100) # max number of spaces per sentence
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-decoder_extra_layers', type=int, default=0)
    parser.add_argument('-nmt_model_type', type=str, default='transformer')
    parser.add_argument('-word_embedding_type', type=str, default=None)
    parser.add_argument('-use_dynamic_batch', action='store_true')

    opt = parser.parse_args()
    print(opt)

    # class InputArgs():
    #     def __init__(self):
    #         self.src_data = 'data/port_train.txt'
    #         self.trg_data = 'data/eng_train.txt'
    #         self.src_lang = 'pt'
    #         self.trg_lang = 'en'
    #         self.no_cuda = True
    #         self.SGDR = False
    #         self.epochs = 5 
    #         self.d_model = 300 
    #         self.n_layers = 6
    #         self.heads = 6
    #         self.dropout = 0.1
    #         self.batchsize = 1024
    #         self.printevery = 100
    #         self.lr = 0.00015
    #         self.load_weights = None 
    #         self.create_valset = False 
    #         self.max_strlen = 100 
    #         self.floyd = False 
    #         self.checkpoint = 1
    #         self.decoder_extra_layers = 0
    #         self.nmt_model_type = 'allign_and_translate' # 'transformer', 'rnn_naive_model', 'allign_and_translate' ...
    #         self.word_embedding_type = None # None, 'glove' or 'fast_text'
    #         self.use_dynamic_batch = None
    # opt = InputArgs()
    # print(opt.__dict__)

    # opt.device = 0 if opt.no_cuda is False else torch.device("cpu")
    if opt.no_cuda is False:
        assert torch.cuda.is_available()
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

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
    
    read_data(opt)
    SRC, TRG = create_fields(opt)
    opt.train, SRC, TRG = create_dataset(opt, SRC, TRG, word_emb)
    opt.SRC = SRC; opt.TRG = TRG # important, these are used to input embeddings
    opt.word_emb = word_emb # just for querying vocabulary
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab), word_emb)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    train_model(model, opt)

    if opt.floyd is False:
        promptNextAction(model, opt, SRC, TRG)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input(f"{pd.to_datetime('today')}: training complete, save results? [y/n] : "))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main()
