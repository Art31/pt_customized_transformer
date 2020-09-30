git clone https://github.com/Art31/pt_customized_transformer
cd pt_customized_transformer
pip3 install -U spacy
python3 -m spacy download pt_core_news_md
python3 -m spacy download en_core_web_md
pip3 install ipdb dill
pip3 install torchtext==0.4.0
ipython3 train.py -- -src_data data/port_train.txt -trg_data data/eng_train.txt -src_lang pt -trg_lang en -epochs 500 -checkpoint 10 -batchsize 4096 -lr 0.00015 -n_layers 6 -decoder_extra_layers 
