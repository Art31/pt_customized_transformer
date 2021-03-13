import argparse
import six
from tqdm import tqdm
import pandas as pd
import string, re
from nltk.translate.bleu_score import corpus_bleu
from typing import List, Tuple, Dict, Set, Union

def compute_corpus_level_bleu_score(references: List[str], hypotheses: List[str]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    bleu_score = corpus_bleu([[ref] for ref in references],
                                [hyp for hyp in hypotheses])
    return bleu_score

def standardize_punctuation_and_lowercase(sentences: List[str]) -> List[str]:
    '''
    This function removes the last punctuation mark and lowers the words.
    Input: 'realmente gostei dessa camisa . deveria ter comprado .'
    Output: 'realmente gostei dessa camisa . deveria ter comprado'
    '''
    for i, sentence in enumerate(sentences):
        punkt_occurrences = 0
        sentence_list = []
        matches = re.finditer(r'[.?!]', sentence)
        output_generator = [match for match in matches]
        if len(output_generator) == 0:
            continue
        last_match = output_generator[-1]
        last_match_ind = last_match.start()
        # remove space before punctuation
        if sentence[last_match_ind-1:last_match_ind+1].__contains__(' '):
            sentences[i] = sentence[:last_match_ind-1].lower()
        else:
            sentences[i] = sentence[:last_match_ind].lower()
    return sentences

def translate_sentences(text_list, output):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    from google.cloud import translate_v2 as translate
    # RODAR export GOOGLE_APPLICATION_CREDENTIALS="/home/arthurtelles/gtranslate-api-290022-4899f0c9d3f7.json"
    translate_client = translate.Client()

    # text = 'ola eu gosto de framboesa'
    result_list = []
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    for line in tqdm(text_list):
        if isinstance(line, six.binary_type):
            line = line.decode("utf-8")
        result = translate_client.translate(line, target_language='en')
        result_list.append(result)
    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    text_file2 = open("eng_gtranslate.txt", "a")
    for line in result_list:
        text_file2.write(f"{line['translatedText']}\n")
    text_file2.close()
    return result_list

parser = argparse.ArgumentParser()
parser.add_argument('-input_file', required=True)
parser.add_argument('-reference_file', required=True)
parser.add_argument('-bleu_output', required=True)
parser.add_argument('-google_translate_api', action='store_true')
parser.add_argument('-output_translations_csv', type=str)

opt = parser.parse_args()
print(f'Inputs: {opt}\n')

# class InputArgs():
#     def __init__(self):
#         self.input_file = 'seq2seq_test_translations2.txt'
#         self.reference_file = 'data/eng_test.txt' # 'rnn_naive_model_translations.txt' # 'vanilla_transformer.txt' 
#         self.bleu_output = 'seq2seq_test_bleu2.txt' # 'weights_test' # 'rnn_naive_model' # 'transformer_test'
#         self.google_translate_api = False
#         self.output_translations_csv = 'seq2seq_test_bleu2.csv'
# opt = InputArgs()
# print(opt.__dict__)

ref_file = open(opt.reference_file, "r+").read()
ref_list = ref_file.split('\n')

# não rodar pois já sabemos o resultado
if opt.google_translate_api == True: 
    result_list = translate_sentences(translated_list)
    translated_list = [result['translatedText'] for result in result_list]
else: 
    text_file1 = open(opt.input_file, "r+").read()
    translated_list = text_file1.split('\n')

ref_list = standardize_punctuation_and_lowercase(ref_list)
translated_list = standardize_punctuation_and_lowercase(translated_list)
bleu_score = compute_corpus_level_bleu_score(ref_list, translated_list)
print(f"Corpus bleu from file {opt.input_file}: {bleu_score}\n")

if opt.output_translations_csv is not None:
    data_list = [{'Reference': ref_list[i], 'Hypothesis': translated_list[i]} for i in range(len(ref_list))]
    data_df = pd.DataFrame(data_list)
    tqdm.pandas()
    data_df['sentence_bleu'] = data_df.progress_apply(lambda x: compute_corpus_level_bleu_score(
                                                        [x['Reference']], [x['Hypothesis']]), axis=1)
    data_df.to_csv(opt.output_translations_csv, index=False)

text_file2 = open(opt.bleu_output, "a")
text_file2.write(f"Corpus bleu from file {opt.input_file}: {bleu_score}\n")
text_file2.close()