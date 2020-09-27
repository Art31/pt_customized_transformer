import argparse
from google.cloud import translate_v2 as translate
import six
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from typing import List, Tuple, Dict, Set, Union

def compute_corpus_level_bleu_scoreresult_list(references: List[List[str]], hypotheses: List[str]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp for hyp in hypotheses])
    return bleu_score


def translate_sententeces(text_list, output):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
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
parser.add_argument('-translate_from_input', action='store_true')

opt = parser.parse_args()

print(f'Inputs: {opt}\n')

# text_file1 = open("/home/arthurtelles/Downloads/port_test.txt", "r+").read()
text_file1 = open(opt.input_file, "r+").read()
translated_list = text_file1.split('\n')

ref_file = open(opt.reference_file, "r+").read()
ref_list = ref_file.split('\n')

# não rodar pois já sabemos o resultado
if opt.translate_from_input == True: 
    result_list = translate_sententeces(translated_list)
    translated_list = [result['translatedText'] for result in result_list]
bleu_score = compute_corpus_level_bleu_score(list(ref_list), translated_list)

text_file2 = open(opt.bleu_output, "a")
message = f"Corpus bleu from file {opt.input_file}: {bleu_score}\n"
print(message)
text_file2.write(f"Corpus bleu from file {opt.input_file}: {bleu_score}\n")
text_file2.close()