"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import torch
import numpy as np
import pandas as pd
import logging
from sklearn import metrics
from ast import literal_eval
import itertools as it


# csv.field_size_limit(sys.maxsize)
def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.matmul(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()


def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(len(feature_2.shape)).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def get_max_lengths(data_path, limit_rows=None):
    word_length_list = []
    sent_length_list = []
    paragraph_length_list = []

    dataset = pd.read_csv(data_path, nrows=limit_rows)

    document_encode = dataset['current_article_text']
    document_encode = pd.concat([document_encode, dataset['previous_article_text']])

    for index, article in enumerate(document_encode):
        for paragraph in literal_eval(article):
            for sentences in paragraph:
                word_length_list.append(len(sentences))

            sent_length_list.append(len(paragraph))

        paragraph_length_list.append(len(literal_eval(article)))

    return max(word_length_list), max(sent_length_list), max(paragraph_length_list)


def get_padded_document(document, max_length_word, max_length_sentences, max_length_paragraph):
    for paragraph in document:
        for sentences in paragraph:
            if len(sentences) < max_length_word:
                extended_words = [-1 for _ in range(max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(paragraph) < max_length_sentences:
            extended_sentences = [[-1 for _ in range(max_length_word)] for _ in
                                  range(max_length_sentences - len(paragraph))]
            paragraph.extend(extended_sentences)

    if len(document) < max_length_paragraph:
        extended_paragraphs = [[[-1 for _ in range(max_length_word)]
                                for _ in range(max_length_sentences)]
                               for _ in range(max_length_paragraph - len(document))]

        document.extend(extended_paragraphs)

    document = [sentences[:max_length_word] for sentences in document][
               :max_length_sentences]

    document = np.stack(arrays=document, axis=0)
    document += 1

    return document


def get_document_at_word_level(document, words_per_sentence):
    word_level_document = []

    for paragraph_idx, paragraph in enumerate(words_per_sentence[0]):
        for sentence_idx, number_of_words in enumerate(paragraph):
            if number_of_words > 0:
                word_level_document.extend(document[0, paragraph_idx, sentence_idx, :number_of_words].tolist())

    return torch.LongTensor([word_level_document])


def get_document_at_sentence_level(document):
    rearranged_tensor = list(it.chain.from_iterable(document[0].tolist()))

    return torch.LongTensor([rearranged_tensor])


def get_words_per_sentence_at_sentence_level(words_per_sentence):
    words_per_sentence_at_sentence_level = []

    for paragraph in words_per_sentence:
        for sentence in paragraph:
            for words in sentence:
                if words > 0:
                    words_per_sentence_at_sentence_level.append(words)

    return torch.LongTensor(words_per_sentence_at_sentence_level)


if __name__ == "__main__":
    word, sent, paragraph = get_max_lengths('../data/wiki_df_small.csv')
    print(word)
    print(sent)
    print(paragraph)

    click_stream_dump_path = '../data/clickstream-enwiki-2019-08.tsv'
    wiki_documents_path = '../data/simplewiki.jsonl'

    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
