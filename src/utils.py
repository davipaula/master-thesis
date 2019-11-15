"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import sys
import csv
import numpy as np
import pandas as pd
import logging
from sklearn import metrics
from ast import literal_eval

csv.field_size_limit(sys.maxsize)


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
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    paragraph_length_list = []

    dataset = pd.read_csv(data_path)

    document_encode = dataset['current_article_text']
    document_encode = pd.concat([document_encode, dataset['previous_article_text']])

    for index, article in enumerate(document_encode):
        for paragraph in literal_eval(article):
            for sentences in paragraph:
                word_length_list.append(len(sentences))

            sent_length_list.append(len(paragraph))

        paragraph_length_list.append(len(literal_eval(article)))

    return max(word_length_list), max(sent_length_list), max(paragraph_length_list)


if __name__ == "__main__":
    word, sent, paragraph = get_max_lengths('../data/wiki_df_small.csv')
    print(word)
    print(sent)
    print(paragraph)

    click_stream_dump_path = '../data/clickstream-enwiki-2019-08.tsv'
    wiki_documents_path = '../data/simplewiki.jsonl'

    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
