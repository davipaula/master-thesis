"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import sys
import csv
import numpy as np
import json
import pandas as pd
import logging
from sklearn import metrics

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
        feature = torch.mm(feature, weight)
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

    document_encode = []

    with open(data_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        document_encode.append(result['sections'][0]['paragraphs'])

    for index, article in enumerate(document_encode):
        for paragraph in article:
            for sentences in paragraph:
                word_length_list.append(len(sentences))

            sent_length_list.append(len(paragraph))

    return max(word_length_list), max(sent_length_list)


def read_click_stream_dump(click_stream_dump_path):
    rows = []

    with open(click_stream_dump_path, 'r') as f:
        for i, line in enumerate(f):
            cols = line.split('\t')

            if cols[2] == 'link':  # type
                rows.append([
                    cols[0],  # prev
                    cols[1],  # current
                    int(cols[3]),  # n
                ])

    # TODO use dumps from more months

    return pd.DataFrame(rows, columns=['prev', 'current', 'n'])


def get_available_titles(dataset_path):
    # Load preprocessed
    title2sects = {}

    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            title = doc['title'].replace(' ', '_')
            title2sects[title] = [sect['paragraphs'] for sect in doc['sections']]

    print(f'Completed after {i} lines')

    return set(title2sects.keys())  # save as set (makes it faster)


def normalize_dataset(cdf, available_titles):

    # Clicks for that we have matching articles
    fdf = cdf[(cdf['prev'].isin(available_titles)) & (cdf['current'].isin(available_titles))].copy()

    print(f'Click pairs with articles: {len(fdf):,}')

    max_n = fdf.groupby(['prev']).agg({'n': 'max'})

    # Normalize click count with max value
    fdf['rel_n'] = 0.

    for idx, r in fdf.iterrows():
        fdf.at[idx, 'rel_n'] = r['n'] / max_n['n'][r['prev']]

    return fdf


def dataset_creation(click_stream_dump_path, dataset_path):

    cdf = read_click_stream_dump(click_stream_dump_path)

    print(f'Total click pairs: {len(cdf):,}')

    available_titles = get_available_titles(dataset_path)

    fdf = normalize_dataset(cdf, available_titles)

    print(fdf.sample(n=10))

    fdf.to_csv('../data/click_pair.csv',
               index=False,
               header=['previous_article', 'current_article', 'number_of_clicks', 'click_rate'])


if __name__ == "__main__":
    word, sent = get_max_lengths("../data/simplewiki_small.jsonl")
    print(word)
    print(sent)

    cs_dump_path = '../data/clickstream-enwiki-2019-08.tsv'
    docs_path = '../data/simplewiki.jsonl'

    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
