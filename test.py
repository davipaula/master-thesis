"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import os
import pandas as pd
from comet_ml import Experiment
import torch
from torch import nn
from src.smash_rnn_model import SmashRNNModel
import csv
import numpy as np
import argparse
from src.utils import get_max_lengths, get_words_per_document_at_word_level, get_document_at_word_level, \
    get_document_at_sentence_level, get_words_per_sentence_at_sentence_level, \
    get_sentences_per_paragraph_at_sentence_level
from datetime import datetime


def test(opt):
    test_generator = torch.load(opt.test_dataset_path)

    experiment = Experiment(api_key="NPD7aHoJxhZgG0MNWBkFb3hzZ",
                            project_name="thesis-davi",
                            workspace="davipaula")

    criterion = nn.MSELoss()

    model = load_model(opt.model_folder, opt.full_dataset_path, opt.level, opt.word2vec_path)

    loss_list = []
    columns_names = ['previous_document', 'current_document', 'actual_click_rate', 'predicted_click_rate']
    predictions_list = pd.DataFrame(columns=columns_names)

    print('Starting test')

    for current_document, previous_document, click_rate_tensor in test_generator:
        if torch.cuda.is_available():
            current_document['text'] = current_document['text'].cuda()
            current_document['words_per_sentence'] = current_document['words_per_sentence'].cuda()
            current_document['sentences_per_paragraph'] = current_document['sentences_per_paragraph'].cuda()
            current_document['paragraphs_per_document'] = current_document['paragraphs_per_document'].cuda()
            previous_document['text'] = previous_document['text'].cuda()
            previous_document['words_per_sentence'] = previous_document['words_per_sentence'].cuda()
            previous_document['sentences_per_paragraph'] = previous_document['sentences_per_paragraph'].cuda()
            previous_document['paragraphs_per_document'] = previous_document['paragraphs_per_document'].cuda()
            click_rate_tensor = click_rate_tensor.cuda()

        if opt.level == 'sentence':
            current_document = transform_to_sentence_level(current_document)
            previous_document = transform_to_sentence_level(previous_document)

        elif opt.level == 'word':
            current_document = transform_to_word_level(current_document)
            previous_document = transform_to_word_level(previous_document)

        predictions = model(current_document['text'],
                            current_document['words_per_sentence'],
                            current_document['sentences_per_paragraph'],
                            current_document['paragraphs_per_document'],
                            previous_document['text'],
                            previous_document['words_per_sentence'],
                            previous_document['sentences_per_paragraph'],
                            previous_document['paragraphs_per_document'])

        loss = criterion(predictions, click_rate_tensor)
        loss_list.append(loss)

        batch_results = pd.DataFrame(
            zip(current_document['title'],
                previous_document['title'],
                click_rate_tensor.squeeze(1).tolist(),
                predictions.squeeze(1).tolist()),
            columns=columns_names
        )

        predictions_list = predictions_list.append(batch_results, ignore_index=True)

        loss = sum(loss_list) / len(loss_list)

        experiment.log_metric('test_{}_level_loss'.format(opt.level), loss.item())

        print('Test: loss: {}\n\n'.format(loss))
        predictions_list.to_csv('results_{}_level_{}.csv'.format(opt.level, datetime.now()), index=False)


def load_model(model_folder, full_dataset_path, level, word2vec_path):
    # Load from txt file (in word2vec format)
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

    max_word_length, max_sent_length, max_paragraph_length = get_max_lengths(full_dataset_path)

    # Siamese + Attention model
    model = SmashRNNModel(dict, dict_len, embed_dim, max_word_length, max_sent_length, max_paragraph_length)

    model_path = model_folder + os.sep + level + '_level_model.pt'
    if torch.cuda.is_available():
        model_state_dict = torch.load(model_path)
    else:
        model_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(model_state_dict)
    for parameter in model.parameters():
        parameter.requires_grad = False

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    return model


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Semantic Text Matching for Long-Form Documents to predict the number of clicks for Wikipedia articles""")
    parser.add_argument("--model_folder", type=str, default='./trained_models/')
    parser.add_argument("--level", type=str, default='paragraph')
    parser.add_argument("--full_dataset_path", type=str, default='./data/wiki_df.csv')
    parser.add_argument("--test_dataset_path", type=str, default='./data/test.pth')
    parser.add_argument("--word2vec_path", type=str, default='./data/glove.6B.50d.txt')

    return parser.parse_args()


def transform_to_word_level(document):
    batch_size = document['text'].shape[0]

    document['words_per_sentence'] = get_words_per_document_at_word_level(document['words_per_sentence'])
    document['text'] = get_document_at_word_level(document['text'], document['words_per_sentence'])
    document['sentences_per_paragraph'] = torch.ones((batch_size, 1), dtype=int)
    document['paragraphs_per_document'] = torch.ones((batch_size, 1), dtype=int)

    return document


def transform_to_sentence_level(document):
    batch_size = document['text'].shape[0]

    document['text'] = get_document_at_sentence_level(document['text'])
    document['words_per_sentence'] = get_words_per_sentence_at_sentence_level(document['words_per_sentence'])
    document['sentences_per_paragraph'] = get_sentences_per_paragraph_at_sentence_level(
        document['sentences_per_paragraph'])
    document['paragraphs_per_document'] = torch.ones((batch_size, 1), dtype=int)

    return document


if __name__ == "__main__":
    opt = get_args()
    test(opt)
