"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import os
import pandas as pd
import torch
from torch import nn
from src.smash_rnn_model import SmashRNNModel
import csv
import numpy as np
import argparse
from src.utils import get_max_lengths


def test(opt):
    test_generator = torch.load(opt.test_dataset_path)
    output_file = open('trained_models' + os.sep + 'logs.txt', 'a+')

    criterion = nn.MSELoss()

    model = load_model(opt.model_path, opt.full_dataset_path)

    loss_list = []
    predictions_list = []

    for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in test_generator:
        if torch.cuda.is_available():
            current_document = current_document.cuda()
            words_per_sentence_current_document = words_per_sentence_current_document.cuda()
            sentences_per_paragraph_current_document = sentences_per_paragraph_current_document.cuda()
            paragraphs_per_document_current_document = paragraphs_per_document_current_document.cuda()
            previous_document = previous_document.cuda()
            words_per_sentence_previous_document = words_per_sentence_previous_document.cuda()
            sentences_per_paragraph_previous_document = sentences_per_paragraph_previous_document.cuda()
            paragraphs_per_document_previous_document = paragraphs_per_document_previous_document.cuda()
            click_rate_tensor = click_rate_tensor.cuda()

        predictions = model(current_document, words_per_sentence_current_document,
                            sentences_per_paragraph_current_document,
                            paragraphs_per_document_current_document,
                            previous_document, words_per_sentence_previous_document,
                            sentences_per_paragraph_previous_document,
                            paragraphs_per_document_previous_document,
                            click_rate_tensor)
        loss = criterion(predictions, click_rate_tensor)

        loss_list.append(loss)
        predictions_list.append(predictions.clone().cpu())

    loss = sum(loss_list) / test_generator.dataset.__len__()

    output_file.write('Test: loss: {}\n\n'.format(loss))

    print('Test: loss: {}\n\n'.format(loss))


def load_model(model_path, full_dataset_path):
    if torch.cuda.is_available():
        model_state_dict = torch.load(model_path)
    else:
        model_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    word2vec_path = './data/glove.6B.50d.txt'

    # Load from txt file (in word2vec format)
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

    max_word_length, max_sent_length, max_paragraph_length = get_max_lengths(full_dataset_path)

    # Siamese + Attention model
    model = SmashRNNModel(dict, dict_len, embed_dim, max_word_length, max_sent_length, max_paragraph_length)

    # Overall model optimization and evaluation parameters
    criterion = nn.MSELoss()

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
    parser.add_argument("--model_path", type=str, default='./trained_models/model.pt')
    parser.add_argument("--full_dataset_path", type=str, default='./data/wiki_df.csv')
    parser.add_argument("--test_dataset_path", type=str, default='./data/validation.pth')

    return parser.parse_args()


if __name__ == "__main__":
    opt = get_args()
    test(opt)
