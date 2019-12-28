"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from smash_rnn_model import SmashRNNModel
from src.utils import get_evaluation
import argparse
import shutil
import csv
import numpy as np


def test():
    pre_trained_model_path = './model.pt'

    test_generator = torch.load('./data/test.pth')
    output_file = open('trained_models' + os.sep + 'logs.txt', 'a+')

    criterion = nn.MSELoss()

    model = load_model(pre_trained_model_path)

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

    # self.writer.add_scalar('{}/Loss'.format(step.capitalize()), loss, epoch)


def load_model(filepath):
    if torch.cuda.is_available():
        model_state_dict = torch.load(filepath)
    else:
        model_state_dict = torch.load(filepath, map_location=lambda storage, loc: storage)

    word2vec_path = './data/glove.6B.50d.txt'
    # Load from txt file (in word2vec format)
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

    # Siamese + Attention model
    model = SmashRNNModel(dict, dict_len, embed_dim)

    model.load_state_dict(model_state_dict)
    for parameter in model.parameters():
        parameter.requires_grad = False

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    return model


if __name__ == "__main__":
    test()
