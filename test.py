"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import os
import pandas as pd
from comet_ml import Experiment
import torch
from torch import nn
from src.smash_rnn_model import SmashRNNModel
from src.word_smash_rnn_model import WordLevelSmashRNNModel
from src.sentence_smash_rnn_model import SentenceLevelSmashRNNModel
import csv
import numpy as np
import argparse
from src.utils import get_max_lengths


def test(opt):
    test_generator = torch.load(opt.test_dataset_path)
    output_file = open('./trained_models' + os.sep + 'logs.txt', 'a+')

    experiment = Experiment(api_key="NPD7aHoJxhZgG0MNWBkFb3hzZ",
                            project_name="thesis-davi",
                            workspace="davipaula")

    criterion = nn.MSELoss()

    model = load_model(opt.model_folder, opt.full_dataset_path, opt.level, opt.word2vec_path)

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

        if opt.level == 'paragraph':
            predictions = model(current_document, words_per_sentence_current_document,
                                sentences_per_paragraph_current_document,
                                paragraphs_per_document_current_document,
                                previous_document, words_per_sentence_previous_document,
                                sentences_per_paragraph_previous_document,
                                paragraphs_per_document_previous_document,
                                click_rate_tensor)
        elif opt.level == 'sentence':
            predictions = model(current_document, words_per_sentence_current_document,
                                sentences_per_paragraph_current_document,
                                previous_document, words_per_sentence_previous_document,
                                sentences_per_paragraph_previous_document,
                                click_rate_tensor)
        elif opt.level == 'word':
            predictions = model(current_document, words_per_sentence_current_document,
                                previous_document, words_per_sentence_previous_document,
                                click_rate_tensor)

        loss = criterion(predictions, click_rate_tensor)

        loss_list.append(loss)
        predictions_list.append(predictions.clone().cpu())

    loss = sum(loss_list) / test_generator.dataset.__len__()

    experiment.log_metric('test_{}_level_loss'.format(opt.level), loss.item())

    print('Test: loss: {}\n\n'.format(loss))


def load_model(model_folder, full_dataset_path, level, word2vec_path):
    # Load from txt file (in word2vec format)
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

    max_word_length, max_sent_length, max_paragraph_length = get_max_lengths(full_dataset_path)

    # Siamese + Attention model
    if level == 'paragraph':
        model = SmashRNNModel(dict, dict_len, embed_dim, max_word_length, max_sent_length, max_paragraph_length)
    elif level == 'sentence':
        model = SentenceLevelSmashRNNModel(dict, dict_len, embed_dim, max_word_length, max_sent_length)
    elif level == 'word':
        model = WordLevelSmashRNNModel(dict, dict_len, embed_dim)
    else:
        raise SystemExit(0)

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


if __name__ == "__main__":
    opt = get_args()
    test(opt)
