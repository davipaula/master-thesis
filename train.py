"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import csv
import os
from ast import literal_eval
import gensim
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import get_max_lengths
from smash_rnn_model import SmashRNNModel
from tensorboardX import SummaryWriter
from json_dataset import SMASHDataset


class SmashRNN:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        # Basic config. Should be customizable in the future
        self.batch_size = 1
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.num_epoches = 2
        self.log_path = 'tensorboard/smash_rnn'
        self.early_stopping_minimum_delta = 0
        self.early_stopping_patience = 0

        train_dataset_path = './data/wiki_df_small.csv'
        word2vec_path = './data/glove.6B.50d.txt'
        # End of configs

        self.output_file = self.create_output_file()
        self.writer = SummaryWriter(self.log_path)

        self.max_word_length, self.max_sent_length, self.max_paragraph_length = get_max_lengths(train_dataset_path)

        self.training_dataset = SMASHDataset(train_dataset_path, word2vec_path, self.max_sent_length,
                                             self.max_word_length,
                                             self.max_paragraph_length)

        words_ids_current_document = [literal_eval(text) for text in self.training_dataset.current_article_text.values]
        words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document = self.get_padded_document_structures(
            words_ids_current_document)
        current_document_tensor = torch.LongTensor(
            [self.training_dataset.get_padded_document(current_document) for current_document in
             words_ids_current_document])

        words_ids_previous_document = [literal_eval(text) for text in
                                       self.training_dataset.previous_article_text.values]
        words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document = self.get_padded_document_structures(
            words_ids_previous_document)
        previous_document_tensor = torch.LongTensor(
            [self.training_dataset.get_padded_document(previous_document) for previous_document in
             words_ids_previous_document])

        click_rate_tensor = torch.Tensor([click_rate for click_rate in self.training_dataset.click_rate])

        self.dataset = TensorDataset(current_document_tensor, words_per_sentence_current_document,
                                     sentences_per_paragraph_current_document, paragraphs_per_document_current_document,
                                     previous_document_tensor, words_per_sentence_previous_document,
                                     sentences_per_paragraph_previous_document,
                                     paragraphs_per_document_previous_document,
                                     click_rate_tensor)

        # Load from txt file (in word2vec format)
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_dim = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_dim))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        # Siamese + Attention model
        self.model = SmashRNNModel(dict, dict_len, embed_dim)

        # Overall model optimization and evaluation parameters
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.learning_rate,
                                         momentum=self.momentum)
        self.best_loss = 1e5
        self.best_epoch = 0
        self.model.train()
        # self.num_iter_per_epoch = len(self.training_generator)

    def train(self):
        training_params = {'batch_size': self.batch_size,
                           'shuffle': True,
                           'drop_last': True}

        training_generator = DataLoader(self.dataset, **training_params)

        loss_list = []
        predictions_list = []
        step = 'train'

        for epoch in range(self.num_epoches):
            self.model.train()

            for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in training_generator:
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

                self.optimizer.zero_grad()
                predictions = self.model(current_document, words_per_sentence_current_document,
                                         sentences_per_paragraph_current_document,
                                         paragraphs_per_document_current_document,
                                         previous_document, words_per_sentence_previous_document,
                                         sentences_per_paragraph_previous_document,
                                         paragraphs_per_document_previous_document,
                                         click_rate_tensor)
                loss = self.criterion(predictions, click_rate_tensor)
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss)
                predictions_list.append(predictions.clone().cpu())

            loss = sum(loss_list) / self.dataset.__len__()

            print('Final loss: %1.4f' % loss)

            self.output_file.write(
                'Epoch: {}/{} \n{} loss: {}\n\n'.format(
                    epoch + 1,
                    self.num_epoches,
                    step.capitalize(),
                    loss
                ))

            print('Epoch: {}/{}, Lr: {}, Loss: {}'.format(
                epoch + 1,
                self.num_epoches,
                self.optimizer.param_groups[0]['lr'],
                loss
            ))

            self.writer.add_scalar('{}/Loss'.format(step.capitalize()), loss, epoch)

        return loss

    def get_padded_document_structures(self, words_ids_a):
        document_structures = [self.training_dataset.get_document_structure(document) for document in words_ids_a]
        words_per_sentences_tensor = torch.LongTensor([
            self.training_dataset.get_padded_words_per_sentence(document_structure['words_per_sentence']) for
            document_structure in document_structures])
        sentences_per_paragraph_tensor = torch.LongTensor([
            self.training_dataset.get_padded_sentence_per_paragraph(document_structure['sentences_per_paragraph']) for
            document_structure in document_structures])

        paragraphs_per_document_tensor = torch.LongTensor(
            [document_structure['paragraphs_per_document'] for document_structure in document_structures])

        return words_per_sentences_tensor, sentences_per_paragraph_tensor, paragraphs_per_document_tensor

    def is_best_loss(self, loss):
        return (loss + self.es_min_delta) < self.best_loss

    def save_model(self, loss, epoch):
        self.best_loss = loss
        self.best_epoch = epoch

        torch.save(self.model.state_dict(), './model.pt')

    def should_stop(self, epoch):
        return epoch - self.best_epoch > self.es_patience > 0

    def create_output_file(self):
        output_file = open('trained_models' + os.sep + 'logs.txt', 'a+')
        # TODO save all config values
        # output_file.write('Model\'s parameters: {}'.format(vars(self.opt)))

        return output_file


if __name__ == '__main__':
    model = SmashRNN()
    model.train()
