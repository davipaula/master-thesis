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
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils import get_max_lengths
from smash_rnn_model import SmashRNNModel
from tensorboardX import SummaryWriter
from smash_dataset import SMASHDataset
from datetime import datetime


class SmashRNN:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(456)
        else:
            torch.manual_seed(456)

        # Basic config. Should be customizable in the future
        self.batch_size = 1
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.num_epoches = 6
        self.validation_interval = 3
        self.num_validations = int(self.num_epoches / self.validation_interval)
        self.log_path = 'tensorboard/smash_rnn'
        self.early_stopping_minimum_delta = 0
        self.early_stopping_patience = 0
        self.should_split_dataset = False
        self.train_dataset_split = 0.8
        # For development purposes. This limits the number of rows read from the dataset
        limit_rows_dataset = 9999

        train_dataset_path = './data/wiki_df_small.csv'
        word2vec_path = './data/glove.6B.50d.txt'
        # End of configs

        self.output_file = self.create_output_file()
        self.writer = SummaryWriter(self.log_path)

        print('Getting max lengths', datetime.now())
        self.max_word_length, self.max_sent_length, self.max_paragraph_length = get_max_lengths(train_dataset_path,
                                                                                                limit_rows_dataset)
        print('Finished getting max lengths', datetime.now())

        self.complete_dataset = SMASHDataset(train_dataset_path, word2vec_path, self.max_sent_length,
                                             self.max_word_length,
                                             self.max_paragraph_length, limit_rows_dataset)

        if self.should_split_dataset:
            self.split_dataset(self.train_dataset_split)

        # Load from txt file (in word2vec format)
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_dim = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_dim))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        # Siamese + Attention model
        self.model = SmashRNNModel(dict, dict_len, embed_dim, self.max_word_length, self.max_sent_length,
                                   self.max_paragraph_length)

        # Overall model optimization and evaluation parameters
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.learning_rate,
                                         momentum=self.momentum)
        self.best_loss = 1e5
        self.best_epoch = 0
        self.model.train()
        # self.num_iter_per_epoch = len(self.training_generator)

    def save_tensor_dataset(self, words_ids, name):
        words_per_sentence, sentences_per_paragraph, paragraphs_per_document = self.get_padded_document_structures(
            words_ids)
        print('Transforming dataset into tensors', datetime.now())
        document_tensor = self.get_document_tensor(words_ids)
        print('Finished Transforming dataset into tensors', datetime.now())
        print('Beginning creation of TensorDataset', datetime.now())
        dataset = TensorDataset(document_tensor, words_per_sentence, sentences_per_paragraph, paragraphs_per_document)

        torch.save(dataset, name + '.pth')
        print('Saved TensorDataset {} {}'.format(name, datetime.now()))

        del document_tensor, words_per_sentence, sentences_per_paragraph, paragraphs_per_document, dataset

    def get_document_tensor(self, documents):
        __slot__ = ('document_placeholder')
        num_documents = len(documents)

        document_placeholder = torch.zeros(
            (num_documents, self.max_paragraph_length, self.max_sent_length, self.max_word_length), dtype=int)

        print('Number of documents: {}'.format(num_documents))

        for document_idx, document in enumerate(documents):
            if (document_idx % 100) == 0:
                print('Finished document {} of {}. {}'.format(document_idx, num_documents, datetime.now()))
            for paragraph_idx, paragraph in enumerate(document):
                for sentence_idx, sentence in enumerate(paragraph):
                    document_placeholder[document_idx, paragraph_idx, sentence_idx, 0:len(sentence)] = torch.LongTensor(
                        sentence)

        return document_placeholder

    def train(self):
        training_generator = torch.load('./data/training.pth')
        print('Starting training')

        # Trying to avoid TensorDataset. Didn't work
        # training_generator = torch.utils.data.DataLoader(self.complete_dataset, batch_size=self.batch_size)

        step = 'train'

        for epoch in range(self.num_epoches):
            self.model.train()

            loss_list = []
            predictions_list = []

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

            loss = sum(loss_list) / training_generator.dataset.__len__()

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

            if self.should_run_validation(epoch):
                self.validate(int(epoch / self.validation_interval))

            torch.save(self.model.state_dict(), './model.pt')

    def validate(self, validation_step):
        validation_generator = torch.load('./data/validation.pth')
        validation_step = int(validation_step) + 1

        step = 'validation'

        for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in validation_generator:
            loss_list = []
            predictions_list = []

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

            with torch.no_grad():
                predictions = self.model(current_document, words_per_sentence_current_document,
                                         sentences_per_paragraph_current_document,
                                         paragraphs_per_document_current_document,
                                         previous_document, words_per_sentence_previous_document,
                                         sentences_per_paragraph_previous_document,
                                         paragraphs_per_document_previous_document,
                                         click_rate_tensor)
            loss = self.criterion(predictions, click_rate_tensor)

            loss_list.append(loss)
            predictions_list.append(predictions.clone().cpu())

        loss = sum(loss_list) / validation_generator.dataset.__len__()

        self.output_file.write(
            'Validation: {}/{} \n{} loss: {}\n\n'.format(
                validation_step,
                self.num_validations,
                step.capitalize(),
                loss
            ))

        print('Validation: {}/{}, Lr: {}, Loss: {}'.format(
            validation_step,
            self.num_validations,
            self.optimizer.param_groups[0]['lr'],
            loss
        ))

        self.writer.add_scalar('{}/Loss'.format(step.capitalize()), loss, validation_step)

    def get_padded_document_structures(self, words_ids_a):
        document_structures = [self.complete_dataset.get_document_structure(document) for document in words_ids_a]
        words_per_sentences_tensor = torch.LongTensor([
            self.complete_dataset.get_padded_words_per_sentence(document_structure['words_per_sentence']) for
            document_structure in document_structures])
        sentences_per_paragraph_tensor = torch.LongTensor([
            self.complete_dataset.get_padded_sentences_per_paragraph(document_structure['sentences_per_paragraph']) for
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

    def should_run_validation(self, epoch):
        return ((epoch + 1) % self.validation_interval) == 0

    def create_output_file(self):
        output_file = open('trained_models' + os.sep + 'logs.txt', 'a+')
        # TODO save all config values
        # output_file.write('Model\'s parameters: {}'.format(vars(self.opt)))

        return output_file

    def split_dataset(self, train_split):
        print('Beginning of dataset split')
        dataset_size = len(self.complete_dataset)
        train_dataset_size = int(dataset_size * train_split)
        validation_dataset_size = int((dataset_size - train_dataset_size) / 2)
        test_dataset_size = dataset_size - train_dataset_size - validation_dataset_size

        train_dataset, validation_dataset, test_dataset = random_split(self.complete_dataset,
                                                                       [train_dataset_size, validation_dataset_size,
                                                                        test_dataset_size])

        print('Datasets split. Starting saving them', datetime.now())

        training_params = {'batch_size': self.batch_size,
                           'shuffle': True,
                           'drop_last': True}
        train_loader = torch.utils.data.DataLoader(train_dataset, **training_params)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, **training_params)

        test_params = {'batch_size': self.batch_size,
                       'shuffle': True,
                       'drop_last': False}
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_params)

        torch.save(train_loader, './data/training.pth')
        print('Training dataset saved', datetime.now())

        torch.save(validation_loader, './data/validation.pth')
        print('Validation dataset saved', datetime.now())

        torch.save(test_loader, './data/test.pth')
        print('Test dataset saved', datetime.now())


if __name__ == '__main__':
    model = SmashRNN()
    model.train()
