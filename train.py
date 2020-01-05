"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import argparse
import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.utils import get_max_lengths
from src.smash_rnn_model import SmashRNNModel
from tensorboardX import SummaryWriter
from src.smash_dataset import SMASHDataset
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
        self.log_path = 'tensorboard/smash_rnn'
        self.early_stopping_minimum_delta = 0
        self.early_stopping_patience = 0

        self.opt = self.get_args()
        self.num_validations = int(self.opt.num_epoches / self.opt.validation_interval)

        # End of configs

        self.output_file = self.create_output_file()
        self.writer = SummaryWriter(self.log_path)

        print('Getting max lengths', datetime.now())
        self.max_word_length, self.max_sent_length, self.max_paragraph_length = get_max_lengths(
            self.opt.full_dataset_path,
            self.opt.limit_rows_dataset)
        print('Finished getting max lengths', datetime.now())

        self.complete_dataset = SMASHDataset(self.opt.full_dataset_path, self.opt.word2vec_path, self.max_sent_length,
                                             self.max_word_length,
                                             self.max_paragraph_length, self.opt.limit_rows_dataset)

        if self.opt.should_split_dataset:
            self.split_dataset(self.opt.train_dataset_split)

        # Load from txt file (in word2vec format)
        dict = pd.read_csv(filepath_or_buffer=self.opt.word2vec_path, header=None, sep=" ",
                           quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_dim = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_dim))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        # Siamese + Attention model
        self.model = SmashRNNModel(dict, dict_len, embed_dim, self.max_word_length, self.max_sent_length,
                                   self.max_paragraph_length)

        if torch.cuda.is_available():
            self.model.cuda()

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
        training_generator = torch.load(self.opt.train_dataset_path)
        print('Starting training {}'.format(datetime.now()))

        # Trying to avoid TensorDataset. Didn't work
        # training_generator = torch.utils.data.DataLoader(self.complete_dataset, batch_size=self.batch_size)

        step = 'train'

        for epoch in range(self.opt.num_epoches):
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
                    self.opt.num_epoches,
                    step.capitalize(),
                    loss
                ))

            print('Epoch: {}/{}, Lr: {}, Loss: {}, Time: {}'.format(
                epoch + 1,
                self.opt.num_epoches,
                self.optimizer.param_groups[0]['lr'],
                loss,
                datetime.now()
            ))

            self.writer.add_scalar('{}/Loss'.format(step.capitalize()), loss, epoch)

            if self.should_run_validation(epoch):
                self.validate(int(epoch / self.opt.validation_interval))

            torch.save(self.model.state_dict(), self.opt.model_path)
            print('Training finished {}'.format(datetime.now()))

    def validate(self, validation_step):
        validation_generator = torch.load(self.opt.validation_dataset_path)
        validation_step = int(validation_step) + 1

        step = 'validation'

        loss_list = []
        predictions_list = []

        for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in validation_generator:
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

        torch.save(self.model.state_dict(), self.opt.model_path)

    def should_stop(self, epoch):
        return epoch - self.best_epoch > self.es_patience > 0

    def should_run_validation(self, epoch):
        return ((epoch + 1) % self.opt.validation_interval) == 0

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

        torch.save(train_loader, self.opt.train_dataset_path)
        print('Training dataset saved', datetime.now())

        torch.save(validation_loader, self.opt.validation_dataset_path)
        print('Validation dataset saved', datetime.now())

        torch.save(test_loader, self.opt.test_dataset_path)
        print('Test dataset saved', datetime.now())

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            """Implementation of the model described in the paper: Semantic Text Matching for Long-Form Documents to predict the number of clicks for Wikipedia articles""")
        parser.add_argument("--model_path", type=str, default='./trained_models/model.pt')
        parser.add_argument("--full_dataset_path", type=str, default='./data/wiki_df.csv')
        parser.add_argument("--word2vec_path", type=str, default='./data/glove.6B.50d.txt')
        parser.add_argument("--train_dataset_path", type=str, default='./data/training.pth')
        parser.add_argument("--validation_dataset_path", type=str, default='./data/validation.pth')
        parser.add_argument("--test_dataset_path", type=str, default='./data/test.pth')
        parser.add_argument("--num_epoches", type=int, default=1)
        parser.add_argument("--validation_interval", type=int, default=1)
        parser.add_argument("--should_split_dataset", type=bool, default=False)
        parser.add_argument("--train_dataset_split", type=float, default=0.8)
        parser.add_argument("--limit_rows_dataset", type=int, default=9999,
                            help='For development purposes. This limits the number of rows read from the dataset. Change to None to ignore it')

        return parser.parse_args()


if __name__ == '__main__':
    model = SmashRNN()
    model.train()
