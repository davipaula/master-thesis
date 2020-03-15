"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import argparse
import csv
import os
import numpy as np
import pandas as pd
from comet_ml import Experiment
import torch
import torch.nn as nn
from src.utils import get_max_lengths, get_document_at_sentence_level, get_words_per_sentence_at_sentence_level, \
    get_document_at_word_level, get_sentences_per_paragraph_at_sentence_level, get_words_per_document_at_word_level
from src.smash_rnn_model import SmashRNNModel
from src.word_smash_rnn_model import WordLevelSmashRNNModel
from src.sentence_smash_rnn_model import SentenceLevelSmashRNNModel
from src.smash_dataset import SMASHDataset
from datetime import datetime
from src.dataset_creation import split_dataset
import sys


class SmashRNN:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        # Basic config. Should be customizable in the future
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.log_path = './tensorboard/smash_rnn'
        self.early_stopping_minimum_delta = 0
        self.early_stopping_patience = 0

        self.opt = self.get_args()
        self.batch_size = self.opt.batch_size
        self.num_validations = int(self.opt.num_epoches / self.opt.validation_interval)

        # End of configs

        print('Getting max lengths', datetime.now())
        self.max_word_length, self.max_sent_length, self.max_paragraph_length = get_max_lengths(
            self.opt.full_dataset_path,
            self.opt.limit_rows_dataset)
        print('Finished getting max lengths', datetime.now())

        self.complete_dataset = SMASHDataset(self.opt.full_dataset_path, self.opt.word2vec_path, self.max_sent_length,
                                             self.max_word_length, self.max_paragraph_length,
                                             self.opt.limit_rows_dataset)

        if self.opt.should_split_dataset:
            split_dataset(self.complete_dataset, self.opt.train_dataset_split, self.batch_size,
                          self.opt.train_dataset_path, self.opt.validation_dataset_path, self.opt.test_dataset_path)

        # Load from txt file (in word2vec format)
        dict = pd.read_csv(filepath_or_buffer=self.opt.word2vec_path, header=None, sep=" ",
                           quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_dim = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_dim))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        # Paragraph level model
        self.model = SmashRNNModel(dict, dict_len, embed_dim, self.max_word_length,
                                   self.max_sent_length,
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

        self.is_debug_mode = getattr(sys, 'gettrace', None) is None

        if not self.is_debug_mode:
            self.experiment = Experiment(api_key="NPD7aHoJxhZgG0MNWBkFb3hzZ",
                                         project_name="thesis-davi",
                                         workspace="davipaula")

    def train(self, level='paragraph'):
        training_generator = torch.load(self.opt.train_dataset_path)
        print('Starting training {}'.format(datetime.now()))

        for epoch in range(self.opt.num_epoches):
            self.model.train()

            loss_list = []
            predictions_list = []

            for current_document, previous_document, click_rate_tensor in training_generator:
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

                self.optimizer.zero_grad()

                if level == 'sentence':
                    current_document = self.transform_to_sentence_level(current_document)
                    previous_document = self.transform_to_sentence_level(previous_document)

                elif level == 'word':
                    current_document = self.transform_to_word_level(current_document)
                    previous_document = self.transform_to_word_level(previous_document)

                predictions = self.model(current_document['text'],
                                         current_document['words_per_sentence'],
                                         current_document['sentences_per_paragraph'],
                                         current_document['paragraphs_per_document'],
                                         previous_document['text'],
                                         previous_document['words_per_sentence'],
                                         previous_document['sentences_per_paragraph'],
                                         previous_document['paragraphs_per_document'],
                                         click_rate_tensor)

                loss = self.criterion(predictions, click_rate_tensor)
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss)
                predictions_list.append(predictions.clone().cpu())

            loss = self.calculate_loss(loss_list)

            if not self.is_debug_mode:
                self.experiment.log_metric('train_paragraph_level_loss', loss.item(), epoch=epoch + 1)

            print('Epoch: {}/{}, Lr: {}, Loss: {}, Time: {}'.format(
                epoch + 1,
                self.opt.num_epoches,
                self.optimizer.param_groups[0]['lr'],
                loss,
                datetime.now()
            ))

            if self.should_run_validation(epoch):
                self.validate(int(epoch / self.opt.validation_interval), level)

            self.save_model(self.model, loss, epoch)
            print('Training finished {}'.format(datetime.now()))

    def transform_to_word_level(self, document):
        batch_size = document['text'].shape[0]

        document['words_per_sentence'] = get_words_per_document_at_word_level(document['words_per_sentence'])
        document['text'] = get_document_at_word_level(document['text'], document['words_per_sentence'])
        document['sentences_per_paragraph'] = torch.ones((batch_size, 1), dtype=int)
        document['paragraphs_per_document'] = torch.ones((batch_size, 1), dtype=int)

        return document

    def transform_to_sentence_level(self, document):
        batch_size = document['text'].shape[0]

        document['text'] = get_document_at_sentence_level(document['text'])
        document['words_per_sentence'] = get_words_per_sentence_at_sentence_level(document['words_per_sentence'])
        document['sentences_per_paragraph'] = get_sentences_per_paragraph_at_sentence_level(
            document['sentences_per_paragraph'])
        document['paragraphs_per_document'] = torch.ones((batch_size, 1), dtype=int)

        return document

    def validate(self, validation_step, level):
        validation_generator = torch.load(self.opt.validation_dataset_path)
        validation_step = int(validation_step) + 1

        loss_list = []
        predictions_list = []

        for current_document, previous_document, click_rate_tensor in validation_generator:
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

            with torch.no_grad():
                if level == 'sentence':
                    current_document = self.transform_to_sentence_level(current_document)
                    previous_document = self.transform_to_sentence_level(previous_document)

                elif level == 'word':
                    current_document = self.transform_to_word_level(current_document)
                    previous_document = self.transform_to_word_level(previous_document)

                predictions = self.model(current_document['text'],
                                         current_document['words_per_sentence'],
                                         current_document['sentences_per_paragraph'],
                                         current_document['paragraphs_per_document'],
                                         previous_document['text'],
                                         previous_document['words_per_sentence'],
                                         previous_document['sentences_per_paragraph'],
                                         previous_document['paragraphs_per_document'],
                                         click_rate_tensor)

            loss = self.criterion(predictions, click_rate_tensor)

            loss_list.append(loss)
            predictions_list.append(predictions.clone().cpu())

        loss = self.calculate_loss(loss_list)

        if not self.is_debug_mode:
            self.experiment.log_metric('validation_loss', loss.item(), epoch=validation_step)

        print('{} level\n Validation: {}/{}, Lr: {}, Loss: {}'.format(
            level.capitalize(),
            validation_step,
            self.num_validations,
            self.optimizer.param_groups[0]['lr'],
            loss
        ))

    def is_best_loss(self, loss):
        return (loss + self.es_min_delta) < self.best_loss

    def save_model(self, model, loss, epoch):
        self.best_loss = loss
        self.best_epoch = epoch

        model_path = self.opt.model_folder + os.sep + self.opt.level + '_level_model.pt'
        torch.save(model.state_dict(), model_path)

    def should_stop(self, epoch):
        return epoch - self.best_epoch > self.es_patience > 0

    def should_run_validation(self, epoch):
        return ((epoch + 1) % self.opt.validation_interval) == 0

    @staticmethod
    def calculate_loss(loss_list):
        return sum(loss_list) / len(loss_list)

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            """Implementation of the model described in the paper: Semantic Text Matching for Long-Form Documents to predict the number of clicks for Wikipedia articles""")
        parser.add_argument("--model_folder", type=str, default='./trained_models/')
        parser.add_argument("--full_dataset_path", type=str, default='./data/wiki_df.csv')
        parser.add_argument("--word2vec_path", type=str, default='./data/glove.6B.50d.txt')
        parser.add_argument("--train_dataset_path", type=str, default='./data/training.pth')
        parser.add_argument("--validation_dataset_path", type=str,
                            default='./data/validation.pth')
        parser.add_argument("--test_dataset_path", type=str, default='./data/test.pth')
        parser.add_argument("--num_epoches", type=int, default=1)
        parser.add_argument("--validation_interval", type=int, default=1)
        parser.add_argument("--should_split_dataset", type=bool, default=False)
        parser.add_argument("--train_dataset_split", type=float, default=0.8)
        parser.add_argument("--limit_rows_dataset", type=int, default=9999999,
                            help='For development purposes. This limits the number of rows read from the dataset.')
        parser.add_argument("--level", type=str, default='paragraph')
        parser.add_argument("--batch_size", type=int, default=3)

        return parser.parse_args()

    def run(self):
        model.train(self.opt.level)


if __name__ == '__main__':
    model = SmashRNN()
    model.run()
