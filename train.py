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
from src.utils import get_max_lengths
from src.smash_rnn_model import SmashRNNModel
from src.word_smash_rnn_model import WordLevelSmashRNNModel
from src.sentence_smash_rnn_model import SentenceSmashRNNModel
from tensorboardX import SummaryWriter
from src.smash_dataset import SMASHDataset
from datetime import datetime
from src.dataset_creation import split_dataset


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
        self.log_path = './tensorboard/smash_rnn'
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
        self.paragraph_level_model = SmashRNNModel(dict, dict_len, embed_dim, self.max_word_length,
                                                   self.max_sent_length,
                                                   self.max_paragraph_length)

        # Sentence level model
        self.sentence_level_model = SentenceSmashRNNModel(dict, dict_len, embed_dim, self.max_word_length,
                                                          self.max_sent_length)

        # Word level model
        self.word_level_model = WordLevelSmashRNNModel(dict, dict_len, embed_dim, self.max_word_length,
                                                       self.max_sent_length,
                                                       self.max_paragraph_length)

        if torch.cuda.is_available():
            self.paragraph_level_model.cuda()
            self.sentence_level_model.cuda()
            self.word_level_model.cuda()

        # Overall model optimization and evaluation parameters
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.paragraph_level_model.parameters()),
                                         lr=self.learning_rate,
                                         momentum=self.momentum)
        self.best_loss = 1e5
        self.best_epoch = 0

        self.paragraph_level_model.train()
        self.sentence_level_model.train()
        self.word_level_model.train()

        self.experiment = Experiment(api_key="NPD7aHoJxhZgG0MNWBkFb3hzZ",
                                     project_name="thesis-davi",
                                     workspace="davipaula")

    def train(self):
        training_generator = torch.load(self.opt.train_dataset_path)
        print('Starting training {}'.format(datetime.now()))

        step = 'train'

        for epoch in range(self.opt.num_epoches):
            self.paragraph_level_model.train()

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
                predictions = self.paragraph_level_model(current_document, words_per_sentence_current_document,
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

            self.experiment.log_metric('train_paragraph_level_loss', loss.item(), epoch=epoch + 1)

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
                self.validate(int(epoch / self.opt.validation_interval), 'paragraph')

        torch.save(self.paragraph_level_model.state_dict(), self.opt.model_path)
        print('Training finished {}'.format(datetime.now()))

    def validate(self, validation_step, level):
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
                if level == 'paragraph':
                    predictions = self.paragraph_level_model(current_document, words_per_sentence_current_document,
                                                             sentences_per_paragraph_current_document,
                                                             paragraphs_per_document_current_document,
                                                             previous_document, words_per_sentence_previous_document,
                                                             sentences_per_paragraph_previous_document,
                                                             paragraphs_per_document_previous_document,
                                                             click_rate_tensor)
                elif level == 'sentence':
                    predictions = self.sentence_level_model(current_document, words_per_sentence_current_document,
                                                            sentences_per_paragraph_current_document,
                                                            previous_document, words_per_sentence_previous_document,
                                                            sentences_per_paragraph_previous_document,
                                                            click_rate_tensor)
                elif level == 'word':
                    predictions = self.word_level_model(current_document, words_per_sentence_current_document,
                                                        previous_document, words_per_sentence_previous_document,
                                                        click_rate_tensor)

            loss = self.criterion(predictions, click_rate_tensor)

            loss_list.append(loss)
            predictions_list.append(predictions.clone().cpu())

        loss = sum(loss_list) / validation_generator.dataset.__len__()

        self.experiment.log_metric('validation_loss', loss.item(), epoch=validation_step)

        self.output_file.write(
            '{} level\n Validation: {}/{} \n{} loss: {}\n\n'.format(
                level.capitalize(),
                validation_step,
                self.num_validations,
                step.capitalize(),
                loss
            ))

        print('{} level\n Validation: {}/{}, Lr: {}, Loss: {}'.format(
            level.capitalize(),
            validation_step,
            self.num_validations,
            self.optimizer.param_groups[0]['lr'],
            loss
        ))

        self.writer.add_scalar('{}/Loss'.format(step.capitalize()), loss, validation_step)

    def is_best_loss(self, loss):
        return (loss + self.es_min_delta) < self.best_loss

    def save_model(self, loss, epoch):
        self.best_loss = loss
        self.best_epoch = epoch

        torch.save(self.paragraph_level_model.state_dict(), self.opt.model_path)

    def should_stop(self, epoch):
        return epoch - self.best_epoch > self.es_patience > 0

    def should_run_validation(self, epoch):
        return ((epoch + 1) % self.opt.validation_interval) == 0

    def create_output_file(self):
        output_file = open('trained_models' + os.sep + 'logs.txt', 'a+')
        # TODO save all config values
        # output_file.write('Model\'s parameters: {}'.format(vars(self.opt)))

        return output_file

    def train_word_level(self):
        training_generator = torch.load(self.opt.train_dataset_path)
        print('Starting training {}'.format(datetime.now()))

        step = 'train'

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.word_level_model.parameters()),
                                    lr=self.learning_rate,
                                    momentum=self.momentum)

        for epoch in range(self.opt.num_epoches):
            self.word_level_model.train()

            loss_list = []
            predictions_list = []

            for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in training_generator:
                # current_document = get_document_at_word_level(current_document, words_per_sentence_current_document)
                # previous_document = get_document_at_word_level(previous_document, words_per_sentence_previous_document)

                if torch.cuda.is_available():
                    current_document = current_document.cuda()
                    words_per_sentence_current_document = words_per_sentence_current_document.cuda()
                    previous_document = previous_document.cuda()
                    words_per_sentence_previous_document = words_per_sentence_previous_document.cuda()
                    click_rate_tensor = click_rate_tensor.cuda()

                optimizer.zero_grad()
                predictions = self.word_level_model(current_document, words_per_sentence_current_document,
                                                    previous_document, words_per_sentence_previous_document,
                                                    click_rate_tensor)
                loss = self.criterion(predictions, click_rate_tensor)
                loss.backward()
                optimizer.step()

                loss_list.append(loss)
                predictions_list.append(predictions.clone().cpu())

            loss = sum(loss_list) / training_generator.dataset.__len__()

            self.experiment.log_metric('train_word_level_loss', loss.item(), epoch=epoch + 1)

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
                optimizer.param_groups[0]['lr'],
                loss,
                datetime.now()
            ))

            self.writer.add_scalar('{}/Loss'.format(step.capitalize()), loss, epoch)

            if self.should_run_validation(epoch):
                self.validate(int(epoch / self.opt.validation_interval), 'word')

        torch.save(self.word_level_model.state_dict(), self.opt.model_path)
        print('Training finished {}'.format(datetime.now()))

    def train_sentence_level(self):
        training_generator = torch.load(self.opt.train_dataset_path)

        print('Starting training {}'.format(datetime.now()))

        # Trying to avoid TensorDataset. Didn't work
        # training_generator = torch.utils.data.DataLoader(self.complete_dataset, batch_size=self.batch_size)

        step = 'train'

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.sentence_level_model.parameters()),
                                    lr=self.learning_rate,
                                    momentum=self.momentum)

        for epoch in range(self.opt.num_epoches):
            self.sentence_level_model.train()

            loss_list = []
            predictions_list = []

            for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in training_generator:
                if torch.cuda.is_available():
                    current_document = current_document.cuda()
                    words_per_sentence_current_document = words_per_sentence_current_document.cuda()
                    previous_document = previous_document.cuda()
                    words_per_sentence_previous_document = words_per_sentence_previous_document.cuda()
                    click_rate_tensor = click_rate_tensor.cuda()

                optimizer.zero_grad()
                predictions = self.sentence_level_model(current_document, words_per_sentence_current_document,
                                                        sentences_per_paragraph_current_document,
                                                        previous_document, words_per_sentence_previous_document,
                                                        sentences_per_paragraph_previous_document,
                                                        click_rate_tensor)
                loss = self.criterion(predictions, click_rate_tensor)
                loss.backward()
                optimizer.step()

                loss_list.append(loss)
                predictions_list.append(predictions.clone().cpu())

            loss = sum(loss_list) / training_generator.dataset.__len__()

            self.experiment.log_metric('train_sentence_level_loss', loss.item(), epoch=epoch + 1)

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
                optimizer.param_groups[0]['lr'],
                loss,
                datetime.now()
            ))

            self.writer.add_scalar('{}/Loss'.format(step.capitalize()), loss, epoch)

            if self.should_run_validation(epoch):
                self.validate(int(epoch / self.opt.validation_interval), 'sentence')

        torch.save(self.paragraph_level_model.state_dict(), self.opt.model_path)
        print('Training finished {}'.format(datetime.now()))

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            """Implementation of the model described in the paper: Semantic Text Matching for Long-Form Documents to predict the number of clicks for Wikipedia articles""")
        parser.add_argument("--model_path", type=str, default='/home/dnascimento/thesis-davi/trained_models/model.pt')
        parser.add_argument("--full_dataset_path", type=str, default='/home/dnascimento/thesis-davi/data/wiki_df.csv')
        parser.add_argument("--word2vec_path", type=str, default='/home/dnascimento/thesis-davi/data/glove.6B.50d.txt')
        parser.add_argument("--train_dataset_path", type=str, default='/home/dnascimento/thesis-davi/data/training.pth')
        parser.add_argument("--validation_dataset_path", type=str,
                            default='/home/dnascimento/thesis-davi/data/validation.pth')
        parser.add_argument("--test_dataset_path", type=str, default='/home/dnascimento/thesis-davi/data/test.pth')
        parser.add_argument("--num_epoches", type=int, default=1)
        parser.add_argument("--validation_interval", type=int, default=1)
        parser.add_argument("--should_split_dataset", type=bool, default=False)
        parser.add_argument("--train_dataset_split", type=float, default=0.8)
        parser.add_argument("--limit_rows_dataset", type=int, default=9999999,
                            help='For development purposes. This limits the number of rows read from the dataset.')

        # parser = argparse.ArgumentParser(
        #     """Implementation of the model described in the paper: Semantic Text Matching for Long-Form Documents to predict the number of clicks for Wikipedia articles""")
        # parser.add_argument("--model_path", type=str, default='./trained_models/model.pt')
        # parser.add_argument("--full_dataset_path", type=str, default='./data/wiki_df.csv')
        # parser.add_argument("--word2vec_path", type=str, default='./data/glove.6B.50d.txt')
        # parser.add_argument("--train_dataset_path", type=str, default='./data/training.pth')
        # parser.add_argument("--validation_dataset_path", type=str, default='./data/validation.pth')
        # parser.add_argument("--test_dataset_path", type=str, default='./data/test.pth')
        # parser.add_argument("--num_epoches", type=int, default=1)
        # parser.add_argument("--validation_interval", type=int, default=1)
        # parser.add_argument("--should_split_dataset", type=bool, default=False)
        # parser.add_argument("--train_dataset_split", type=float, default=0.8)
        # parser.add_argument("--limit_rows_dataset", type=int, default=9999,
        #                     help='For development purposes. This limits the number of rows read from the dataset. Change to None to ignore it')

        return parser.parse_args()


if __name__ == '__main__':
    model = SmashRNN()
    # model.train_word_level()
    # model.train_sentence_level()
    model.train()
