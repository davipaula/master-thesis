"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_max_lengths
from smash_rnn_model import SmashRNNModel
from tensorboardX import SummaryWriter
import argparse
import shutil
from json_dataset import SMASHDataset


class SmashRNN:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        self.opt = self.get_args()

        self.output_file = self.create_output_file()

        training_params = {'batch_size': self.opt.batch_size,
                           'shuffle': True,
                           'drop_last': True}

        # TODO work on model parameters (batch size) and change test_params to shuffle: false and drop_last:false
        test_params = {'batch_size': self.opt.batch_size,
                       'shuffle': True,
                       'drop_last': True}

        self.max_word_length, self.max_sent_length, self.max_paragraph_length = get_max_lengths(self.opt.train_set)
        self.training_dataset = SMASHDataset(self.opt.train_set, self.opt.word2vec_path, self.max_sent_length,
                                             self.max_word_length,
                                             self.max_paragraph_length)
        self.training_generator = DataLoader(self.training_dataset, **training_params)
        # TODO create validation & test datasets
        self.test_dataset = SMASHDataset(self.opt.test_set, self.opt.word2vec_path, self.max_sent_length,
                                         self.max_word_length,
                                         self.max_paragraph_length)
        self.test_generator = DataLoader(self.test_dataset, **test_params)

        self.model = SmashRNNModel(self.opt.word_hidden_size, self.opt.sent_hidden_size, self.opt.paragraph_hidden_size,
                                   self.opt.batch_size,
                                   self.training_dataset.num_classes,
                                   self.opt.word2vec_path, self.max_sent_length, self.max_word_length)

        if torch.cuda.is_available():
            self.model.cuda()

        if os.path.isdir(self.opt.log_path):
            shutil.rmtree(self.opt.log_path)
        os.makedirs(self.opt.log_path)

        self.writer = SummaryWriter(self.opt.log_path)
        # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.lr,
                                         momentum=self.opt.momentum)
        self.best_loss = 1e5
        self.best_epoch = 0
        self.model.train()
        self.num_iter_per_epoch = len(self.training_generator)

    def get_args(self):
        parser = argparse.ArgumentParser(
            '''Implementation of the model described in the paper: Hierarchical Attention Networks for Document 
            Classification''')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_epoches', type=int, default=1)  # 100
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--word_hidden_size', type=int, default=50)  # 50
        parser.add_argument('--sent_hidden_size', type=int,
                            default=13)  # 50 TODO this is hardcoded to match max_paragraph_length
        parser.add_argument('--paragraph_hidden_size', type=int,
                            default=13)  # 50 TODO this is hardcoded to match max_paragraph_length
        parser.add_argument('--es_min_delta', type=float, default=0.0,
                            help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
        parser.add_argument('--es_patience', type=int, default=0,
                            help='Early stopping\'s parameter: number of epochs with no improvement after which '
                                 'training will be stopped. Set to 0 to disable this technique.')
        parser.add_argument('--train_set', type=str, default='./data/wiki_df_small.csv')
        parser.add_argument('--test_set', type=str, default='./data/wiki_df_small.csv')
        parser.add_argument('--validation_interval', type=int, default=1,
                            help='Number of epoches between validation phases')
        parser.add_argument('--word2vec_path', type=str, default='./data/glove.6B.50d.txt')
        parser.add_argument('--log_path', type=str, default='tensorboard/han_voc')
        parser.add_argument('--saved_path', type=str, default='trained_models')
        args = parser.parse_args()

        return args

    def train(self):
        for epoch in range(self.opt.num_epoches):
            self.model.train()

            loss = self.run_model(epoch, 'train')

            # TODO implement validation step
            if self.should_run_validation(epoch):
                self.validate(epoch)

                if self.is_best_loss(loss):
                    self.save_model(loss, epoch)

                if self.should_stop(epoch):
                    print('Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, loss))
                    break

    def validate(self, epoch):
        self.model.eval()
        self.run_model(epoch, 'validate')

    def test(self):
        self.model.eval()
        self.run_model(1, 'test')

    def run_model(self, epoch, step='train'):
        loss_list = []
        predictions_list = []

        if step == 'validate':
            # TODO create `validation_generator` & `validation_dataset`
            generator = self.test_generator
            dataset = self.test_dataset
        elif step == 'test':
            generator = self.test_generator
            dataset = self.test_dataset
        else:
            generator = self.training_generator
            dataset = self.training_dataset

        for current_article_text, current_article_structure, current_article_title, previous_article_text, previous_article_structure, previous_article_title, click_rate in generator:
            if torch.cuda.is_available():
                current_article_text = current_article_text.cuda()
                previous_article_text = previous_article_text.cuda()
                click_rate = click_rate.cuda()

            if step == 'train':
                self.optimizer.zero_grad()
                predictions = self.get_predictions(current_article_text, current_article_structure, previous_article_text, previous_article_structure)
                loss = self.criterion(predictions, click_rate)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    predictions = self.get_predictions(current_article_text, current_article_structure, previous_article_text, previous_article_structure)
                loss = self.criterion(predictions, click_rate)

            loss_list.append(loss)
            predictions_list.append(predictions.clone().cpu())

        loss = sum(loss_list) / dataset.__len__()

        self.output_file.write(
            'Epoch: {}/{} \n{} loss: {}\n\n'.format(
                epoch + 1,
                self.opt.num_epoches,
                step.capitalize(),
                loss
            ))

        print('Epoch: {}/{}, Lr: {}, Loss: {}'.format(
            epoch + 1,
            self.opt.num_epoches,
            self.optimizer.param_groups[0]['lr'],
            loss
        ))

        self.writer.add_scalar('{}/Loss'.format(step.capitalize()), loss, epoch)

        return loss

    def should_run_validation(self, epoch):
        return (epoch % self.opt.validation_interval) == 0

    def get_predictions(self, current_article_text, current_article_structure, previous_article_text, previous_article_structure):
        self.model._init_hidden_state()
        predictions = self.model(current_article_text, current_article_structure, previous_article_text, previous_article_structure)

        return predictions

    def is_best_loss(self, loss):
        return (loss + self.opt.es_min_delta) < self.best_loss

    def save_model(self, loss, epoch):
        self.best_loss = loss
        self.best_epoch = epoch

        torch.save(self.model.state_dict(), './model.pt')

    def should_stop(self, epoch):
        return epoch - self.best_epoch > self.opt.es_patience > 0

    def create_output_file(self):
        output_file = open(self.opt.saved_path + os.sep + 'logs.txt', 'w')
        output_file.write('Model\'s parameters: {}'.format(vars(self.opt)))

        return output_file


if __name__ == '__main__':
    model = SmashRNN()
    model.train()
    # model.test()
