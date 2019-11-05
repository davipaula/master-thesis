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

        training_params = {"batch_size": self.opt.batch_size,
                           "shuffle": True,
                           "drop_last": True}

        # TODO work on model parameters (batch size) and change test_params to shuffle: false and drop_last:false
        test_params = {"batch_size": self.opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}

        self.max_word_length, self.max_sent_length, self.max_paragraph_length = get_max_lengths(self.opt.train_set)
        self.training_set = SMASHDataset(self.opt.train_set, self.opt.word2vec_path, self.max_sent_length,
                                         self.max_word_length,
                                         self.max_paragraph_length)
        self.training_generator = DataLoader(self.training_set, **training_params)
        # TODO create test dataset
        self.test_set = SMASHDataset(self.opt.test_set, self.opt.word2vec_path, self.max_sent_length,
                                     self.max_word_length,
                                     self.max_paragraph_length)
        self.test_generator = DataLoader(self.test_set, **test_params)

        self.model = SmashRNNModel(self.opt.word_hidden_size, self.opt.sent_hidden_size, self.opt.paragraph_hidden_size,
                                   self.opt.batch_size,
                                   self.training_set.num_classes,
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
            """Implementation of the model described in the paper: Hierarchical Attention Networks for Document 
            Classification""")
        parser.add_argument("--batch_size", type=int, default=24)
        parser.add_argument("--num_epoches", type=int, default=1)  # 100
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--word_hidden_size", type=int, default=50)  # 50
        parser.add_argument("--sent_hidden_size", type=int, default=13)  # 50
        parser.add_argument("--paragraph_hidden_size", type=int, default=13)  # 50
        parser.add_argument("--es_min_delta", type=float, default=0.0,
                            help="Early stopping's parameter: minimum change loss to qualify as an improvement")
        parser.add_argument("--es_patience", type=int, default=5,
                            help="Early stopping's parameter: number of epochs with no improvement after which training"
                                 " will be stopped. Set to 0 to disable this technique.")
        parser.add_argument("--train_set", type=str, default="./data/wiki_df_small.csv")
        parser.add_argument("--test_set", type=str, default="./data/wiki_df_small.csv")
        parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
        parser.add_argument("--word2vec_path", type=str, default="./data/glove.6B.50d.txt")
        parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
        parser.add_argument("--saved_path", type=str, default="trained_models")
        args = parser.parse_args()

        return args

    def train(self):
        for epoch in range(self.opt.num_epoches):
            for iter, (current_article_text, current_article_title, previous_article_text, previous_article_title,
                       click_rate) in enumerate(self.training_generator):
                if torch.cuda.is_available():
                    current_article_text = current_article_text.cuda()
                    previous_article_text = previous_article_text.cuda()
                    click_rate = click_rate.cuda()

                self.optimizer.zero_grad()
                self.model._init_hidden_state()
                predictions = self.model(current_article_text, previous_article_text)
                loss = self.criterion(predictions, click_rate)
                loss.backward()
                self.optimizer.step()

                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}".format(
                    epoch + 1,
                    self.opt.num_epoches,
                    iter + 1,
                    self.num_iter_per_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    loss))

                self.writer.add_scalar('Train/Loss', loss, epoch * self.num_iter_per_epoch + iter)

            torch.save(self.model.state_dict(), './model.pt')
            # TODO implement validation step
            # TODO move lines below to validation step

            # self.model.train()
            #
            # if test_loss + self.opt.es_min_delta < self.best_loss:
            #     self.best_loss = test_loss
            #     self.best_epoch = epoch
            #     torch.save(self.model, self.opt.saved_path + os.sep + "whole_model_han")
            #
            # # Early stopping
            # if epoch - self.best_epoch > self.opt.es_patience > 0:
            #     print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, test_loss))
            #     break

    def test(self):
        self.model.load_state_dict(torch.load('./model.pt'))

        for epoch in range(self.opt.num_epoches):
            if epoch % self.opt.test_interval == 0:
                self.model.eval()
                loss_list = []
                predictions_list = []

                for current_article_text, current_article_title, previous_article_text, previous_article_title, click_rate in self.test_generator:
                    if torch.cuda.is_available():
                        current_article_text = current_article_text.cuda()
                        previous_article_text = previous_article_text.cuda()

                    with torch.no_grad():
                        self.model._init_hidden_state()
                        predictions = self.model(current_article_text, previous_article_text)

                    loss = self.criterion(predictions, click_rate)
                    loss_list.append(loss)

                    predictions_list.append(predictions.clone().cpu())

                test_loss = sum(loss_list) / self.test_set.__len__()

                self.output_file.write(
                    "Epoch: {}/{} \nTest loss: {}\n\n".format(
                        epoch + 1, self.opt.num_epoches,
                        test_loss))

                print("Epoch: {}/{}, Lr: {}, Loss: {}".format(
                    epoch + 1,
                    self.opt.num_epoches,
                    self.optimizer.param_groups[0]['lr'],
                    test_loss
                ))

                self.writer.add_scalar('Test/Loss', test_loss, epoch)

    def create_output_file(self):
        output_file = open(self.opt.saved_path + os.sep + "logs.txt", "w")
        output_file.write("Model's parameters: {}".format(vars(self.opt)))

        return output_file


if __name__ == "__main__":
    model = SmashRNN()
    model.test()
