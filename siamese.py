# coding=utf-8
import csv
import os
import yaml
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

import json

# utils
from siamese.utils import get_embedding, load_embed, save_embed
# data
from siamese.data import OriginalMyDS, OriginalMytestDS

# model
from siamese.model import Siamese_lstm

from nltk.tokenize import sent_tokenize, word_tokenize

FLAGS = None


def main(_):
    # Load the configuration file.
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    print('**********', config['experiment_name'], '**********')

    """ Cuda Check """
    if torch.cuda.is_available():
        print('Using GPU!')
    else:
        print('No GPU!')

    """ Read Data & Get Embedding """
    train_dataset_path = config['train_dataset_path']
    test_dataset_path = config['test_dataset_path']

    """ Data Preprocessing """

    if config['data_preprocessing']:
        print('Pre-processing Original Data ...')
        get_max_lengths(train_dataset_path)
        print('Data Pre-processing Done!')

    with open('./siamese_max_lengths.txt', 'r') as json_file:
        max_lengths = json.load(json_file)

    # max_word_length = max_lengths['max_word_length']
    # max_sent_length = max_lengths['max_sent_length']

    # max_word_length = 1
    # max_sent_length = 1

    train_data = create_dataset_for_siamese(train_dataset_path)
    test_data = create_dataset_for_siamese(test_dataset_path)

    # split dataset
    msk = np.random.rand(len(train_data)) < 0.8
    train = train_data[msk]
    valid = train_data[~msk]

    full_embed_path = config['embedding']['full_embedding_path']
    cur_embed_path = config['embedding']['cur_embedding_path']

    # dataset
    all_sentences = train_data['review_1'].to_list() + train_data['review_2'].to_list()

    trainDS = OriginalMyDS(train, all_sentences)
    validDS = OriginalMyDS(valid, all_sentences)

    print('Data size:', train_data.shape[0], test_data.shape[0])

    # embed_dict = trainDS.dict

    if os.path.exists(cur_embed_path) and not config['make_dict']:
        embed_dict = load_embed(cur_embed_path)
        print('Loaded existing embedding.')
    else:
        print('Making embedding...')
        embed_dict = get_embedding(trainDS.vocab._id2word, full_embed_path)
        save_embed(embed_dict,cur_embed_path)
        print('Saved generated embedding.')

    vocab_size = len(embed_dict)
    # initialize nn embedding
    embedding = nn.Embedding(vocab_size, config['model']['embed_size'])
    embed_list = []
    for word in trainDS.vocab._id2word:
        embed_list.append(embed_dict[word])
    weight_matrix = np.array(embed_list)

    # TODO check how to adapt this piece of code when using GLOVE embeddings
    # # pass weights to nn embedding
    # embedding.weight = nn.Parameter(torch.from_numpy(weight_matrix).type(torch.FloatTensor), requires_grad=False)

    """ Model Preparation """

    # embedding
    config['embedding_matrix'] = embedding
    config['vocab_size'] = len(embed_dict)

    # model
    siamese = Siamese_lstm(config)
    print(siamese)

    # loss func
    loss_weights = Variable(torch.FloatTensor([1, 3]))
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(loss_weights)

    # optimizer
    learning_rate = config['training']['learning_rate']
    if config['training']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
    elif config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
    elif config['training']['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
    elif config['training']['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda x: x.requires_grad, siamese.parameters()), lr=learning_rate)
    print('Optimizer:', config['training']['optimizer'])
    print('Learning rate:', config['training']['learning_rate'])

    # log info
    train_log_string = '%s :: Epoch %i :: Iter %i / %i :: train loss: %0.4f'
    valid_log_string = '%s :: Epoch %i :: valid loss: %0.4f\n'

    # Restore saved model (if one exists).
    ckpt_path = os.path.join(config['ckpt_dir'], config['experiment_name'] + '.pt')

    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        siamese.load_state_dict(ckpt['siamese'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        epoch = 1
        print('Fresh start!\n')

    # if torch.cuda.is_available():
    #     criterion = criterion.cuda()
    #     siamese = siamese.cuda()

    """ Train """

    if config['task'] == 'train':

        # save every epoch for visualization
        train_loss_record = []
        valid_loss_record = []
        best_record = 10.0

        # training
        print('Experiment: {}\n'.format(config['experiment_name']))

        while epoch < config['training']['num_epochs']:

            print('Start Epoch {} Training...'.format(epoch))

            # loss
            train_loss = []
            train_loss_sum = []
            # dataloader
            train_dataloader = DataLoader(dataset=trainDS, shuffle=True, num_workers=2, batch_size=1)

            for idx, data in enumerate(train_dataloader, 0):

                # get data
                s1, s2, label = data

                # clear gradients
                optimizer.zero_grad()

                # input
                output = siamese(s1, s2)
                output = output.squeeze(0)

                # label cuda
                label = Variable(label)
                if torch.cuda.is_available():
                    label = label.cuda()

                # loss backward
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.data.cpu())
                train_loss_sum.append(loss.data.cpu())

                # Every once and a while check on the loss
                if ((idx + 1) % 5000) == 0:
                    print(train_log_string % (datetime.now(), epoch, idx + 1, len(train), np.mean(train_loss)))
                    train_loss = []

            # Record at every epoch
            print('Train Loss at epoch {}: {}\n'.format(epoch, np.mean(train_loss_sum)))
            train_loss_record.append(np.mean(train_loss_sum))

            # Valid
            print('Epoch {} Validating...'.format(epoch))

            # loss
            valid_loss = []
            # dataloader
            valid_dataloader = DataLoader(dataset=validDS, shuffle=True, num_workers=2, batch_size=1)

            for idx, data in enumerate(valid_dataloader, 0):
                # get data
                s1, s2, label = data

                # input
                output = siamese(s1, s2)
                output = output.squeeze(0)

                # label cuda
                label = Variable(label)
                if torch.cuda.is_available():
                    label = label.cuda()

                # loss
                loss = criterion(output, label)
                valid_loss.append(loss.data.cpu())

            print(valid_log_string % (datetime.now(), epoch, np.mean(valid_loss)))
            # Record
            valid_loss_record.append(np.mean(valid_loss))
            epoch += 1

            if np.mean(valid_loss) - np.mean(train_loss_sum) > 0.02:
                print("Early Stopping!")
                break

            # Keep track of best record
            if np.mean(valid_loss) < best_record:
                best_record = np.mean(valid_loss)
                # save the best model
                state_dict = {
                    'epoch': epoch,
                    'siamese': siamese.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_dict, ckpt_path)
                print('Model saved!\n')

    """ Inference """

    if config['task'] == 'inference':
        testDS = OriginalMytestDS(test_data, all_sentences)
        # Do not shuffle here
        test_dataloader = DataLoader(dataset=testDS, num_workers=2, batch_size=1)

        result = []
        for idx, data in enumerate(test_dataloader, 0):
            # get data
            s1, s2 = data

            # input
            output = siamese(s1, s2)
            output = output.squeeze(0)

            # feed output into softmax to get prob prediction
            sm = nn.Softmax(dim=1)
            res = sm(output.data)[:, 1]
            result += res.data.tolist()

        result = pd.DataFrame(result)
        print(result.shape)
        print('Inference Done.')
        res_path = os.path.join(config['result']['filepath'], config['result']['filename'])
        result.to_csv(res_path, header=False, index=False)
        print('Result was written to', res_path, ', Good Luck!')


def create_dataset_for_siamese(path):
    # creating pair of reviews

    # loading dataset
    dataframe = pd.read_csv(path, header=None)

    # reducing size of dataframe for development purposes
    dataframe = dataframe[:500]

    # divide datasets into two with the same size
    perm = np.random.permutation(dataframe.index)
    dataframe_half_size = int(len(dataframe) * 0.5)

    dataframe_1 = dataframe.iloc[perm[:dataframe_half_size]].reset_index(drop=True)
    dataframe_2 = dataframe.iloc[perm[dataframe_half_size:]].reset_index(drop=True)

    dataframe_combined = pd.concat([dataframe_1, dataframe_2], axis=1)

    dataframe_combined.columns = ['evaluation_1', 'review_1', 'evaluation_2', 'review_2']

    # if evaluations are equal, then the reviews texts are similar
    dataframe_combined['similar'] = dataframe_combined.apply(
        lambda x: 1 if x['review_1'] == x['review_2'] else 0, axis=1
    )

    dataframe_combined = dataframe_combined.drop(['evaluation_1', 'evaluation_2'], axis=1)

    return dataframe_combined


def save_reduced_dataset():
    train_dataset = pd.read_csv('./input/yelp_polarized/train.csv', header=None)
    train_dataset = train_dataset[:500]
    train_dataset.to_csv('./input/yelp_polarized/train_reduced.csv', index=None, header=None)


def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    data = {
        'max_word_length': sorted_word_length[int(0.8 * len(sorted_word_length))],
        'max_sent_length': sorted_sent_length[int(0.8 * len(sorted_sent_length))]
    }

    with open('siamese_max_lengths.txt', 'w') as output_file:
        json.dump(data, output_file)


if __name__ == '__main__':
    # defining seed to get consistent results
    np.random.seed(123)

    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='siamese-config.yaml',
                        help='Configuration file.')
    FLAGS, _ = parser.parse_known_args()
    main(_)

