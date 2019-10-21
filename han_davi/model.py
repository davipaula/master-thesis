import os
import re
import random
import pickle
import unicodedata
import string
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from bs4 import BeautifulSoup
import itertools
import more_itertools
import numpy as np
import logging
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from collections import defaultdict
import time
import math
from han_davi.utils import Utils


class WordRNN(nn.Module):
    # The word RNN model for generating a sentence vector
    def __init__(self, vocab_size, embedding_size, batch_size, hidden_size):
        super(WordRNN, self).__init__()

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Word Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_rnn = nn.GRU(embedding_size, hidden_size, bidirectional=True)

        # Word Attention
        self.word_attention = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.word_attention_combined = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inputs, hidden):
        embedding_output = self.embedding(inputs)

        output, hidden = self.word_rnn(embedding_output, hidden)

        word_annotation = self.word_attention(output)
        attention = F.softmax(self.word_attention_combined(word_annotation), dim=1)

        sentence = Utils.attention_mul(output, attention)

        return sentence, hidden


class SentenceRNN(nn.Module):
    # The HAN model
    def __init__(self, vocabulary_size, embedding_size, batch_size, hidden_size, c, max_sequence_of_sentences_size):
        super(SentenceRNN, self).__init__()

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.cls = c
        self.max_sequence_of_sentences_size = max_sequence_of_sentences_size

        self.word_rnn = WordRNN(vocabulary_size, embedding_size, batch_size, hidden_size)

        # Sentence Encoder
        self.sentence_rnn = nn.GRU(embedding_size, hidden_size, bidirectional=True)

        # Sentence Attention
        self.sentence_attention = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attention_combined = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.document_linear_layer = nn.Linear(2 * hidden_size, c)

    def forward(self, inp, hidden_state_sentence, hidden_state_word):
        s = None
        # Generating sentence vector through WordRNN
        for i in range(len(inp[0])):
            r = None
            for j in range(len(inp)):
                if (r is None):
                    r = [inp[j][i]]
                else:
                    r.append(inp[j][i])
            r1 = np.asarray([sub_list + [0] * (self.max_sequence_of_sentences_size - len(sub_list)) for sub_list in r])
            _s, state_word = self.word_rnn(torch.LongTensor(r1).view(-1, self.batch_size), hidden_state_word)

            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)

                output_state, hidden_state = self.sentence_rnn(s, hidden_state_sentence)

        sentence_annotation = self.sentence_attention(output_state)
        attention = F.softmax(self.attention_combined(sentence_annotation), dim=1)

        document = Utils.attention_mul(output_state, attention)
        d = self.document_linear_layer(document)
        cls = F.log_softmax(d.view(-1, self.cls), dim=1)

        return cls, hidden_state

    def init_hidden_sent(self):
        return Variable(torch.zeros(2, self.batch_size, self.hidden_size))  # .cuda()

    def init_hidden_word(self):
        return Variable(torch.zeros(2, self.batch_size, self.hidden_size))  # .cuda()


class HANModel:
    def __init__(self, han_data):
        x_train_vec = han_data.x_train_vec
        x_test_vec = han_data.x_test_vec
        x_val_vec = han_data.x_val_vec
        y_train = han_data.y_train
        y_test = han_data.y_test
        y_val = han_data.y_val
        vocab_size = han_data.vocab_size
        weights = han_data.weights
        cls_arr = han_data.cls_arr
        classes = han_data.classes

        # converting list to tensor
        self.y_train_tensor = [torch.FloatTensor([cls_arr.index(label)]) for label in y_train]
        self.y_val_tensor = [torch.FloatTensor([cls_arr.index(label)]) for label in y_val]
        self.y_test_tensor = [torch.FloatTensor([cls_arr.index(label)]) for label in y_test]

        self.max_sequence_of_sentences_size = max([len(sequence_of_sentences) for sequence_of_sentences in
                                                   itertools.chain.from_iterable(x_train_vec + x_val_vec + x_test_vec)])
        self.max_sentence_size = max([len(sentence) for sentence in (x_train_vec + x_val_vec + x_test_vec)])

        # Padding the input
        self.X_train_pad = [sub_list + [[0]] * (self.max_sentence_size - len(sub_list)) for sub_list in x_train_vec]
        self.X_val_pad = [sub_list + [[0]] * (self.max_sentence_size - len(sub_list)) for sub_list in x_val_vec]
        self.X_test_pad = [sub_list + [[0]] * (self.max_sentence_size - len(sub_list)) for sub_list in x_test_vec]

        self.batch_size = 8  # 64
        hidden_size = 100
        embedding_size = 200

        self.sentence_attention = SentenceRNN(vocab_size, embedding_size, self.batch_size, hidden_size, classes,
                                              self.max_sequence_of_sentences_size)
        # sentence_attention.cuda()

        self.sentence_attention.word_rnn.embedding.from_pretrained(weights)

        torch.backends.cudnn.benchmark = True

        learning_rate = 1e-3
        momentum = 0.9
        self.sentence_optimizer = torch.optim.SGD(self.sentence_attention.parameters(), lr=learning_rate,
                                                  momentum=momentum)

        self.criterion = nn.NLLLoss()
