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


class Utils:

    @staticmethod
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    @staticmethod
    def gen_batch(x, y, batch_size):
        k = random.sample(range(len(x) - 1), batch_size)
        x_batch = []
        y_batch = []

        for t in k:
            x_batch.append(x[t])
            y_batch.append(y[t])

        return [x_batch, y_batch]

    @staticmethod
    def validation_accuracy(batch_size, x_val, y_val, sent_attn_model):
        acc = []
        val_length = len(x_val)

        for j in range(int(val_length / batch_size)):
            x, y = Utils.gen_batch(x_val, y_val, batch_size)
            state_word = sent_attn_model.init_hidden_word()
            state_sent = sent_attn_model.init_hidden_sent()

            y_pred, state_sent = sent_attn_model(x, state_sent, state_word)
            max_index = y_pred.max(dim=1)[1]
            correct = (max_index == torch.LongTensor(y)).sum()
            acc.append(float(correct) / batch_size)

        return np.mean(acc)

    @staticmethod
    def attention_mul(rnn_outputs, attention_weights):
        # Make the the multiple attention with word vectors.
        attention_vectors = None

        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = attention_weights[i]
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)

            if (attention_vectors is None):
                attention_vectors = h_i
            else:
                attention_vectors = torch.cat((attention_vectors, h_i), 0)

        return torch.sum(attention_vectors, 0).unsqueeze(0)
