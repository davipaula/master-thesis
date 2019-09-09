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


class HANData:

    def __init__(self):

        logging.basicConfig(level=logging.INFO)

        # The dataset is taken from https://github.com/justmarkham/DAT7/blob/master/data/yelp.csv
        df = pd.read_csv('./input/yelp/yelp.csv')

        # limit number of data for local development
        df = df[:500]

        # mark the columns which contains text for classification and target class
        col_text = 'text'
        col_target = 'cool'

        self.cls_arr = np.sort(df[col_target].unique()).tolist()
        self.classes = len(self.cls_arr)

        # divide dataset in 80% train 10% validation 10% test as done in the paper
        length = df.shape[0]
        train_len = int(0.8 * length)
        val_len = int(0.1 * length)

        train = df[:train_len]
        val = df[train_len:train_len + val_len]
        test = df[train_len + val_len:]

        self.prepare_dataset(train, val, test, col_text)

        self.y_train = train[col_target].tolist()
        self.y_test = test[col_target].tolist()
        self.y_val = val[col_target].tolist()

    def prepare_dataset(self, train, val, test, col_text):
        # Fix the maximum length of sentences in a paragraph and words in a sentence
        max_sentences_per_paragraph = 12
        max_words_per_sentence = 25

        # divides review in sentences and sentences into word creating a 3DList
        x_train = self.create_3d_array(train, col_text, max_sentences_per_paragraph, max_words_per_sentence)
        x_val = self.create_3d_array(val, col_text, max_sentences_per_paragraph, max_words_per_sentence)
        x_test = self.create_3d_array(test, col_text, max_sentences_per_paragraph, max_words_per_sentence)

        x_test_texts = self.remove_stop_words(x_test)
        x_train_texts = self.remove_stop_words(x_train)
        x_val_texts = self.remove_stop_words(x_val)

        words_frequency = self.frequency_of_words(x_train_texts, x_test_texts, x_val_texts)

        # Just to compare words_frequency. To remove one of them
        texts = list(more_itertools.collapse(x_train_texts[:] + x_test_texts[:] + x_val_texts[:], levels=1))
        words_frequency_with_texts = self.frequency_of_words_with_texts(texts)

        x_test_texts = self.remove_non_frequent_words(words_frequency, x_test_texts)
        x_train_texts = self.remove_non_frequent_words(words_frequency, x_train_texts)
        x_val_texts = self.remove_non_frequent_words(words_frequency, x_val_texts)

        texts = list(more_itertools.collapse(x_train_texts[:] + x_test_texts[:] + x_val_texts[:], levels=1))

        word2vec = self.train_word2vec(texts)

        self.weights = torch.FloatTensor(word2vec.wv.syn0)  # .cuda()

        vocab = word2vec.wv.vocab

        self.vocab_size = len(vocab)

        self.x_test_vec = self.convert_to_vector(vocab, x_test_texts)
        self.x_train_vec = self.convert_to_vector(vocab, x_train_texts)
        self.x_val_vec = self.convert_to_vector(vocab, x_val_texts)

        print(f'vocab_size = {self.vocab_size}')

    def transform_texts_to_3d_array(self, col_text, test, train, val):
        # Fix the maximum length of sentences in a paragraph and words in a sentence
        max_sentence_per_paragraph_length = 12
        max_words_per_sentence_length = 25

        # divides review in sentences and sentences into word creating a 3DList
        x_train = self.create_3d_array(train, col_text, max_sentence_per_paragraph_length,
                                       max_words_per_sentence_length)
        x_val = self.create_3d_array(val, col_text, max_sentence_per_paragraph_length, max_words_per_sentence_length)
        x_test = self.create_3d_array(test, col_text, max_sentence_per_paragraph_length, max_words_per_sentence_length)

        return x_test, x_train, x_val

    def remove_stop_words(self, dataset):

        nltk.download('stopwords')

        stop_list = stopwords.words('english') + list(string.punctuation)
        stemmer = SnowballStemmer('english')

        dataset_without_stop_words = [
            [[stemmer.stem(word.lower()) for word in sentence if word not in stop_list] for sentence in paragraph]
            for paragraph in dataset]

        return dataset_without_stop_words

    def frequency_of_words(self, x_train_texts, x_test_texts, x_val_texts):
        # calculate frequency of words
        frequency = defaultdict(int)

        for texts in x_train_texts:
            for text in texts:
                for token in text:
                    frequency[token] += 1
        for texts in x_test_texts:
            for text in texts:
                for token in text:
                    frequency[token] += 1
        for texts in x_val_texts:
            for text in texts:
                for token in text:
                    frequency[token] += 1

        return frequency

    def remove_non_frequent_words(self, frequency, text_to_clean):
        # remove  words with frequency less than 5.
        clean_text = [[[token for token in text if frequency[token] > 5]
                       for text in texts] for texts in text_to_clean]

        return clean_text

    def train_word2vec(self, texts):

        # train word2vec model on all the words
        word2vec = Word2Vec(texts, size=200, min_count=5)
        word2vec.save("dictionary_yelp")

        return word2vec

    def convert_to_vector(self, vocab, dataset):

        # convert 3D text list to 3D list of index
        dataset_vector = [[[vocab[token].index for token in text]
                           for text in texts] for texts in dataset]

        return dataset_vector

    def clean_str(self, string, max_seq_len):
        """
        adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = BeautifulSoup(string, "lxml").text
        string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
        string = re.sub(r"\"s", " \"s", string)
        string = re.sub(r"\"ve", " \"ve", string)
        string = re.sub(r"n\"t", " n\"t", string)
        string = re.sub(r"\"re", " \"re", string)
        string = re.sub(r"\"d", " \"d", string)
        string = re.sub(r"\"ll", " \"ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        s = string.strip().lower().split(" ")
        if len(s) > max_seq_len:
            return s[0:max_seq_len]
        return s

    # creates a 3D list of format paragraph[sentence[word]]
    def create_3d_array(self, df, col, max_sent_len, max_seq_len):
        x = []
        for docs in df[col].as_matrix():
            x1 = []
            idx = 0
            for seq in "|||".join(re.split("[.?!]", docs)).split("|||"):
                x1.append(self.clean_str(seq, max_sent_len))
                if (idx >= max_seq_len - 1):
                    break
                idx = idx + 1
            x.append(x1)
        return x
