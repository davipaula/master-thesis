"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from datetime import datetime

import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
import numpy as np
import src.utils as utils
from ast import literal_eval
import torch


class SMASHDataset(Dataset):
    def __init__(self, data_path, dict_path, max_length_sentences=6, max_length_word=18, max_length_paragraph=10,
                 limit_rows=None):
        super(SMASHDataset, self).__init__()

        dataset = pd.read_csv(data_path, nrows=limit_rows)
        self.current_article_text = dataset['current_article_text']
        self.current_article_title = dataset['current_article']
        self.previous_article_text = dataset['previous_article_text']
        self.previous_article_title = dataset['previous_article']
        self.click_rate = dataset['click_rate']

        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.max_length_paragraph = max_length_paragraph

    def __len__(self):
        return len(self.current_article_text)

    def __getitem__(self, index):
        current_article_text = literal_eval(self.current_article_text.iloc[index])
        current_article_title = self.current_article_title.iloc[index]
        previous_article_text = literal_eval(self.previous_article_text.iloc[index])
        previous_article_title = self.previous_article_title.iloc[index]

        current_article_structure = self.get_document_structure(current_article_text)
        current_article_words_per_sentence = torch.LongTensor(
            self.get_padded_words_per_sentence(current_article_structure['words_per_sentence']))
        current_article_sentences_per_paragraph = torch.LongTensor(
            self.get_padded_sentences_per_paragraph(current_article_structure['sentences_per_paragraph']))
        current_article_paragraphs_per_document = torch.LongTensor(
            [current_article_structure['paragraphs_per_document']])

        previous_article_structure = self.get_document_structure(previous_article_text)
        previous_article_words_per_sentence = torch.LongTensor(
            self.get_padded_words_per_sentence(previous_article_structure['words_per_sentence']))
        previous_article_sentences_per_paragraph = torch.LongTensor(
            self.get_padded_sentences_per_paragraph(previous_article_structure['sentences_per_paragraph']))
        previous_article_paragraphs_per_document = torch.LongTensor([
            previous_article_structure['paragraphs_per_document']])

        current_article_text_padded = torch.LongTensor(self.get_padded_document(current_article_text).astype(np.int64))
        previous_article_text_padded = torch.LongTensor(
            self.get_padded_document(previous_article_text).astype(np.int64))

        click_rate = torch.FloatTensor([self.click_rate.iloc[index]])

        return current_article_text_padded, \
               current_article_words_per_sentence, \
               current_article_sentences_per_paragraph, \
               current_article_paragraphs_per_document, \
               previous_article_text_padded, \
               previous_article_words_per_sentence, \
               previous_article_sentences_per_paragraph, \
               previous_article_paragraphs_per_document, \
               click_rate

    def get_document_structure(self, document):
        paragraphs_per_document = len(document)
        sentences_per_paragraph = []
        words_per_sentence = []
        for paragraph in document:
            words_per_sentence_in_paragraph = []
            for sentence in paragraph:
                words_per_sentence_in_paragraph.append(len(sentence))
            words_per_sentence.append(words_per_sentence_in_paragraph)
            sentences_per_paragraph.append(len(paragraph))

        document_structure = {
            'paragraphs_per_document': paragraphs_per_document,
            'sentences_per_paragraph': sentences_per_paragraph,
            'words_per_sentence': words_per_sentence
        }

        return document_structure

    def get_document(self, index):
        return literal_eval(self.current_article_text.iloc[index])

    def get_padded_document(self, document):
        for paragraph in document:
            for sentences in paragraph:
                if len(sentences) < self.max_length_word:
                    extended_words = list([-1] * (self.max_length_word - len(sentences)))
                    sentences.extend(extended_words)

            if len(paragraph) < self.max_length_sentences:
                extended_sentences = list(
                    [[-1] * self.max_length_word] * (self.max_length_sentences - len(paragraph)))
                paragraph.extend(extended_sentences)

        if len(document) < self.max_length_paragraph:
            extended_paragraphs = list([[[-1] * self.max_length_word] * self.max_length_sentences] * (
                    self.max_length_paragraph - len(document)))
            document.extend(extended_paragraphs)

        document = np.stack(arrays=document, axis=0)
        document += 1

        return document

    def get_padded_words_per_sentence(self, document_structure):
        for paragraph in document_structure:
            if len(paragraph) < self.max_length_sentences:
                extended_sentences = [0 for _ in range(self.max_length_sentences - len(paragraph))]
                paragraph.extend(extended_sentences)

        if len(document_structure) < self.max_length_paragraph:
            extended_paragraphs = [list([0] * self.max_length_sentences)] * (
                    self.max_length_paragraph - len(document_structure))

            document_structure.extend(extended_paragraphs)

        document_structure = np.stack(arrays=document_structure, axis=0)

        return document_structure

    def get_padded_words_per_sentence_refactor(self, document_structure):
        for paragraph in document_structure:
            if len(paragraph) < self.max_length_sentences:
                extended_sentences = list([0] * (self.max_length_sentences - len(paragraph)))
                paragraph.extend(extended_sentences)

        if len(document_structure) < self.max_length_paragraph:
            extended_paragraphs = [[0 for _ in range(self.max_length_sentences)]
                                   for _ in range(self.max_length_paragraph - len(document_structure))]

            document_structure.extend(extended_paragraphs)

        document_structure = np.stack(arrays=document_structure, axis=0)

        return document_structure

    def get_padded_sentences_per_paragraph(self, document_structure):
        if len(document_structure) < self.max_length_paragraph:
            extended_paragraphs = [0 for _ in range(self.max_length_paragraph - len(document_structure))]
            document_structure.extend(extended_paragraphs)

        return document_structure


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    torch.set_printoptions(profile="full")

    wiki_data_path = '../data/wiki_df_small.csv'
    max_word_length, max_sent_length, max_paragraph_length = utils.get_max_lengths(wiki_data_path)
    test = SMASHDataset(data_path=wiki_data_path, dict_path="../data/glove.6B.50d.txt", max_length_word=max_word_length,
                        max_length_sentences=max_sent_length, max_length_paragraph=max_paragraph_length)

    print(test.__getitem__(1)[2])
