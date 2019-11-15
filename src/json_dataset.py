"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
import numpy as np
import json
import utils
from ast import literal_eval


class SMASHDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=6, max_length_word=18, max_length_paragraph=10):
        super(SMASHDataset, self).__init__()

        dataset = pd.read_csv(data_path)

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
        self.num_classes = 1  # TODO deprecate this attribute
        self.max_length_paragraph = max_length_paragraph

    def __len__(self):
        return len(self.current_article_text)

    def __getitem__(self, index):
        current_article_text = literal_eval(self.current_article_text.iloc[index])
        current_article_title = self.current_article_title.iloc[index]
        previous_article_text = literal_eval(self.previous_article_text.iloc[index])
        previous_article_title = self.previous_article_title.iloc[index]

        current_article_text_padded = self.get_padded_document(current_article_text).astype(np.int64)
        previous_article_text_padded = self.get_padded_document(previous_article_text).astype(np.int64)

        click_rate = self.click_rate.iloc[index]

        return current_article_text_padded, current_article_title, previous_article_text_padded, previous_article_title, click_rate

    def get_padded_document(self, document_encode):
        for paragraph in document_encode:
            for sentences in paragraph:
                if len(sentences) < self.max_length_word:
                    extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                    sentences.extend(extended_words)

            if len(paragraph) < self.max_length_sentences:
                extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                      range(self.max_length_sentences - len(paragraph))]
                paragraph.extend(extended_sentences)

        if len(document_encode) < self.max_length_paragraph:
            extended_paragraphs = [[[-1 for _ in range(self.max_length_word)]
                                    for _ in range(self.max_length_sentences)]
                                   for _ in range(self.max_length_paragraph - len(document_encode))]

            document_encode.extend(extended_paragraphs)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode


if __name__ == '__main__':
    wiki_data_path = '../data/wiki_df_small.csv'
    max_word_length, max_sent_length, max_paragraph_length = utils.get_max_lengths(wiki_data_path)
    test = SMASHDataset(data_path=wiki_data_path, dict_path="../data/glove.6B.50d.txt", max_length_word=max_word_length,
                        max_length_sentences=max_sent_length, max_length_paragraph=max_paragraph_length)
    print(len(test))
    print(test.__getitem__(index=95))
