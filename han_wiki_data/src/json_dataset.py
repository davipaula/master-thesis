"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import json


class JSONDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=6, max_length_word=18, max_length_paragraph=10):
        super(JSONDataset, self).__init__()

        texts, labels, articles = [], [], []

        with open(data_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)

            texts.append(result['sections'][0]['paragraphs'])
            articles.append(result['title'])
            labels.append(0)  # TODO figure out how to deal with labels (if needed)

        self.texts = texts
        self.labels = labels
        self.articles = articles
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))
        self.max_length_paragraph = max_length_paragraph

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        article = self.articles[index]

        document_encode = text

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

        return document_encode.astype(np.int64), label

    def get_max_lengths(self, document_encode):

        word_length_list = []
        sent_length_list = []

        for index, article in enumerate(document_encode):
            for paragraph in article:
                for sentences in paragraph:
                    word_length_list.append(len(sentences))

                sent_length_list.append(len(paragraph))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

        print(sorted_word_length)
        print(sorted_sent_length)


if __name__ == '__main__':
    test = JSONDataset(data_path="../data/simplewiki_small.jsonl", dict_path="../data/glove.6B.50d.txt")
    print(test.__getitem__(index=203)[0])
