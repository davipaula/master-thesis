import csv

from nltk import word_tokenize, sent_tokenize
from torch.utils.data import Dataset
from collections import Counter
import pandas as pd
import numpy as np


class OriginalMyDS(Dataset):

    def __init__(self, df, all_sents):
        # Assign vocabularies.
        self.s1 = df['review_1'].tolist()
        self.s2 = df['review_2'].tolist()
        self.label = df['similar'].tolist()
        self.vocab = Vocab(all_sents, sos_token='<sos>', eos_token='<eos>', unk_token='<unk>')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Split sentence into words.
        s1_words = self.s1[idx].split()
        s2_words = self.s2[idx].split()

        # Add <SOS> and <EOS> tokens.
        s1_words = [self.vocab.sos_token] + s1_words + [self.vocab.eos_token]
        s2_words = [self.vocab.sos_token] + s2_words + [self.vocab.eos_token]

        # Lookup word ids in vocabularies.
        s1_ids = [self.vocab.word2id(word) for word in s1_words]
        s2_ids = [self.vocab.word2id(word) for word in s2_words]
        label = self.label[idx]

        return s1_ids, s2_ids, label


class OriginalMytestDS(Dataset):

    def __init__(self, df, all_sents):
        # Assign vocabularies.
        self.s1 = df['review_1'].tolist()
        self.s2 = df['review_2'].tolist()
        self.vocab = Vocab(all_sents, sos_token='<sos>', eos_token='<eos>', unk_token='<unk>')

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        # Split sentence into words.
        s1_words = self.s1[idx].split()
        s2_words = self.s2[idx].split()

        # Add <SOS> and <EOS> tokens.
        s1_words = [self.vocab.sos_token] + s1_words + [self.vocab.eos_token]
        s2_words = [self.vocab.sos_token] + s2_words + [self.vocab.eos_token]

        # Lookup word ids in vocabularies.
        s1_ids = [self.vocab.word2id(word) for word in s1_words]
        s2_ids = [self.vocab.word2id(word) for word in s2_words]

        return s1_ids, s2_ids


class Vocab(object):
    def __init__(self, all_sents, max_size=None, sos_token=None, eos_token=None, unk_token=None):
        """Initialize the vocabulary.
        Args:
            iter: An iterable which produces sequences of tokens used to update
                the vocabulary.
            max_size: (Optional) Maximum number of tokens in the vocabulary.
            sos_token: (Optional) Token denoting the start of a sequence.
            eos_token: (Optional) Token denoting the end of a sequence.
            unk_token: (Optional) Token denoting an unknown element in a
                sequence.
        """
        self.max_size = max_size
        self.pad_token = '<pad>'
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        # Add special tokens.
        id2word = [self.pad_token]
        if sos_token is not None:
            id2word.append(self.sos_token)
        if eos_token is not None:
            id2word.append(self.eos_token)
        if unk_token is not None:
            id2word.append(self.unk_token)

        # Update counter with token counts.
        counter = Counter()
        for x in all_sents:
            counter.update(x.split())

        # Extract lookup tables.
        if max_size is not None:
            counts = counter.most_common(max_size)
        else:
            counts = counter.items()
            counts = sorted(counts, key=lambda x: x[1], reverse=True)
        words = [x[0] for x in counts]
        id2word.extend(words)
        word2id = {x: i for i, x in enumerate(id2word)}

        self._id2word = id2word
        self._word2id = word2id

    def __len__(self):
        return len(self._id2word)

    def word2id(self, word):
        """Map a word in the vocabulary to its unique integer id.
        Args:
            word: Word to lookup.
        Returns:
            id: The integer id of the word being looked up.
        """
        if word in self._word2id:
            return self._word2id[word]
        elif self.unk_token is not None:
            return self._word2id[self.unk_token]
        else:
            raise KeyError('Word "%s" not in vocabulary.' % word)

    def id2word(self, id):
        """Map an integer id to its corresponding word in the vocabulary.
        Args:
            id: Integer id of the word being looked up.
        Returns:
            word: The corresponding word.
        """
        return self._id2word[id]

    def dummy_function(self):
        return False



# Please ignore the classes below. They were a tentative to combine the approach from HAN model (https://github.com/vietnguyen1991/Hierarchical-attention-networks-pytorch/)
class myDS(Dataset):

    def __init__(self, df, dict_path, max_length_sentences, max_length_word):
        # Assign vocabularies.
        self.s1 = df['review_1'].tolist()
        self.s2 = df['review_2'].tolist()
        self.label = df['similar'].tolist()

        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Split sentence into words.
        s1_words = self.s1[idx]
        s2_words = self.s2[idx]

        s1_encode = self.document_encode(s1_words)
        s2_encode = self.document_encode(s2_words)

        # # Add <SOS> and <EOS> tokens.
        # s1_words = [self.vocab.sos_token] + s1_words + [self.vocab.eos_token]
        # s2_words = [self.vocab.sos_token] + s2_words + [self.vocab.eos_token]
        #
        # # Lookup word ids in vocabularies.
        # s1_ids = [self.vocab.word2id(word) for word in s1_words]
        # s2_ids = [self.vocab.word2id(word) for word in s2_words]
        label = self.label[idx]

        return s1_encode.astype(np.int64), s2_encode.astype(np.int64), label

        # return s1_ids, s2_ids, label

    def document_encode(self, texts):
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in sent_tokenize(text=texts)]

        for sentences in sent_tokenize(text=texts):
            print(sentences)

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)

        document_encode += 1

        return document_encode

class myDS2(Dataset):

    def __init__(self, df, dict_path, max_length_sentences, max_length_word):
        # Assign vocabularies.
        self.s1 = df['review_1'].tolist()
        self.s2 = df['review_2'].tolist()
        self.label = df['similar'].tolist()

        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Split sentence into words.
        s1_words = self.s1[idx]
        s2_words = self.s2[idx]

        s1_encode = self.document_encode(s1_words)
        s2_encode = self.document_encode(s2_words)

        # # Add <SOS> and <EOS> tokens.
        # s1_words = [self.vocab.sos_token] + s1_words + [self.vocab.eos_token]
        # s2_words = [self.vocab.sos_token] + s2_words + [self.vocab.eos_token]
        #
        # # Lookup word ids in vocabularies.
        # s1_ids = [self.vocab.word2id(word) for word in s1_words]
        # s2_ids = [self.vocab.word2id(word) for word in s2_words]

        return s1_encode.astype(np.int64), s2_encode.astype(np.int64)

        # return s1_ids, s2_ids, label

    def document_encode(self, texts):
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in sent_tokenize(text=texts)]

        for sentences in sent_tokenize(text=texts):
            print(sentences)

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)

        document_encode += 1

        return document_encode


class mytestDS(Dataset):

    def __init__(self, df):
        # Assign vocabularies.
        self.s1 = df['review_1'].tolist()
        self.s2 = df['review_2'].tolist()

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        # Split sentence into words.
        s1_words = self.s1[idx].split()
        s2_words = self.s2[idx].split()

        # Add <SOS> and <EOS> tokens.
        s1_words = [self.vocab.sos_token] + s1_words + [self.vocab.eos_token]
        s2_words = [self.vocab.sos_token] + s2_words + [self.vocab.eos_token]

        # Lookup word ids in vocabularies.
        s1_ids = [self.vocab.word2id(word) for word in s1_words]
        s2_ids = [self.vocab.word2id(word) for word in s2_words]

        return s1_ids, s2_ids



if __name__ == '__main__':
    df = pd.read_csv('../input/yelp_polarized/train_to_debug.csv', header=None,
                     names=['review_1', 'review_2', 'similar'])

    test = myDS(df=df,
                dict_path="../glove.6B.50d.txt",
                max_length_sentences=50,
                max_length_word=50)

    test[35]
    # print(test.__getitem__(index=1)[0].shape)
