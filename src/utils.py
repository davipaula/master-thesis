"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import torch
import numpy as np
import pandas as pd
import logging
from sklearn import metrics
from ast import literal_eval
import itertools as it


# csv.field_size_limit(sys.maxsize)
def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.matmul(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()


def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(len(feature_2.shape)).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def get_max_lengths(data_path, limit_rows=None):
    word_length_list = []
    sent_length_list = []
    paragraph_length_list = []

    dataset = pd.read_csv(data_path, nrows=limit_rows)

    document_encode = dataset['current_article_text']
    document_encode = pd.concat([document_encode, dataset['previous_article_text']])

    for index, article in enumerate(document_encode):
        for paragraph in literal_eval(article):
            for sentences in paragraph:
                word_length_list.append(len(sentences))

            sent_length_list.append(len(paragraph))

        paragraph_length_list.append(len(literal_eval(article)))

    return max(word_length_list), max(sent_length_list), max(paragraph_length_list)


def get_padded_document(document, max_length_word, max_length_sentences, max_length_paragraph):
    for paragraph in document:
        for sentences in paragraph:
            if len(sentences) < max_length_word:
                extended_words = [-1 for _ in range(max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(paragraph) < max_length_sentences:
            extended_sentences = [[-1 for _ in range(max_length_word)] for _ in
                                  range(max_length_sentences - len(paragraph))]
            paragraph.extend(extended_sentences)

    if len(document) < max_length_paragraph:
        extended_paragraphs = [[[-1 for _ in range(max_length_word)]
                                for _ in range(max_length_sentences)]
                               for _ in range(max_length_paragraph - len(document))]

        document.extend(extended_paragraphs)

    document = [sentences[:max_length_word] for sentences in document][
               :max_length_sentences]

    document = np.stack(arrays=document, axis=0)
    document += 1

    return document


def get_words_per_document_at_word_level(words_per_sentence):
    return words_per_sentence.sum(dim=2).sum(dim=1).unsqueeze(1).unsqueeze(1)


def get_document_at_word_level(document_batch, words_per_sentence_at_word_level):
    batch_size = document_batch.shape[0]
    paragraph_length = 1
    sentence_length = 1
    word_length = max(words_per_sentence_at_word_level).item()

    word_level_document_placeholder = torch.zeros((batch_size, paragraph_length, sentence_length, word_length),
                                                  dtype=int)

    for document_idx, document in enumerate(document_batch):
        non_zero_indices = document.nonzero(as_tuple=True)

        word_level_document_placeholder[document_idx, 0, 0, : len(document[non_zero_indices])] = document[non_zero_indices]

    return word_level_document_placeholder


def remove_zero_tensors_from_batch(sentences_in_batch):
    non_zero_indices = torch.nonzero(sentences_in_batch, as_tuple=True)[0].unique().tolist()

    max_sentence_length = sentences_in_batch.shape[1]
    non_zero_tensor = torch.zeros((len(non_zero_indices), max_sentence_length), dtype=sentences_in_batch.dtype)

    for i, non_zero_index in enumerate(non_zero_indices):
        non_zero_tensor[i] = sentences_in_batch[non_zero_index]

    return non_zero_tensor


def add_filtered_tensors_to_original_batch(filtered_batch, original_batch):
    non_zero_indices = torch.nonzero(original_batch, as_tuple=True)[0].unique().tolist()

    batch_size = original_batch.shape[0]
    sequence_length = filtered_batch.shape[1]
    tensor_size = [batch_size, sequence_length]

    if len(filtered_batch.shape) > 2:
        tensor_size.append(filtered_batch.shape[2])

    original_batch_reshaped = torch.zeros(tuple(tensor_size),
                                          dtype=filtered_batch.dtype)

    for i, non_zero_tensor_index in enumerate(non_zero_indices):
        original_batch_reshaped[non_zero_tensor_index] = filtered_batch[i]

    return original_batch_reshaped


def remove_zeros_from_words_per_sentence(words_per_sentence):
    non_zero_indices = words_per_sentence.nonzero().squeeze(1)

    return words_per_sentence[non_zero_indices].tolist()


def get_document_at_sentence_level(document):
    batch_size = document.shape[0]
    paragraph_length = document.shape[1]
    sentence_length = document.shape[2]
    word_length = document.shape[3]

    document_at_sentence_level_tensor = torch.zeros(
        (batch_size, 1, paragraph_length * sentence_length, word_length),
        dtype=int)

    for document_index, document_in_batch in enumerate(document):
        document_at_sentence_level = []
        for paragraph in document_in_batch:
            for sentence in paragraph:
                if sum(sentence).item() > 0:
                    document_at_sentence_level.append(sentence.tolist())

        document_at_sentence_level_tensor[document_index, 0, : len(document_at_sentence_level), :] = torch.LongTensor(
            document_at_sentence_level)

    return document_at_sentence_level_tensor


def get_words_per_sentence_at_sentence_level(words_per_sentence):
    batch_size = words_per_sentence.shape[0]
    paragraph_length = words_per_sentence.shape[1]
    sentence_length = words_per_sentence.shape[2]

    words_per_sentence_at_sentence_level_tensor = torch.zeros((batch_size, 1, paragraph_length * sentence_length),
                                                              dtype=int)

    for paragraph_index, paragraph in enumerate(words_per_sentence):
        words_per_sentence_at_sentence_level = []
        for sentence in paragraph:
            for words in sentence:
                if words > 0:
                    words_per_sentence_at_sentence_level.append(words)

        words_per_sentence_at_sentence_level_tensor[paragraph_index, 0,
        : len(words_per_sentence_at_sentence_level)] = torch.LongTensor(words_per_sentence_at_sentence_level)

    return torch.LongTensor(words_per_sentence_at_sentence_level_tensor)


def get_sentences_per_paragraph_at_sentence_level(sentences_per_paragraph):
    return sentences_per_paragraph.sum(dim=1).unsqueeze(1)


if __name__ == "__main__":
    torch.manual_seed(123)
    # word, sent, paragraph = get_max_lengths('../data/wiki_df.csv')
    word = 98
    sent = 60
    paragraph = 36

    training_generator = torch.load('../data/training.pth')

    i = 0
    for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in training_generator:
        words_per_sentence_current_document = get_words_per_document_at_word_level(
            words_per_sentence_current_document)

        current_document = get_document_at_word_level(current_document, words_per_sentence_current_document)

        break

    print(words_per_sentence_current_document.shape)
    print(current_document.shape)
