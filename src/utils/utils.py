"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import re
import timeit

import numpy as np
import pandas as pd
import torch


def get_padded_document(document, max_length_word, max_length_sentences, max_length_paragraph):
    for paragraph in document:
        for sentences in paragraph:
            if len(sentences) < max_length_word:
                extended_words = [-1 for _ in range(max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(paragraph) < max_length_sentences:
            extended_sentences = [
                [-1 for _ in range(max_length_word)] for _ in range(max_length_sentences - len(paragraph))
            ]
            paragraph.extend(extended_sentences)

    if len(document) < max_length_paragraph:
        extended_paragraphs = [
            [[-1 for _ in range(max_length_word)] for _ in range(max_length_sentences)]
            for _ in range(max_length_paragraph - len(document))
        ]

        document.extend(extended_paragraphs)

    document = [sentences[:max_length_word] for sentences in document][:max_length_sentences]

    document = np.stack(arrays=document, axis=0)
    document += 1

    return document


def remove_zero_tensors_from_batch(tensors_in_batch):
    non_zero_indices = torch.nonzero(tensors_in_batch, as_tuple=True)[0].unique().tolist()

    # If there are no empty tensors, there's no need to perform calculations and should return filtered_batch
    if len(non_zero_indices) == tensors_in_batch.shape[0]:
        return tensors_in_batch

    non_zero_tensor = torch.stack([tensors_in_batch[non_zero_index] for non_zero_index in non_zero_indices])

    if torch.cuda.is_available():
        non_zero_tensor = non_zero_tensor.cuda()

    return non_zero_tensor


def remove_batches_with_no_sentences(tensors_in_batch):
    non_zero_indices = torch.nonzero(tensors_in_batch, as_tuple=True)[0].unique().tolist()

    # If there are no empty tensors, there's no need to perform calculations and should return filtered_batch
    if len(non_zero_indices) == tensors_in_batch.shape[0]:
        return tensors_in_batch

    non_zero_tensor = torch.stack([tensors_in_batch[non_zero_index] for non_zero_index in non_zero_indices])

    if torch.cuda.is_available():
        non_zero_tensor = non_zero_tensor.cuda()

    return non_zero_tensor


def add_filtered_tensors_to_original_batch(filtered_batch, original_batch):
    non_zero_indices = set(torch.nonzero(original_batch, as_tuple=True)[0].tolist())

    # If there are no empty tensors, there's no need to perform calculations and should return filtered_batch
    if len(original_batch.shape) == 2 and len(non_zero_indices) == original_batch.shape[0]:
        return filtered_batch

    batch_size = original_batch.shape[0]
    sequence_length = filtered_batch.shape[1]
    tensor_size = [batch_size, sequence_length]

    if len(filtered_batch.shape) > 2:
        tensor_size.append(filtered_batch.shape[2])

    original_batch_reshaped = torch.zeros(tuple(tensor_size), dtype=filtered_batch.dtype)

    for i, non_zero_tensor_index in enumerate(non_zero_indices):
        try:
            original_batch_reshaped[non_zero_tensor_index] = filtered_batch[i]
        except:
            print("Check")

    if torch.cuda.is_available():
        original_batch_reshaped = original_batch_reshaped.cuda()

    return original_batch_reshaped


def add_filtered_tensors_to_original_sentences_batch(filtered_batch, original_batch):
    non_zero_indices = set(torch.nonzero(original_batch, as_tuple=True)[0].tolist())

    batch_size = original_batch.shape[0]
    sequence_length = filtered_batch.shape[1]
    tensor_size = [batch_size, sequence_length]

    if len(filtered_batch.shape) > 2:
        tensor_size.append(filtered_batch.shape[2])

    original_batch_reshaped = torch.zeros(tuple(tensor_size), dtype=filtered_batch.dtype)

    for i, non_zero_index in enumerate(non_zero_indices):
        original_batch_reshaped[non_zero_index] = filtered_batch[i]

    if torch.cuda.is_available():
        original_batch_reshaped = original_batch_reshaped.cuda()

    return original_batch_reshaped


def remove_zeros(elements_in_sequence):
    return [element if element else 1 for element in elements_in_sequence.tolist()]


def get_words_per_document_at_word_level(words_per_sentence):
    return words_per_sentence.sum(dim=2).sum(dim=1).unsqueeze(1).unsqueeze(1)


def get_words_per_sentence_at_sentence_level(words_per_sentence):
    batch_size = words_per_sentence.shape[0]
    paragraph_length = words_per_sentence.shape[1]
    sentence_length = words_per_sentence.shape[2]

    words_per_sentence_placeholder = torch.zeros((batch_size, 1, paragraph_length * sentence_length), dtype=int)

    flat_words_per_sentence = words_per_sentence.flatten(start_dim=1)

    for document_index, document in enumerate(flat_words_per_sentence):
        non_zero_indices = document.nonzero(as_tuple=True)
        non_zero_sentences = document[non_zero_indices]

        words_per_sentence_placeholder[document_index, 0, : len(non_zero_sentences)] = non_zero_sentences

    return words_per_sentence_placeholder


def get_sentences_per_paragraph_at_sentence_level(sentences_per_paragraph):
    return sentences_per_paragraph.sum(dim=1).unsqueeze(1)


def get_document_at_word_level(document_batch, words_per_sentence_at_word_level):
    batch_size = document_batch.shape[0]
    paragraph_length = 1
    sentence_length = 1
    word_length = max(words_per_sentence_at_word_level).item()

    word_level_document_placeholder = torch.zeros(
        (batch_size, paragraph_length, sentence_length, word_length), dtype=int
    )

    for document_index, document in enumerate(document_batch):
        non_zero_indices = document.nonzero(as_tuple=True)

        word_level_document_placeholder[document_index, 0, 0, : len(document[non_zero_indices])] = document[
            non_zero_indices
        ]

    return word_level_document_placeholder


def get_document_at_sentence_level(documents_in_batch):
    batch_size = documents_in_batch.shape[0]
    paragraph_length = documents_in_batch.shape[1]
    sentence_length = documents_in_batch.shape[2]
    word_length = documents_in_batch.shape[3]

    document_at_sentence_level_tensor = torch.zeros(
        (batch_size, 1, paragraph_length * sentence_length, word_length), dtype=int
    )

    for document_index, document_in_batch in enumerate(documents_in_batch):
        non_zero_sentences = document_in_batch[document_in_batch.sum(dim=2) > 0]

        document_at_sentence_level_tensor[document_index, 0, : len(non_zero_sentences), :] = non_zero_sentences

    return document_at_sentence_level_tensor


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return round(dot_product / (norm_a * norm_b), 8)


def measure_document():
    setup = """
from __main__ import get_document_at_sentence_level
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from ast import literal_eval

training_generator = torch.load('../data/training.pth')
current_document = next(iter(training_generator))[0]
        """

    print("starting")
    print(timeit.timeit("get_document_at_sentence_level(current_document['text'])", setup=setup, number=100))
    print("finished")


def measure_words_per_sentence():
    setup = """
from __main__ import get_words_per_sentence_at_sentence_level
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from ast import literal_eval

training_generator = torch.load('../data/training.pth')
current_document = next(iter(training_generator))[0]
        """

    print("starting")
    print(
        timeit.timeit(
            "get_words_per_sentence_at_sentence_level(current_document['words_per_sentence'])", setup=setup, number=100
        )
    )
    print("finished")


def test_sentence_level():
    training_generator = torch.load("../data/training.pth")

    for current_document, previous_document, click_rate_tensor in training_generator:
        current_document_ = get_document_at_sentence_level(current_document["text"])
        words_per_sentence = get_words_per_sentence_at_sentence_level(current_document["words_per_sentence"])

        break

    print(current_document_.shape)
    # print(words_per_sentence)


def remove_special_characters(text):
    try:
        text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", text)
        text = re.sub(r"\'s", " 's ", text)
        text = re.sub(r"\'ve", " 've ", text)
        text = re.sub(r"n\'t", " 't ", text)
        text = re.sub(r"\'re", " 're ", text)
        text = re.sub(r"\'d", " 'd ", text)
        text = re.sub(r"\'ll", " 'll ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ) ", text)
        text = re.sub(r"\?", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.lower()
    except:
        pass

    return text


def remove_special_characters_df(text_column: pd.Series):
    text_column = text_column.str.replace(r"_", " ")

    return text_column


def clean_title(title):
    if isinstance(title, str):
        title = title.replace(r"_", " ")

    return title


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.set_printoptions(profile="full")

    # test_sentence_level()

    measure_document()
