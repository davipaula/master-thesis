"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import csv
import json
import os
import re
import timeit
from itertools import chain
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from ..utils.constants import WORD2VEC_200D_PATH, WORD2VEC_50D_PATH


def remove_zeros(elements_in_sequence):
    return [element if element else 1 for element in elements_in_sequence.tolist()]


def get_words_per_document_at_word_level(words_per_sentence):
    return words_per_sentence.sum(dim=2).sum(dim=1).unsqueeze(1).unsqueeze(1)


def get_words_per_sentence_at_sentence_level(words_per_sentence, device):
    batch_size = words_per_sentence.shape[0]
    paragraph_length = words_per_sentence.shape[1]
    sentence_length = words_per_sentence.shape[2]

    words_per_sentence_placeholder = torch.zeros(
        (batch_size, 1, paragraph_length * sentence_length), dtype=int, device=device
    )

    flat_words_per_sentence = words_per_sentence.flatten(start_dim=1)

    for document_index, document in enumerate(flat_words_per_sentence):
        non_zero_indices = document.nonzero(as_tuple=True)
        non_zero_sentences = document[non_zero_indices]

        words_per_sentence_placeholder[
            document_index, 0, : len(non_zero_sentences)
        ] = non_zero_sentences

    return words_per_sentence_placeholder


def get_sentences_per_paragraph_at_sentence_level(sentences_per_paragraph):
    return sentences_per_paragraph.sum(dim=1).unsqueeze(1)


def get_document_at_word_level(
    document_batch, words_per_sentence_at_word_level, device
):
    batch_size = document_batch.shape[0]
    paragraph_length = 1
    sentence_length = 1
    word_length = max(words_per_sentence_at_word_level).item()

    word_level_document_placeholder = torch.zeros(
        (batch_size, paragraph_length, sentence_length, word_length),
        dtype=int,
        device=device,
    )

    for document_index, document in enumerate(document_batch):
        non_zero_indices = document.nonzero(as_tuple=True)

        word_level_document_placeholder[
            document_index, 0, 0, : len(document[non_zero_indices])
        ] = document[non_zero_indices]

    return word_level_document_placeholder


def get_document_at_sentence_level(documents_in_batch, device):
    batch_size = documents_in_batch.shape[0]
    paragraph_length = documents_in_batch.shape[1]
    sentence_length = documents_in_batch.shape[2]
    word_length = documents_in_batch.shape[3]

    document_at_sentence_level_tensor = torch.zeros(
        (batch_size, 1, paragraph_length * sentence_length, word_length),
        dtype=int,
        device=device,
    )

    for document_index, document_in_batch in enumerate(documents_in_batch):
        non_zero_sentences = document_in_batch[document_in_batch.sum(dim=2) > 0]

        document_at_sentence_level_tensor[
            document_index, 0, : len(non_zero_sentences), :
        ] = non_zero_sentences

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
    print(
        timeit.timeit(
            "get_document_at_sentence_level(current_document['text'])",
            setup=setup,
            number=100,
        )
    )
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
            "get_words_per_sentence_at_sentence_level(current_document['words_per_sentence'])",
            setup=setup,
            number=100,
        )
    )
    print("finished")


def remove_special_characters(text):
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

    return text


def remove_special_characters_df(text_column: pd.Series):
    text_column = text_column.str.replace(r"_", " ")

    return text_column


def clean_title(title):
    if isinstance(title, str):
        title = title.replace(r"_", " ")

    return title


def get_word2vec_path(w2v_dimension: int):
    word2vec_paths = {50: WORD2VEC_50D_PATH, 200: WORD2VEC_200D_PATH}

    w2v_dimension = 50 if w2v_dimension in word2vec_paths.keys() else w2v_dimension

    return word2vec_paths[w2v_dimension]


def get_model_name(level: str, model_name: str = None, introduction_only: bool = False):
    formatted_name: str = f"{level}_level"

    if model_name:
        formatted_name += f"_{model_name}"

    if introduction_only:
        formatted_name = "_introduction_only"

    return formatted_name


def flatten_article(article):
    flatten_sentences = list(chain.from_iterable(article))
    flatten_words = list(chain.from_iterable(flatten_sentences))

    return flatten_words


def calculate_tokenized_lengths_original(articles):
    tokenized_length = [
        len(flatten_article(json.loads(article))) for article in articles
    ]

    return pd.Series(tokenized_length)


def calculate_tokenized_lengths(article):
    tokenized_length = len(flatten_article(json.loads(article)))

    return tokenized_length


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.set_printoptions(profile="full")

    # test_sentence_level()

    measure_document()


def load_embeddings_from_file(embeddings_path: str) -> Tuple[torch.Tensor, int, int]:
    """Loads the embeddings from a txt file in word2vec format

    Parameters
    ----------
    embeddings_path : str
        Path to load the embeddings file

    Returns
    -------
    Tuple[torch.Tensor, int, int]

    """
    embeddings = pd.read_csv(
        filepath_or_buffer=embeddings_path,
        header=None,
        skiprows=1,
        sep="\s",
        engine="python",
        quoting=csv.QUOTE_NONE,
    ).values[:, 1:]

    vocab_size, embeddings_dimension_size = embeddings.shape
    vocab_size += 1

    # Adds a tensor filled with zeros that will be used to replace unknown words
    # This should not impact this project, since the unknown words were filtered in the tokenization step
    # TODO refactor this function and check if we can remove this line
    unknown_word = np.zeros((1, embeddings_dimension_size))
    embeddings = torch.from_numpy(
        np.concatenate([unknown_word, embeddings], axis=0).astype(np.float)
    )
    return embeddings, vocab_size, embeddings_dimension_size


def get_model_path(model_folder: str, model_name: str):
    model_file = model_name + "_model.pt"

    model_path = os.path.join(model_folder, model_file)

    return model_path
