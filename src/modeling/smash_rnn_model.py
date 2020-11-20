"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from ..utils.constants import (
    PARAGRAPHS_PER_DOCUMENT_COLUMN,
    SENTENCES_PER_PARAGRAPH_COLUMN,
    TEXT_IDS_COLUMN,
    WORDS_PER_SENTENCE_COLUMN,
)
from ..utils.utils import (
    get_document_at_sentence_level,
    get_document_at_word_level,
    get_sentences_per_paragraph_at_sentence_level,
    get_words_per_document_at_word_level,
    get_words_per_sentence_at_sentence_level,
    remove_zeros,
)

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

GRU_HIDDEN_SIZE = 128
HIDDEN_LAYER_SIZE = 128

LEVELS = {
    "paragraph": ["paragraph"],
    "sentence": ["paragraph", "sentence"],
    "word": ["paragraph", "sentence", "word"],
}


class SmashRNNModel(nn.Module):
    """
    Class that represents the Smash RNN model
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        vocab_size: int,
        embedding_dimension_size: int,
        levels: str,
    ):
        """

        Parameters
        ----------
        embeddings : torch.Tensor
            Dictionary containing words and embedding
        vocab_size : int
            Number of words in the vocabulary
        embedding_dimension_size : int
            Number of dimensions of embedding layer
        levels : str
        """
        super(SmashRNNModel, self).__init__()

        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(123)
            self.device = torch.device("cpu")

        self.dict = embeddings

        self.levels = LEVELS[levels]

        # Init embedding layer
        self.embedding = (
            nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dimension_size
            )
            .from_pretrained(embeddings)
            .to(self.device)
        )

        # RNN + attention layers
        self.word_gru = nn.GRU(
            embedding_dimension_size,
            GRU_HIDDEN_SIZE,
            bidirectional=True,
            batch_first=True,
        ).to(self.device)

        self.word_gru_out_size = GRU_HIDDEN_SIZE * 2

        self.word_attention = nn.Linear(self.word_gru_out_size, GRU_HIDDEN_SIZE).to(
            self.device
        )
        self.word_context_layer = nn.Linear(GRU_HIDDEN_SIZE, 1, bias=False).to(
            self.device
        )

        sentence_gru_hidden_size = GRU_HIDDEN_SIZE
        self.sentence_gru_out_size = sentence_gru_hidden_size * 2
        self.sentence_gru = nn.GRU(
            self.word_gru_out_size,
            sentence_gru_hidden_size,
            bidirectional=True,
            batch_first=True,
        ).to(self.device)
        self.sentence_attention = nn.Linear(
            self.sentence_gru_out_size, sentence_gru_hidden_size
        ).to(self.device)
        self.sentence_context_layer = nn.Linear(
            sentence_gru_hidden_size, 1, bias=False
        ).to(self.device)

        paragraph_gru_hidden_size = GRU_HIDDEN_SIZE
        self.paragraph_gru_out_size = paragraph_gru_hidden_size * 2
        self.paragraph_gru = nn.GRU(
            self.sentence_gru_out_size,
            paragraph_gru_hidden_size,
            bidirectional=True,
            batch_first=True,
        ).to(self.device)
        self.paragraph_attention = nn.Linear(
            self.paragraph_gru_out_size, GRU_HIDDEN_SIZE
        ).to(self.device)
        self.paragraph_context_layer = nn.Linear(GRU_HIDDEN_SIZE, 1, bias=False).to(
            self.device
        )

        # Estimator config
        self.estimator_input_dim = (
            2 * GRU_HIDDEN_SIZE * 4 * len(self.levels)
        )  # 4 = number of concatenations

        self.estimator_output_dim = 1

        self.estimator = nn.Sequential(
            nn.Linear(self.estimator_input_dim, self.estimator_input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.estimator_input_dim // 2, self.estimator_output_dim),
        ).to(self.device)

    def forward(
        self,
        target_article: Dict[str, List[str]],
        source_article: Dict[str, List[str]],
        paragraphs_limit: int = None,
    ) -> torch.Tensor:
        """Executes a forward pass for Smash RNN

        Parameters
        ----------
        target_article : Dict[str, List[str]]
            Batch of target articles
        source_article : Dict[str, List[str]]
            Batch of source articles
        paragraphs_limit : int

        Returns
        -------
        torch.Tensor

        """
        target_article_representation = self.get_document_representation(
            target_article, levels=self.levels, paragraphs_limit=paragraphs_limit
        )

        del target_article

        source_article_representation = self.get_document_representation(
            source_article, levels=self.levels, paragraphs_limit=paragraphs_limit
        )

        del source_article

        # Concatenates article representations. This is the siamese part of the model
        concatenated_articles_representation = torch.cat(
            (
                source_article_representation,
                target_article_representation,
                torch.abs(
                    source_article_representation - target_article_representation
                ),
                source_article_representation * target_article_representation,
            ),
            1,
        )

        del source_article_representation
        del target_article_representation

        articles_similarity = self.estimator(concatenated_articles_representation)

        return articles_similarity

    def get_document_representation(
        self,
        articles_batch: Dict[str, List[str]],
        levels: List[str] = None,
        paragraphs_limit: int = None,
    ) -> torch.Tensor:
        """Obtains the representation of an article

        Parameters
        ----------
        articles_batch : Dict[str, List[str]]
            Batch of articles
        levels : List[str]
            List with the level encoders
        paragraphs_limit : int
            Maximum number of paragraphs that will be processed. Ignored if `None`.

        Returns
        -------
        torch.Tensor

        """
        document_representation_list = []
        for level in levels:
            (
                articles,
                paragraphs_per_document,
                sentences_per_paragraph,
                words_per_sentence,
            ) = self.get_tensors_at_level(articles_batch, level)

            if paragraphs_limit and articles.shape[1] > paragraphs_limit:
                paragraphs_per_document = paragraphs_per_document.clamp(
                    1, paragraphs_limit
                )
                sentences_per_paragraph = sentences_per_paragraph[:, :paragraphs_limit]
                max_sentences_per_paragraph = sentences_per_paragraph.max()

                words_per_sentence = words_per_sentence[
                    :, :paragraphs_limit, :max_sentences_per_paragraph
                ]
                max_words_per_sentence = words_per_sentence.max()

                articles = articles[
                    :,
                    :paragraphs_limit,
                    :max_sentences_per_paragraph,
                    :max_words_per_sentence,
                ]

            batch_size = articles.shape[0]
            max_paragraphs_per_article = articles.shape[1]
            max_sentences_per_paragraph = articles.shape[2]
            max_words_per_sentence = words_per_sentence.max()

            flatten_size = (
                batch_size * max_paragraphs_per_article * max_sentences_per_paragraph
            )
            flatten_word_ids = articles.reshape((flatten_size, max_words_per_sentence))

            flatten_words_per_sentence = words_per_sentence.reshape(flatten_size)

            # deleting tensors to release memory
            del articles
            del words_per_sentence

            # Limit of number of elements per tensor
            TENSOR_SIZE_LIMIT = 500000

            split_factor = max(
                1, int(np.prod(flatten_word_ids.shape) / TENSOR_SIZE_LIMIT)
            )
            split_size = int(len(flatten_word_ids) / split_factor)
            flatten_word_ids = flatten_word_ids.split(split_size)
            flatten_words_per_sentence = flatten_words_per_sentence.split(split_size)

            word_level_representation_list = []
            for index, _ in enumerate(flatten_word_ids):
                word_level_representation = self.get_word_level_representation(
                    batch_size,
                    flatten_word_ids[index],
                    flatten_words_per_sentence[index],
                    max_paragraphs_per_article,
                    max_sentences_per_paragraph,
                    max_words_per_sentence,
                )

                word_level_representation_list.append(word_level_representation)

            word_level_representation = torch.cat(word_level_representation_list)

            word_level_representation = word_level_representation.reshape(
                (
                    batch_size * max_paragraphs_per_article,
                    max_sentences_per_paragraph,
                    word_level_representation.shape[-1],
                )
            )

            del flatten_word_ids
            del flatten_words_per_sentence

            # attention over sentences
            sentence_level_representation = self.get_sentence_level_representation(
                batch_size,
                word_level_representation,
                max_paragraphs_per_article,
                sentences_per_paragraph,
            )

            # attention over paragraphs
            paragraph_level_representation = self.get_paragraph_level_representation(
                paragraphs_per_document, sentence_level_representation
            )

            document_representation_list.append(paragraph_level_representation)

        document_representation = torch.cat(document_representation_list, dim=1)

        return document_representation

    def get_tensors_at_level(
        self, articles_batch: Dict[str, List[str]], level: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts a tensor to the specified level

        Parameters
        ----------
        articles_batch : Dict[str, List[str]]
            Batch of articles
        level : str
            Level to convert the articles

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        if level == "sentence":
            articles = self.transform_to_sentence_level(articles_batch)

        elif level == "word":
            articles = self.transform_to_word_level(articles_batch)

        else:
            articles = articles_batch

        articles_text = articles[TEXT_IDS_COLUMN].to(self.device, non_blocking=True)
        words_per_sentence = articles[WORDS_PER_SENTENCE_COLUMN].to(
            self.device, non_blocking=True
        )
        sentences_per_paragraph = articles[SENTENCES_PER_PARAGRAPH_COLUMN].to(
            self.device, non_blocking=True
        )
        paragraphs_per_document = articles[PARAGRAPHS_PER_DOCUMENT_COLUMN].to(
            self.device, non_blocking=True
        )
        return (
            articles_text,
            paragraphs_per_document,
            sentences_per_paragraph,
            words_per_sentence,
        )

    def get_paragraph_level_representation(
        self,
        paragraphs_per_article: torch.Tensor,
        sentence_level_representation: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the representation of a sequence of paragraphs

        Parameters
        ----------
        paragraphs_per_article : torch.Tensor
            Tensor with the number of paragraphs per article in the batch
        sentence_level_representation : torch.Tensor
            Tensor with the sentences representation of the paragraph

        Returns
        -------
        torch.Tensor

        """
        packed_paragraphs = pack_padded_sequence(
            sentence_level_representation,
            lengths=paragraphs_per_article,
            batch_first=True,
            enforce_sorted=False,
        )

        paragraph_gru_output, _ = self.paragraph_gru(packed_paragraphs)

        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        paragraph_state = torch.tanh(
            self.paragraph_attention(paragraph_gru_output.data)
        )

        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        paragraph_context_vector = self.paragraph_context_layer(
            paragraph_state
        ).squeeze(1)

        exponent = torch.exp(paragraph_context_vector)

        del paragraph_state
        del paragraph_context_vector

        paragraph_attention, _ = pad_packed_sequence(
            PackedSequence(
                data=exponent,
                batch_sizes=paragraph_gru_output.batch_sizes,
                sorted_indices=paragraph_gru_output.sorted_indices,
                unsorted_indices=paragraph_gru_output.unsorted_indices,
            ),
            batch_first=True,
        )  # (n_sentences, max(words_per_sentence))

        paragraph_alphas = paragraph_attention / torch.sum(
            paragraph_attention, dim=1, keepdim=True
        )

        paragraph_gru_output, _ = pad_packed_sequence(
            paragraph_gru_output, batch_first=True
        )

        document_representation = (
            paragraph_gru_output * paragraph_alphas.unsqueeze(2)
        ).sum(dim=1)

        return document_representation

    def get_sentence_level_representation(
        self,
        batch_size: int,
        flatten_sentences: torch.Tensor,
        max_paragraphs_per_article: int,
        sentences_per_paragraph: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the representation of a sequence of sentences

        Parameters
        ----------
        batch_size : int
            Size of the batch
        flatten_sentences : torch.Tensor
            Tensor with the words representation of the sentences
        max_paragraphs_per_article : int
            Maximum number of paragraphs per article in the batch
        sentences_per_paragraph : torch.Tensor
            Tensor with the number of sentences per paragraph in the batch

        Returns
        -------
        torch.Tensor

        """
        sentences_per_paragraph = sentences_per_paragraph.reshape(
            batch_size * max_paragraphs_per_article
        )
        non_empty_sentences_per_paragraph = remove_zeros(sentences_per_paragraph)

        packed_sentences = pack_padded_sequence(
            flatten_sentences,
            lengths=non_empty_sentences_per_paragraph,
            batch_first=True,
            enforce_sorted=False,
        )
        del flatten_sentences

        sentence_level_gru_output, _ = self.sentence_gru(packed_sentences)
        sentence_alphas = self.get_sentence_alphas(sentence_level_gru_output)

        sentence_level_gru_output, _ = pad_packed_sequence(
            sentence_level_gru_output, batch_first=True
        )

        sentence_level_representation = (
            sentence_alphas.unsqueeze(2) * sentence_level_gru_output
        ).sum(dim=1)

        sentence_level_representation = sentence_level_representation.reshape(
            batch_size,
            max_paragraphs_per_article,
            sentence_level_representation.shape[-1],
        )

        return sentence_level_representation

    def get_word_level_representation(
        self,
        batch_size: int,
        flatten_word_ids: torch.Tensor,
        flatten_words_per_sentence: torch.Tensor,
        max_paragraphs_per_article: int,
        max_sentences_per_paragraph: int,
        max_words_per_sentence: torch.Tensor,
    ) -> torch.Tensor:
        """Obtains the representation of a sequence of words

        Parameters
        ----------
        batch_size : int
            Size of the batch
        flatten_word_ids : torch.Tensor
            Tensor with word embeddings of a sentence
        flatten_words_per_sentence : torch.Tensor
            Tensor with the number of words per sentence in the batch
        max_paragraphs_per_article : int
            Maximum number of paragraphs per article in the batch
        max_sentences_per_paragraph : int
            Maximum number of sentences per paragraph in the batch
        max_words_per_sentence : torch.Tensor
            Tensor with the maximum number of words per sentence

        Returns
        -------
        torch.Tensor

        """
        word_embeddings = self.embedding(flatten_word_ids)
        # Cleaning memory
        del flatten_word_ids

        flatten_words_per_sentence = remove_zeros(flatten_words_per_sentence)

        packed_word_embeddings = pack_padded_sequence(
            word_embeddings,
            lengths=flatten_words_per_sentence,
            batch_first=True,
            enforce_sorted=False,
        ).float()
        # Cleaning memory
        del word_embeddings
        del flatten_words_per_sentence

        word_level_gru_output, _ = self.word_gru(packed_word_embeddings)
        # Cleaning memory
        del packed_word_embeddings

        word_alphas = self.get_word_alphas(word_level_gru_output)

        # To track when memory limit is reached
        try:
            word_level_gru_output, _ = pad_packed_sequence(
                word_level_gru_output, batch_first=True
            )
        except Exception as error:
            print(
                f"Batch size: {batch_size} \n"
                f"Paragraphs per article: {max_paragraphs_per_article} \n"
                f"Sentences per paragraph: {max_sentences_per_paragraph} \n"
                f"Words per sentence: {max_words_per_sentence} "
            )

            logger.exception(error)

            exit(1)

        word_level_representation = (
            word_alphas.unsqueeze(2) * word_level_gru_output
        ).sum(dim=1)

        return word_level_representation

    def get_sentence_alphas(
        self, sentence_level_gru: torch.nn.utils.rnn.PackedSequence
    ) -> torch.Tensor:
        """Obtains the alphas from the sentence representations

        Parameters
        ----------
        sentence_level_gru : torch.nn.utils.rnn.PackedSequence
            Output of sentence level GRU

        Returns
        -------
        torch.Tensor

        """
        sentence_state = torch.tanh(self.sentence_attention(sentence_level_gru.data))

        sentence_context_vector = self.sentence_context_layer(sentence_state).squeeze(1)

        exponent = torch.exp(sentence_context_vector)

        del sentence_context_vector
        del sentence_state

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentence_attention, _ = pad_packed_sequence(
            PackedSequence(
                data=exponent,
                batch_sizes=sentence_level_gru.batch_sizes,
                sorted_indices=sentence_level_gru.sorted_indices,
                unsorted_indices=sentence_level_gru.unsorted_indices,
            ),
            batch_first=True,
        )

        sentence_alphas = sentence_attention / torch.sum(
            sentence_attention, dim=1, keepdim=True
        )
        del sentence_attention

        return sentence_alphas

    def get_word_alphas(
        self, word_gru_out: torch.nn.utils.rnn.PackedSequence
    ) -> torch.Tensor:
        """Obtains the alphas from the word representations

        Parameters
        ----------
        word_gru_out : torch.nn.utils.rnn.PackedSequence
            Output of words GRU

        Returns
        -------
        torch.Tensor

        """
        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        # This equation is represented in the paper as `u_{it}`
        word_state = torch.tanh(self.word_attention(word_gru_out.data))

        word_context_vector = self.word_context_layer(word_state).squeeze(1)
        # (n_words)

        exponent = torch.exp(word_context_vector)

        del word_state
        del word_context_vector

        word_attention, _ = pad_packed_sequence(
            PackedSequence(
                data=exponent,
                batch_sizes=word_gru_out.batch_sizes,
                sorted_indices=word_gru_out.sorted_indices,
                unsorted_indices=word_gru_out.unsorted_indices,
            ),
            batch_first=True,
        )

        word_alphas = word_attention / torch.sum(word_attention, dim=1, keepdim=True)

        return word_alphas

    def transform_to_word_level(
        self, article: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Prepares the article to be processed at the word level

        Parameters
        ----------
        article : Dict[str, List[str]]
            Document to be prepared

        Returns
        -------
        Dict[str, List[str]]

        """
        batch_size = article[TEXT_IDS_COLUMN].shape[0]

        article[WORDS_PER_SENTENCE_COLUMN] = get_words_per_document_at_word_level(
            article[WORDS_PER_SENTENCE_COLUMN]
        )
        article[TEXT_IDS_COLUMN] = get_document_at_word_level(
            article[TEXT_IDS_COLUMN], article[WORDS_PER_SENTENCE_COLUMN], self.device
        )
        article[SENTENCES_PER_PARAGRAPH_COLUMN] = torch.ones(
            (batch_size, 1), dtype=int, device=self.device
        )
        article[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(
            batch_size, dtype=int, device=self.device
        )

        return article

    def transform_to_sentence_level(
        self, article: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Prepares the article to be processed at the sentence level

        Parameters
        ----------
        article : Dict[str, List[str]]
            Article to be prepared

        Returns
        -------
        Dict[str, List[str]]

        """
        batch_size = article[TEXT_IDS_COLUMN].shape[0]

        article[TEXT_IDS_COLUMN] = get_document_at_sentence_level(
            article[TEXT_IDS_COLUMN], self.device
        )
        article[WORDS_PER_SENTENCE_COLUMN] = get_words_per_sentence_at_sentence_level(
            article[WORDS_PER_SENTENCE_COLUMN], self.device
        )
        article[
            SENTENCES_PER_PARAGRAPH_COLUMN
        ] = get_sentences_per_paragraph_at_sentence_level(
            article[SENTENCES_PER_PARAGRAPH_COLUMN]
        )
        article[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(
            batch_size, dtype=int, device=self.device
        )

        return article
