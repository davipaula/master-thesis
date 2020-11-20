import json
import sys
import os
from datetime import datetime

from tqdm import tqdm

from typing import List, Tuple, Dict

import logging
from itertools import compress

import gensim
import spacy


src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

from database import ArticlesDatabase

from utils.constants import (
    SELECTED_ARTICLES_PATH,
    WIKI_ARTICLES_TOKENIZED_PATH,
    WORD2VEC_50D_PATH,
)

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class WikiArticlesTokenizer:
    def __init__(self):
        __spacy_model = "en_core_web_sm"

        # disable the fancy and slow stuff and add only the pipeline needed
        self._nlp = spacy.load(
            __spacy_model, disable=["tagger", "ner", "textcat", "parser"]
        )
        self._nlp.add_pipe(self._nlp.create_pipe("sentencizer"))

        logger.info(f"Spacy model loaded: {__spacy_model}")

        self.__w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            WORD2VEC_50D_PATH
        )

        self.db = ArticlesDatabase()

    def tokenize(self, title: str, article_text: List) -> Dict:
        """Tokenizes an article

        Parameters
        ----------
        title : str
            Title of the article
        article_text : List
            Text of the article

        Returns
        -------
        Dict
        """
        # Tokenize + find word indexes for tokens
        tokenized_text = []
        raw_text = []

        for section in article_text:
            tokenized_paragraphs = []
            normalized_paragraphs = []

            paragraphs = list(self._nlp.pipe(section))

            for paragraph in paragraphs:
                tokenized_paragraph, normalized_paragraph = self.process_paragraph(
                    paragraph
                )

                if tokenized_paragraph:
                    tokenized_paragraphs.append(tokenized_paragraph)
                    normalized_paragraphs.append(normalized_paragraph)

            tokenized_text.append(tokenized_paragraphs)
            raw_text.append(normalized_paragraphs)

        return {
            "title": title,
            "tokenized_text": tokenized_text,
            "raw_text": raw_text,
        }

    def process_paragraph(self, text: List[spacy.tokens.doc.Doc]) -> Tuple[List, List]:
        """Processes a tokenized paragraph text, returning
            - word vector indices
            - tokens stemmed and filtered from stop words

        Parameters
        ----------
        text : List[spacy.tokens.doc.Doc]
            Tokenized text of the articles

        Returns
        -------
        Tuple[List, List]
        """

        tokenized_sentences = []
        normalized_sentences = []

        for sentence in text.sents:
            valid_tokens = [self.is_valid_token(token) for token in sentence]

            valid_words = list(compress(sentence, valid_tokens))
            if valid_words:
                sentence_embedding_ids = [
                    self.__w2v_model.vocab[word.lemma_.lower()].index
                    for word in valid_words
                ]
                valid_words = [word.lemma_.lower() for word in valid_words]

                tokenized_sentences.append(sentence_embedding_ids)
                normalized_sentences.append(valid_words)

        return tokenized_sentences, normalized_sentences

    def is_valid_token(self, token: spacy.tokens.token.Token) -> bool:
        """Returns `true` if token is alphanumeric AND is not a stop word AND is present in
        word2vec file

        Parameters
        ----------
        token : spacy.tokens.token.Token
            Token

        Returns
        -------
        bool
        """
        normalized_word = token.lemma_.lower()

        return (
            token.is_alpha
            and not token.is_stop
            and normalized_word in self.__w2v_model.vocab
        )

    def process(self) -> None:
        """Tokenizes the text of the articles in the training, validation
        and evaluation datasets

        Returns
        -------
        None
        """
        logger.info("Loading selected articles")
        with open(SELECTED_ARTICLES_PATH, "r") as selected_articles_file:
            selected_articles = [
                article_title.rstrip("\n") for article_title in selected_articles_file
            ]

        logger.info("Loading articles from DB")
        articles_text = self.db.get_text_from_articles(selected_articles)

        logger.info("Tokenizing  articles")
        with open(WIKI_ARTICLES_TOKENIZED_PATH, "w+") as f:
            for row in tqdm(articles_text):
                article_title = row[0]
                article_raw_text = json.loads(row[1])

                if not article_raw_text:
                    continue

                article_sessions = [
                    section["text"].split("\n\n") for section in article_raw_text
                ]

                tokenized_article = self.tokenize(article_title, article_sessions)

                f.write(json.dumps(tokenized_article) + "\n")


if __name__ == "__main__":
    tokenizer = WikiArticlesTokenizer()
    tokenizer.process()
