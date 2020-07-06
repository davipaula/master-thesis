import json
import multiprocessing
import sys
import os
from datetime import datetime

import torch
from tqdm import tqdm

import pandas as pd

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging
from itertools import compress

import gensim
import spacy

from database import ArticlesDatabase

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class WikiArticlesTokenizer:
    def __init__(self, wiki_pre_processed_path: str, output_path: str, w2v_path: str):
        self.__wiki_pre_processed = wiki_pre_processed_path
        self.__output_path = output_path

        __spacy_model = "en_core_web_sm"

        # disable the fancy and slow stuff and add only the pipeline needed
        self._nlp = spacy.load(__spacy_model, disable=["tagger", "ner", "textcat", "parser"])
        self._nlp.add_pipe(self._nlp.create_pipe("sentencizer"))

        logger.info(f"Spacy model loaded: {__spacy_model}")

        self.__w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)

        self.docs = {}

        self.db = ArticlesDatabase()

    def tokenize(self, title, text):
        # Tokenize + find word indexes for tokens
        article_text = text["text"]
        tokenized_article = {"paragraphs": [], "normalized_paragraphs": []}

        for section in article_text:
            tokenized_paragraphs = []
            normalized_paragraphs = []

            paragraphs = list(self._nlp.pipe(section))

            for paragraph in paragraphs:
                tokenized_paragraph, normalized_paragraph = self.process_paragraph(paragraph)

                if tokenized_paragraph:
                    tokenized_paragraphs.append(tokenized_paragraph)
                    normalized_paragraphs.append(normalized_paragraph)

            tokenized_article["paragraphs"].append(tokenized_paragraphs)
            tokenized_article["normalized_paragraphs"].append(normalized_paragraphs)

        return {"title": title, "sections": tokenized_article}

    def process_paragraph(self, text):
        """
        Split plain paragraph text into sentences and tokens, and find their word vectors (with Gensim)
        :param nlp:
        :param w2v_model:
        :param text:
        :return: sentences -> word indexes
        """

        tokenized_sentences = []
        normalized_sentences = []

        for sentence in text.sents:
            valid_tokens = [self.is_valid_token(token) for token in sentence]

            valid_words = list(compress(sentence, valid_tokens))
            if valid_words:
                sentence_embedding_ids = [self.__w2v_model.vocab[word.lemma_.lower()].index for word in valid_words]
                valid_words = [word.lemma_.lower() for word in valid_words]

                tokenized_sentences.append(sentence_embedding_ids)
                normalized_sentences.append(valid_words)

        return tokenized_sentences, normalized_sentences

    def is_valid_token(self, token):
        normalized_word = token.lemma_.lower()

        return token.is_alpha and not token.is_stop and normalized_word in self.__w2v_model.vocab

    @staticmethod
    def load_existing_articles(output_path: str):
        with open(output_path, "r") as f:
            json_list = list(f)

        article_titles = []

        for json_str in tqdm(json_list):
            result = json.loads(json_str)
            article_titles.append(result["title"])

        return article_titles

    def process_article(self, article_title):
        articles_text = self.db.get_text_from_article(article_title)
        # print(f"Selected articles: {len(selected_articles)}")
        # print(f"Loaded articles: {len(articles_text)}")

        article = json.loads(articles_text[1])

        article_text = [section["text"].split("\n\n") for section in article]

        doc = {articles_text[0]: {"text": article_text}}

        for title, text in tqdm(doc.items()):
            result = self.tokenize(title, text)

        print(result)

    def process_from_db(self):
        logger.info("Loading selected articles")
        with open(SELECTED_ARTICLES_PATH, "r") as selected_articles_file:
            selected_articles = [article_title.rstrip("\n") for article_title in selected_articles_file]

        logger.info("Loaded selected articles")
        start = datetime.now()
        articles_text = self.db.get_text_from_articles(selected_articles)
        logger.info(f"Loaded articles from DB. Time elapsed {datetime.now() - start}")
        # print(f"Selected articles: {len(selected_articles)}")
        # print(f"Loaded articles: {len(articles_text)}")

        for result in tqdm(articles_text):
            article = json.loads(result[1])

            if not article:
                continue

            article_text = [section["text"].split("\n\n") for section in article]

            self.docs[result[0]] = {"text": article_text}

        with open(self.__output_path, "w+") as f:
            for title, text in tqdm(self.docs.items()):
                f.write(json.dumps(self.tokenize(title, text)) + "\n")

    def process(self, append: bool = True) -> None:
        """

        :rtype: None
        :param wiki_dump_path:
        :param output_path: 
        :param w2v_path: 
        :param append: 
        :param limit: 
        """
        if append:
            logger.info(f"Appending to output: {self.__output_path}")
            open_mode = "a"
        else:
            logger.info(f"Creating new output file: {self.__output_path}")
            open_mode = "w"

        if not os.path.exists(self.__wiki_pre_processed):
            logger.error(f"Wiki dump does not exist at: {self.__wiki_pre_processed}")
            exit(1)

        logger.info("Loading selected articles")
        with open(SELECTED_ARTICLES_PATH, "r") as selected_articles_file:
            selected_articles = [article_title.rstrip("\n") for article_title in selected_articles_file]

        logger.info("Loading articles")
        with open(self.__wiki_pre_processed, "r") as wiki_dump:
            for line in tqdm(wiki_dump):
                article = json.loads(line)

                if not article["sections"]:
                    continue

                article_text = [section["text"].split("\n\n") for section in article["sections"]]

                self.docs[article["title"]] = {"text": article_text}

        logger.info("Filtering articles")
        selected_docs = {title: text for (title, text) in self.docs.items() if title in selected_articles}

        print(f"Selected articles: {len(selected_articles)}")
        print(f"Loaded articles: {len(selected_docs)}")

        logger.info("Articles loaded. Starting tokenization")

        start = datetime.now()

        with open(self.__output_path, "w+") as f:
            for title, text in tqdm(selected_docs.items()):
                f.write(json.dumps(self.tokenize(title, text)) + "\n")

        time_elapsed = datetime.now() - start
        print(f"Finished. Time elapsed {time_elapsed}")

        # # Removed multiprocessing because this is slowing down the process. Keeping the code here for future improvement
        # workers = max(1, multiprocessing.cpu_count() - 1)
        # # workers = 1
        #
        # pool = multiprocessing.Pool(workers)
        #
        # pool_out = pool.map(self.tokenize_pool, self.docs.items())
        #
        # pool.close()
        # pool.join()

        # with open(self.__output_path, "w+") as f:
        #     for doc in pool_out:
        #         f.write(json.dumps(doc) + "\n")

        logger.info("Tokens saved")

    def tokenize_pool(self, args):
        return self.tokenize(*args)


if __name__ == "__main__":
    is_development = True

    _w2v_path = "./data/source/glove.6B.200d.w2vformat.txt"
    _wiki_dump_path = "./data/source/enwiki-20200401-pages-articles.xml.bz2"
    _wiki_pre_processed_path = "./data/processed/enwiki_raw_text.jsonl"
    _output_path = "./data/processed/enwiki_tokenized_selected_articles.jsonl"
    SELECTED_ARTICLES_PATH = "./data/processed/selected_articles.txt"

    if is_development:
        os.chdir("/Users/dnascimentodepau/Documents/python/thesis/thesis-davi")
        _w2v_path = "./data/source/glove.6B.200d.w2vformat.1k.txt"

    tokenizer = WikiArticlesTokenizer(_wiki_pre_processed_path, _output_path, _w2v_path)
    # tokenizer.process()
    tokenizer.process_article("Extinction")
