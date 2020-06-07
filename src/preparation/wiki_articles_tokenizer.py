import json
import multiprocessing
import sys
import os
from datetime import datetime

import torch
from tqdm import tqdm

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging
from itertools import compress

import gensim
import spacy

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class WikiArticlesTokenizer:
    def __init__(self, wiki_dump_path: str, output_path: str, w2v_path: str):
        self.__wiki_dump_path = wiki_dump_path
        self.__output_path = output_path

        __spacy_model = "en_core_web_sm"

        # disable the fancy and slow stuff and add only the pipeline needed
        self._nlp = spacy.load(__spacy_model, disable=["tagger", "ner", "textcat", "parser"])
        self._nlp.add_pipe(self._nlp.create_pipe("sentencizer"))

        logger.info(f"Spacy model loaded: {__spacy_model}")

        self.__w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)

        # self.__existing_articles = self.load_existing_articles(self.__output_path)

        self.docs = {}

    def tokenize(self, title, article_text):
        # Tokenize + find word indexes for tokens
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

        if not os.path.exists(self.__wiki_dump_path):
            logger.error(f"Wiki dump does not exist at: {self.__wiki_dump_path}")
            exit(1)

        """
            - Load all documents
                - Ok
            - Load only text. Generate: - ok
                {
                "title": abc,
                "text": [[[Sentence 1 2 3], [Sentence 1, 2, 3]]]
                }
            - Tokenize
        """

        n = 0
        with open(self.__wiki_dump_path, "r") as wiki_dump:
            for line in tqdm(wiki_dump):
                article = json.loads(line)

                article_text = []
                for section in article["sections"]:
                    article_text.append(section["text"].split("\n\n"))

                self.docs[article["title"]] = article_text

                n += 1

                if n == 5000:
                    break

        logger.info("Articles loaded. Starting tokenization")

        start = datetime.now()

        # for title, article in self.docs.items():
        #     self.tokenize(title, article)

        processes = []
        for title, article in self.docs.items():
            p = multiprocessing.Process(target=self.tokenize, args=(title, article))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

        # workers = max(1, multiprocessing.cpu_count() - 1)
        #
        # pool = multiprocessing.Pool(workers)
        #
        # pool_out = pool.map(self.tokenize_pool, self.docs.items())
        #
        # # stop threads when done
        # pool.close()
        # pool.join()

        time_elapsed = datetime.now() - start
        print(f"Finished. Time elapsed {time_elapsed}")

        # with open(self.__output_path, open_mode) as f:
        #     for doc in process_dump(wiki_dump_path, nlp, w2v_model, existing_articles, limit):
        #         f.write(json.dumps(doc) + "\n")

    def tokenize_pool(self, args):
        return self.tokenize(*args)


if __name__ == "__main__":
    if torch.cuda.is_available():
        _w2v_path = "/home/dnascimento/thesis-davi/data/source/glove.6B.50d.w2vformat.txt"
        _wiki_dump_path = "/home/dnascimento/thesis-davi/data/source/enwiki-20200401-pages-articles.xml.bz2"
        _wiki_pre_processed_path = "/home/dnascimento/thesis-davi/data/processed/enwiki_raw_text.jsonl"
        SELECTED_ARTICLES_PATH = "/home/dnascimento/thesis-davi/data/processed/selected_articles.csv"

    else:
        _w2v_path = (
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/glove.6B.200d.w2vformat.1k.txt"
        )
        _wiki_dump_path = "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/enwiki-20200401-pages-articles.xml.bz2"
        _wiki_pre_processed_path = (
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/enwiki_raw_text.jsonl"
        )
        SELECTED_ARTICLES_PATH = (
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/selected_articles.csv"
        )

    tokenizer = WikiArticlesTokenizer(_wiki_pre_processed_path, _wiki_pre_processed_path, _w2v_path)
    tokenizer.process()
