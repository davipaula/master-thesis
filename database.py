import csv
import json
import logging
import os.path
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from typing import List

import jsonlines
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.utils.constants import AVAILABLE_TITLES_PATH, WORD2VEC_50D_PATH

WIKIPEDIA_DUMP_DB_PATH = "./wikipedia_dump.db"

logger = logging.getLogger(__name__)
LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ArticlesDatabase:
    def __init__(self):
        self.check_database()
        self.conn = sqlite3.connect(WIKIPEDIA_DUMP_DB_PATH)
        self.cursor = self.conn.cursor()
        self.cursor.arraysize = 10

    def check_database(self):
        if not os.path.isfile(WIKIPEDIA_DUMP_DB_PATH):
            self.prepare_database()

    def prepare_database(self):
        logger.info("Database does not exist. Starting database preparation")
        self.create_table()
        self.create_index()

    def create_table(self):
        logger.info("Creating table")
        self.cursor.execute(
            "CREATE TABLE ARTICLES (title TEXT, sections TEXT, links TEXT)"
        )
        self.conn.commit()

    def create_tokenized_table(self):
        self.cursor.execute(
            "CREATE TABLE PROCESSED_ARTICLES (title TEXT, tokenized_text TEXT, raw_text TEXT)"
        )
        self.cursor.execute(
            "CREATE INDEX idx_article_title ON PROCESSED_ARTICLES (title)"
        )
        self.conn.commit()

    def add_tokenized_articles(self, title, tokenized_text, raw_text):
        query = (
            "INSERT INTO PROCESSED_ARTICLES (title, tokenized_text, raw_text) VALUES (?, ?, ?) "
            "ON CONFLICT(title) DO NOTHING"
        )
        self.cursor.execute(
            query, (title, json.dumps(tokenized_text), json.dumps(raw_text))
        )

    def add_articles(self, title, sections, links):
        query = "INSERT INTO ARTICLES (title, sections, links) VALUES (?, ?, ?) ON CONFLICT(title) DO NOTHING"
        self.cursor.execute(query, (title, json.dumps(sections), json.dumps(links)))

    def add_new_columns(self):
        query = "ALTER TABLE articles ADD COLUMN sentence_count INTEGER"
        self.cursor.execute(query)
        self.conn.commit()

    def import_articles(self):
        json_path = "./data/processed/enwiki_raw_text.jsonl"

        with jsonlines.open(json_path) as json_file:
            for article in tqdm(json_file):
                self.add_articles(
                    article["title"], article["sections"], article["links"]
                )

        self.conn.commit()

        logger.info("Finished")

    def get_links_from_article(self, article):
        query = "SELECT TITLE, LINKS FROM ARTICLES WHERE TITLE = ? LIMIT 1"
        return self.cursor.execute(query, (article,)).fetchmany()

    def get_links_from_articles(self, articles):
        query = f"SELECT TITLE, LINKS FROM ARTICLES WHERE TITLE IN ({','.join(['?'] * len(articles))})"
        return self.cursor.execute(query, articles).fetchall()

    def get_valid_articles(self, articles):
        self.cursor.row_factory = lambda cursor, row: row[0]
        query = f"SELECT TITLE FROM ARTICLES WHERE TITLE IN ({','.join(['?'] * len(articles))})"
        results = self.cursor.execute(query, articles).fetchall()
        self.cursor.row_factory = None

        return results

    def create_index(self):
        query = "CREATE INDEX idx_article_title ON articles (title)"
        self.cursor.execute(query)
        self.conn.commit()

        print("Finished creating index")

    def get_text_from_article(self, article):
        query = "SELECT TITLE, SECTIONS FROM ARTICLES WHERE TITLE = ?"

        return self.cursor.execute(query, (article,)).fetchone()

    def get_text_from_articles(self, articles):
        query = f"SELECT TITLE, SECTIONS FROM ARTICLES WHERE TITLE IN ({','.join(['?'] * len(articles))})"
        result = self.cursor.execute(query, articles).fetchall()

        return result

    def generate_available_articles(self):
        self.cursor.row_factory = lambda cursor, row: row[0]
        query = "SELECT DISTINCT TITLE FROM ARTICLES"
        available_articles = self.cursor.execute(query).fetchall()
        pd.Series(available_articles).to_csv(
            AVAILABLE_TITLES_PATH, header=False, index=False
        )
        self.cursor.row_factory = None

    def get_all_titles(self):
        self.cursor.row_factory = lambda cursor, row: row[0]
        query = "SELECT title FROM articles"
        result = self.cursor.execute(query).fetchall()
        self.cursor.row_factory = None

        return result

    def insert_word_count(self, article_title, word_count):
        query = "UPDATE articles SET word_count = ? WHERE title = ?"

        try:
            self.cursor.execute(query, (word_count, article_title))
        except Exception as e:
            logger.error("Error inserting data")
            logger.info(query)
            logger.info(e)
            exit(1)

    def insert_paragraph_and_sentence_count(
        self, article_title: str, paragraph_count: int, sentence_count: int
    ) -> None:
        query = "UPDATE articles SET paragraph_count = ?, sentence_count = ? WHERE title = ?"

        try:
            self.cursor.execute(query, (paragraph_count, sentence_count, article_title))
        except Exception as e:
            logger.error("Error inserting data")
            logger.info(query)
            logger.info(e)
            exit(1)

    def insert_out_links_count(self, article_title, number_of_links):
        query = "UPDATE articles SET out_links_count = ? WHERE title = ?"

        try:
            self.cursor.execute(query, (number_of_links, article_title))
        except Exception as e:
            logger.error("Error inserting data")
            logger.info(query)
            logger.info(e)
            exit(1)

    def insert_in_links_count(self, article_title, number_of_links):
        query = "UPDATE articles SET in_links_count = ? WHERE title = ?"

        try:
            self.cursor.execute(query, (number_of_links, article_title))
        except Exception as e:
            logger.error("Error inserting data")
            logger.info(query)
            logger.info(e)
            exit(1)

    def calculate_number_of_words(self):

        # all_titles = self.get_all_titles()

        all_titles = pd.read_csv("./data/test_articles.csv", header=None)[0].to_list()
        titles_count = len(all_titles)

        chunk_size = 1000
        start = 0
        end = chunk_size

        logger.info(f"Started inserting")
        start_time = datetime.now()
        for n in tqdm(range(int(titles_count / chunk_size))):
            current_titles = all_titles[start:end]
            articles_text = self.get_text_from_articles(current_titles)

            for article in articles_text:
                word_count = 0
                if not json.loads(article[1]):
                    continue

                article_text = ""
                for section in json.loads(article[1]):
                    article_text += section["text"]

                clean_text = BeautifulSoup(article_text, "lxml").text
                word_count += len(clean_text.split())

                self.insert_word_count(article[0], word_count)

            self.conn.commit()

            start = end
            end += min(chunk_size, titles_count)

        logger.info(f"Finished inserting. Time elapsed: {datetime.now() - start_time}")

    def calculate_number_of_words_clean(self):
        spacy_model = "en_core_web_sm"

        # disable the fancy and slow stuff and add only the pipeline needed
        nlp = spacy.load(spacy_model, disable=["tagger", "ner", "textcat", "parser"])
        nlp.add_pipe(nlp.create_pipe("sentencizer"))

        all_titles = pd.read_csv(
            "./results/test/results_smash_paragraph_level.csv", usecols=[1]
        )["source_article"].unique()
        titles_count = len(all_titles)

        word2vec = pd.read_csv(
            filepath_or_buffer=WORD2VEC_50D_PATH,
            header=None,
            sep="\s",
            engine="python",
            quoting=csv.QUOTE_NONE,
            skiprows=1,
            usecols=[0],
        )[0].to_list()

        chunk_size = 1
        start = 0
        end = chunk_size

        article_length = defaultdict(int)

        logger.info(f"Started inserting")
        start_time = datetime.now()
        for n in tqdm(range(int(titles_count / chunk_size))):
            current_titles = all_titles[start:end]
            articles_text = self.get_text_from_articles(current_titles)

            for article in articles_text:
                if not json.loads(article[1]):
                    continue

                article_text = ""
                for section in json.loads(article[1]):
                    article_text += section["text"]

                clean_text = BeautifulSoup(article_text, "lxml").text

                tokenized_sentences = nlp(clean_text)
                tokenized_words = [
                    token.lemma_.lower()
                    for sentence in tokenized_sentences.sents
                    for token in sentence
                    if token.is_alpha
                ]

                words_in_word2vec = [token in word2vec for token in tokenized_words]

                word_count = sum(words_in_word2vec)

                article_length[article[0]] = len(tokenized_words)

                # self.insert_word_count(article[0], word_count)

            # self.conn.commit()

            start = end
            end += min(chunk_size, titles_count)

        df_article_length = pd.DataFrame.from_dict(
            article_length, orient="index"
        ).reset_index()
        df_article_length.columns = ["article", "word_count"]

        df_article_length.to_csv("./data/articles_word_count.csv", index=False)

        logger.info(f"Finished inserting. Time elapsed: {datetime.now() - start_time}")

    def calculate_out_links_count(self):
        all_titles = self.get_all_titles()
        titles_count = len(all_titles)

        chunk_size = 10000
        start = 0
        end = chunk_size

        logger.info("Started inserting")
        start_time = datetime.now()
        for n in tqdm(range(int(titles_count / chunk_size))):
            current_titles = all_titles[start:end]
            articles_link = self.get_links_from_articles(current_titles)

            for article in articles_link:
                number_of_links = len(json.loads(article[1]))
                self.insert_out_links_count(article[0], number_of_links)

            self.conn.commit()

            start = end
            end += min(chunk_size, titles_count)

        logger.info(f"Finished inserting. Time elapsed: {datetime.now() - start_time}")

    def calculate_in_links_count(self):
        all_titles = self.get_all_titles()
        titles_count = len(all_titles)

        chunk_size = 10000
        start = 0
        end = chunk_size

        logger.info(f"Started counting")
        links_counter = Counter()
        start_time = datetime.now()
        for n in tqdm(range(int(titles_count / chunk_size))):
            current_titles = all_titles[start:end]
            articles_link = self.get_links_from_articles(current_titles)

            for article in articles_link:
                articles_link = json.loads(article[1])
                links_counter.update(articles_link)

            start = end
            end += min(chunk_size, titles_count)

        logger.info(f"Finished counting. Starting insertion")

        for title, in_links in tqdm(links_counter.items()):
            self.insert_in_links_count(title, in_links)

        self.conn.commit()
        logger.info(f"Finished inserting. Time elapsed: {datetime.now() - start_time}")

    def calculate_paragraph_and_sentence_count(self):
        all_titles = self.get_all_titles()
        titles_count = len(all_titles)

        chunk_size = 10000
        start = 0
        end = chunk_size

        logger.info(f"Started calculating paragraph and sentence counts")
        start_time = datetime.now()
        for _ in tqdm(range(int(titles_count / chunk_size))):
            current_titles = all_titles[start:end]
            articles_text = self.get_text_from_articles(current_titles)

            for article_title, article_text in articles_text:
                normalized_article_text = json.loads(article_text)
                if not normalized_article_text:
                    continue

                number_of_paragraphs = sum(
                    [
                        len(section["text"].split("\n\n"))
                        for section in normalized_article_text
                    ]
                )
                number_of_sentences = sum(
                    [
                        len(section["text"].replace("\n\n", " ").split(". "))
                        for section in normalized_article_text
                    ]
                )

                self.insert_paragraph_and_sentence_count(
                    article_title, number_of_paragraphs, number_of_sentences
                )

            self.conn.commit()

            start = end
            end += min(chunk_size, titles_count)

        logger.info(f"Finished inserting. Time elapsed: {datetime.now() - start_time}")

    def get_features_from_articles(self, articles: List[str]) -> list:
        query_placeholders = ",".join(["?"] * len(articles))
        query = (
            f"SELECT title, out_links_count, in_links_count, paragraph_count, sentence_count "
            f"FROM articles WHERE title IN ({query_placeholders})"
        )
        result = self.cursor.execute(query, articles).fetchall()

        return result


if __name__ == "__main__":
    articles_database = ArticlesDatabase()
    articles_database.calculate_number_of_words_clean()
    # print(articles_database.get_features_from_articles(["Anarchism"]))
