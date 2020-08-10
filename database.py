import logging
import sqlite3
from collections import Counter
from datetime import datetime
from typing import List

import jsonlines
import json

from tqdm import tqdm
import pandas as pd

from utils.constants import AVAILABLE_TITLES_PATH

logger = logging.getLogger(__name__)
LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ArticlesDatabase:
    def __init__(self):
        self.conn = sqlite3.connect("./wikipedia_dump.db")
        self.cursor = self.conn.cursor()
        self.cursor.arraysize = 10

    def create_table(self):
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
        query = "ALTER TABLE articles ADD COLUMN in_links_count INTEGER"
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
        query = f"UPDATE articles SET word_count = ? WHERE title = ?"

        try:
            self.cursor.execute(query, (word_count, article_title))
        except Exception as e:
            logger.error("Error inserting data")
            logger.info(query)
            logger.info(e)
            exit(1)

    def insert_out_links_count(self, article_title, number_of_links):
        query = f"UPDATE articles SET out_links_count = ? WHERE title = ?"

        try:
            self.cursor.execute(query, (number_of_links, article_title))
        except Exception as e:
            logger.error("Error inserting data")
            logger.info(query)
            logger.info(e)
            exit(1)

    def insert_in_links_count(self, article_title, number_of_links):
        query = f"UPDATE articles SET in_links_count = ? WHERE title = ?"

        try:
            self.cursor.execute(query, (number_of_links, article_title))
        except Exception as e:
            logger.error("Error inserting data")
            logger.info(query)
            logger.info(e)
            exit(1)

    def calculate_number_of_words(self):

        all_titles = self.get_all_titles()
        titles_count = len(all_titles)

        chunk_size = 10000
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

                for section in json.loads(article[1]):
                    word_count += len(section["text"].split())

                self.insert_word_count(article[0], word_count)

            self.conn.commit()

            start = end
            end += min(chunk_size, titles_count)

        logger.info(f"Finished inserting. Time elapsed: {datetime.now() - start_time}")

    def calculate_out_links_count(self):
        all_titles = self.get_all_titles()
        titles_count = len(all_titles)

        chunk_size = 10000
        start = 0
        end = chunk_size

        logger.info(f"Started inserting")
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

    def get_features_from_articles(self, articles: List[str]) -> list:
        query_placeholders = ','.join(['?'] * len(articles))
        query = (
            f"SELECT title, word_count, out_links_count, in_links_count "
            f"FROM articles WHERE title IN ({query_placeholders})"
        )
        result = self.cursor.execute(query, articles).fetchall()

        return result


if __name__ == "__main__":
    articles_database = ArticlesDatabase()
    print(articles_database.get_features_from_articles(["Anarchism"]))
