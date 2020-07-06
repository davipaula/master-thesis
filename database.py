import logging
import sqlite3
from datetime import datetime

import jsonlines
import json

from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ArticlesDatabase:
    def __init__(self):
        self.conn = sqlite3.connect("./wikipedia_dump.db")
        self.cursor = self.conn.cursor()
        self.cursor.arraysize = 10

    def create_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS ARTICLES")
        self.cursor.execute("CREATE TABLE ARTICLES (title TEXT, sections TEXT, links TEXT)")
        self.conn.commit()

    def add_articles(self, title, sections, links):
        query = "INSERT INTO ARTICLES (title, sections, links) VALUES (?, ?, ?)"
        self.cursor.execute(query, (title, json.dumps(sections), json.dumps(links)))

    def import_articles(self):
        json_path = "./data/processed/enwiki_raw_text.jsonl"

        with jsonlines.open(json_path) as json_file:
            for article in tqdm(json_file):
                self.add_articles(article["title"], article["sections"], article["links"])

        self.conn.commit()

        logger.info("Finished")

    def get_links_from_article(self, article):
        query = "SELECT TITLE, LINKS FROM ARTICLES WHERE TITLE = ? LIMIT 1"
        return self.cursor.execute(query, (article,)).fetchmany()

    def get_links_from_articles(self, articles):
        query = f"SELECT TITLE, LINKS FROM ARTICLES WHERE TITLE IN ({','.join(['?']*len(articles))})"
        return self.cursor.execute(query, articles).fetchall()

    def get_valid_articles(self, articles):
        self.cursor.row_factory = lambda cursor, row: row[0]
        query = f"SELECT TITLE FROM ARTICLES WHERE TITLE IN ({','.join(['?']*len(articles))})"
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
        query = f"SELECT TITLE, SECTIONS FROM ARTICLES WHERE TITLE IN ({','.join(['?']*len(articles))})"
        result = self.cursor.execute(query, articles).fetchall()

        return result

    def generate_available_articles(self):
        self.cursor.row_factory = lambda cursor, row: row[0]
        query = "SELECT DISTINCT TITLE FROM ARTICLES"
        available_articles = self.cursor.execute(query).fetchall()
        pd.Series(available_articles).to_csv("./data/processed/available_titles.txt", header=False, index=False)
        self.cursor.row_factory = None


if __name__ == "__main__":
    source_articles = [
        "Anarchism",
        "Autism",
        "Albedo",
        "A",
        "Alabama",
        "Achilles",
        "Abraham Lincoln",
        "Aristotle",
        "An American in Paris",
        "Academy Award for Best Production Design",
        "Academy Awards",
        "Actrius",
        "Animalia (book)",
        "International Atomic Time",
        "Altruism",
        "Ayn Rand",
        "Alain Connes",
        "Allan Dwan",
        "Algeria",
        "List of Atlas Shrugged characters",
        "Anthropology",
        "Agricultural science",
        "Alchemy",
        "Alien",
        "Astronomer",
        "ASCII",
        "Austin (disambiguation)",
        "Animation",
        "Apollo",
        "Andre Agassi",
        "Austroasiatic languages",
        "Afroasiatic languages",
        "Andorra",
        "Arithmetic mean",
        "American Football Conference",
        "Animal Farm",
        "Amphibian",
        "Alaska",
        "Agriculture",
        "Aldous Huxley",
        "Ada",
        "Aberdeen (disambiguation)",
        "Algae",
        "Analysis of variance",
        "Alkane",
        "Appellate procedure in the United States",
        "Answer (law)",
        "Appellate court",
        "Arraignment",
        "America the Beautiful",
        "Assistive technology",
        "Abacus",
        "Acid",
        "Asphalt",
        "American National Standards Institute",
        "Argument (disambiguation)",
        "Apollo 11",
        "Apollo 8",
        "Astronaut",
        "A Modest Proposal",
        "Alkali metal",
        "Alphabet",
        "Atomic number",
        "Anatomy",
        "Affirming the consequent",
        "Andrei Tarkovsky",
        "Ambiguity",
        "Animal (disambiguation)",
        "Aardvark",
        "Aardwolf",
        "Adobe",
        "Adventure",
        "Asia",
        "Aruba",
        "Articles of Confederation",
        "Asia Minor (disambiguation)",
        "Atlantic Ocean",
        "Arthur Schopenhauer",
        "Angola",
        "Demographics of Angola",
        "Politics of Angola",
        "Economy of Angola",
        "Transport in Angola",
        "Angolan Armed Forces",
        "Foreign relations of Angola",
        "Albert Sidney Johnston",
        "Android (robot)",
        "Alberta",
        "List of anthropologists",
        "Actinopterygii",
        "Albert Einstein",
        "Afghanistan",
        "Albania",
        "Allah",
        "Algorithms (journal)",
        "Azerbaijan",
        "Amateur astronomy",
        "Aikido",
        "Art",
        "Agnostida",
    ]

    articles_database = ArticlesDatabase()
    value = articles_database.get_text_from_article("Lage Raho Munna Bhai")
    print(value)
    # logger.info("Creating table")
    # articles_database.create_table()
    # logger.info("Creating index")
    # articles_database.create_index()
    # logger.info("Importing articles")

    # articles_database.import_articles()
