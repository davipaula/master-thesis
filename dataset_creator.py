import sys
import os

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging

from preparation import convert_to_word2vec
from preparation.click_stream_extractor import ClickStreamExtractor
from preparation.wiki_articles_extractor import extract_wiki_articles
from preparation.wiki_articles_tokenizer import WikiArticlesTokenizer
from processing.click_stream_processor import ClickStreamProcessor
from processing.wiki_articles_processor import WikiArticlesProcessor

from database import ArticlesDatabase

from utils.constants import (
    CLICK_STREAM_DUMP_PATH,
    WIKI_TITLES_PATH,
    WIKI_DUMP_PATH,
    WIKI_ARTICLES_TOKENIZED_PATH,
    WORD2VEC_PATH,
)

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class DatasetCreator:
    """
    - Convert Glove to Wiki2Vec format
    - Pre-process click stream
        - Extract data from click stream dump
            - Keep only `type == 'link'`
            - Clean titles
            - Add metrics
        - Save pre-processed file
    - Prepare page titles
        - Clean titles
        - Save pre-processed file
    - Create click stream dataset
        - Open file
        - Filter available titles in N0
        - Generate random sample
        - Save 'selected_articles.txt'
        - Divide sample into train, validation and test
    - Create Wiki dataset
        - Open 'selected_articles.txt'
        - Open Wiki articles dump
        - Tokenize articles that are in 'selected_articles.txt'
    """

    def __init__(self):
        pass

    def run(self):
        logger.info(f"Process started:")
        # logger.info(f"Converting Glove file to Word2Vec format")
        # convert_to_word2vec.convert("./data/source/glove.6B.200d.txt", "./data/source/glove.6B.200d.w2vformat.txt")
        #
        # logger.info(f"Extracting Click Stream data")
        # ClickStreamExtractor().run()
        #
        # logger.info("Extracting Wiki articles")
        # extract_wiki_articles()
        #
        # logger.info(f"Generating Available Titles")
        # articles_db = ArticlesDatabase()
        # articles_db.generate_available_articles()

        # logger.info(f"Generating Click Stream Dataset")
        # ClickStreamProcessor().run()

        logger.info("Tokenizing articles")
        WikiArticlesTokenizer().process()

        # logger.info("Creating dataset with Wiki Articles")
        # WikiArticlesProcessor().run()


if __name__ == "__main__":
    os.chdir("/Users/dnascimentodepau/Documents/python/thesis/thesis-davi")

    DatasetCreator().run()
