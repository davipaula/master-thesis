import sys
import os

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging

from preparation import convert_to_word2vec
from preparation.click_stream_extractor import ClickStreamExtractor
from preparation.available_titles_extractor import AvailableTitlesExtractor
from preparation.wiki_articles_extractor import extract_wiki_articles
from preparation.wiki_articles_tokenizer import WikiArticlesTokenizer
from processing.click_stream_processor import ClickStreamProcessor
from processing.wiki_articles_processor import WikiArticlesProcessor

click_stream_dump_path = (
    "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/clickstream-enwiki-2020-03.tsv"
)
wiki_titles_path = (
    "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/enwiki-20200401-all-titles-in-ns0.gz"
)
wiki_dump_path = (
    "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/enwiki-20200401-pages-articles.xml.bz2"
)
available_titles_save_path = (
    "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/available_titles.txt"
)

wiki_pre_processed_path = (
    "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/enwiki_articles.jsonl"
)

wiki_articles_tokenized_path = "./data/processed/enwiki_tokens.jsonl"

# Problem. This doesn't exist before running convert_to_word2vec
w2v_path = "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/glove.6B.50d.w2vformat.txt"

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
        logger.info(f"Converting Glove file to Word2Vec format")
        convert_to_word2vec.convert("./data/source/glove.6B.200d.txt", "./data/source/glove.6B.200d.w2vformat.txt")
        #
        # logger.info(f"Extracting Click Stream data")
        # ClickStreamExtractor(click_stream_dump_path).run()
        #
        # logger.info(f"Generating Available Titles")
        # AvailableTitlesExtractor(wiki_titles_path).run(available_titles_save_path)
        #
        # logger.info("Extractiing Wiki articles")
        # extract_wiki_articles(wiki_dump_path=wiki_dump_path, output_path=wiki_pre_processed_path, limit=100)

        # logger.info(f"Generating Click Stream Dataset")
        # ClickStreamProcessor().run()

        # logger.info("Tokenizing articles")
        # WikiArticlesTokenizer(wiki_pre_processed_path, wiki_articles_tokenized_path, w2v_path).process()
        #
        # logger.info("Creating dataset with Wiki Articles")
        # WikiArticlesProcessor(wiki_articles_tokenized_path).run()


if __name__ == "__main__":
    os.chdir("/Users/dnascimentodepau/Documents/python/thesis/thesis-davi")

    DatasetCreator().run()
