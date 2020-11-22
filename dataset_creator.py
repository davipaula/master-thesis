import logging

from src.preparation import convert_to_word2vec
from src.preparation.click_stream_extractor import extract_click_stream_data
from src.preparation.wiki_articles_extractor import extract_wiki_articles
from src.preparation.wiki_articles_tokenizer import WikiArticlesTokenizer
from src.processing.click_stream_processor import generate_datasets
from src.processing.wiki_articles_processor import create_wiki_articles_dataset

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def run():
    """This function executes all the steps needed to extract the raw data from Wikipedia dump and
    Wikipedia Clickstream and create the dataset for our experiments.

    This is a fairly intense process and can take some hours, specially due to the Wikipedia dump extraction
    and SpaCy tokenization processes.

    Summary of the process:
       - Converts Glove to Wiki2Vec format
       - Pre-process clickstream data
       - Extracts raw text from Wikipedia dump
           - Adds text to database
           - Generates a list with all available articles in Wikipedia dump
       - Creates training, validation and evaluation datasets
           - Obtains articles present in Wikipedia dump and Clickstream dataset
           - Randomly selects articles to from training, validation and test datasets
       - Tokenizes articles from training, validation and evaluation datasets
            - Tokenizes only the selected articles to save time in the process
       - Create the Wiki articles dataset
            - Processes the tokenized text and creates a dataset
    """
    logger.info(f"Process started:")
    logger.info(f"Converting Glove file to Word2Vec format")
    convert_to_word2vec.convert(
        "./data/source/glove.6B.50d.txt", "./data/source/glove.6B.50d.w2vformat.txt"
    )

    logger.info(f"Extracting Click Stream data")
    extract_click_stream_data()

    logger.info("Extracting Wiki articles")
    extract_wiki_articles()

    logger.info(f"Generating Clickstream dataset")
    generate_datasets()

    logger.info("Tokenizing articles")
    WikiArticlesTokenizer().process()

    logger.info("Creating dataset with Wiki Articles")
    create_wiki_articles_dataset()
