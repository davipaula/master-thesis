import logging
from database import ArticlesDatabase

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

db = ArticlesDatabase()

def process() -> None:
    cdef list selected_articles_file
    logger.info("Loading selected articles")

    file_path = f"/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/selected_articles.txt"
    with open(file_path, "r") as selected_articles_file:
        selected_articles = [article_title.rstrip("\n") for article_title in selected_articles_file]

    selected_articles = selected_articles [:10]

    articles_text = db.get_text_from_articles(selected_articles)

    print(articles_text)

if __name__ == "__main__":
    process()