CLICK_STREAM_DUMP_PATH = "./data/source/clickstream-enwiki-2020-03.tsv"
CLICK_STREAM_PROCESSED_PATH = "./data/processed/click_stream.csv"

WIKI_TITLES_PATH = "./data/source/enwiki-20200401-all-titles-in-ns0.gz"

WIKI_DUMP_PATH = "./data/source/enwiki-20200401-pages-articles.xml.bz2"

AVAILABLE_TITLES_PATH = "./data/processed/available_titles.txt"

WORD2VEC_50D_PATH = "./data/source/glove.6B.50d.w2vformat.txt"
WORD2VEC_200D_PATH = "./data/source/glove.6B.200d.w2vformat.txt"

SELECTED_ARTICLES_PATH = "./data/processed/selected_articles.txt"

WIKI_ARTICLES_EXTRACTED_PATH = "./data/processed/enwiki_articles.jsonl"
WIKI_ARTICLES_TOKENIZED_PATH = (
    "./data/processed/enwiki_tokenized_selected_articles.jsonl"
)
WIKI_ARTICLES_DATASET_PATH = "./data/processed/wiki_articles_english_complete.csv"

TRAIN_DATASET_PATH = "./data/processed/train.csv"
VALIDATION_DATASET_PATH = "./data/processed/validation.csv"
TEST_DATASET_PATH = "./data/processed/test.csv"

CLICK_STREAM_TRAIN_DATASET_PATH = "./data/dataset/click_stream_train.pth"
CLICK_STREAM_VALIDATION_DATASET_PATH = "./data/dataset/click_stream_validation.pth"
CLICK_STREAM_TEST_DATASET_PATH = "./data/dataset/click_stream_test.pth"

ARTICLE_COLUMN = "article"
WORD_COUNT_COLUMN = "word_count"
OUT_LINKS_COUNT_COLUMN = "out_links_count"
IN_LINKS_COUNT_COLUMN = "in_links_count"

MODEL_COLUMN = "model"
PREDICTED_CLICK_RATE_COLUMN = "predicted_click_rate"
ACTUAL_CLICK_RATE_COLUMN = "actual_click_rate"
TARGET_ARTICLE_COLUMN = "target_article"
SOURCE_ARTICLE_COLUMN = "source_article"

RESULT_FILE_COLUMNS_NAMES = [
    MODEL_COLUMN,
    SOURCE_ARTICLE_COLUMN,
    TARGET_ARTICLE_COLUMN,
    ACTUAL_CLICK_RATE_COLUMN,
    PREDICTED_CLICK_RATE_COLUMN,
]
