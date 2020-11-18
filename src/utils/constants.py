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
    "./data/processed/enwiki_tokenized_selected_articles_v2.jsonl"
)
WIKI_ARTICLES_DATASET_PATH = "./data/dataset/wiki_articles_english_complete_bkp_2.csv"

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
PARAGRAPH_COUNT_COLUMN = "paragraph_count"
SENTENCE_COUNT_COLUMN = "sentence_count"

MODEL_COLUMN = "model"
PREDICTED_CLICK_RATE_COLUMN = "predicted_click_rate"
ACTUAL_CLICK_RATE_COLUMN = "actual_click_rate"
TARGET_ARTICLE_COLUMN = "target_article"
SOURCE_ARTICLE_COLUMN = "source_article"

TEXT_IDS_COLUMN = "text_ids"
CLICK_RATE_COLUMN = "click_rate"
PARAGRAPHS_PER_DOCUMENT_COLUMN = "paragraphs_per_document"
SENTENCES_PER_PARAGRAPH_COLUMN = "sentences_per_paragraph"
WORDS_PER_SENTENCE_COLUMN = "words_per_sentence"

RESULT_FILE_COLUMNS_NAMES = [
    MODEL_COLUMN,
    SOURCE_ARTICLE_COLUMN,
    TARGET_ARTICLE_COLUMN,
    ACTUAL_CLICK_RATE_COLUMN,
    PREDICTED_CLICK_RATE_COLUMN,
]

WORD_COUNT_BIN = "word_count_bin"
OUT_LINKS_BIN = "out_links_bin"
OUT_LINKS_COLUMN = "out_links_count"
IN_LINKS_BIN = "in_links_bin"
IN_LINKS_COLUMN = "in_links_count"
PARAGRAPH_COUNT_BIN = "paragraph_count_bin"
SENTENCE_COUNT_BIN = "sentence_count_bin"
MISSING_WORDS_COLUMN = "missing_words_percentage"
MISSING_WORDS_BIN = "missing_words_percentage_bin"
MODEL_COLUMN = "model"

ALL_FEATURES = [WORD_COUNT_COLUMN, OUT_LINKS_COLUMN, IN_LINKS_COLUMN]

DOC2VEC_SIAMESE = "doc2vec_siamese"
DOC2VEC_COSINE = "doc2vec_cosine"
WIKIPEDIA2VEC_SIAMESE = "wikipedia2vec_siamese"
WIKIPEDIA2VEC_COSINE = "wikipedia2vec_cosine"
SMASH_WORD_LEVEL = "smash_word_level"
SMASH_SENTENCE_LEVEL = "smash_sentence_level"
SMASH_PARAGRAPH_LEVEL = "smash_paragraph_level"
SMASH_WORD_LEVEL_INTRODUCTION = "smash_word_level_introduction"
SMASH_SENTENCE_LEVEL_INTRODUCTION = "smash_sentence_level_introduction"
SMASH_PARAGRAPH_LEVEL_INTRODUCTION = "smash_paragraph_level_introduction"

ALL_MODELS = [
    DOC2VEC_SIAMESE,
    DOC2VEC_COSINE,
    WIKIPEDIA2VEC_SIAMESE,
    WIKIPEDIA2VEC_COSINE,
    SMASH_WORD_LEVEL,
    SMASH_SENTENCE_LEVEL,
    SMASH_PARAGRAPH_LEVEL,
    SMASH_WORD_LEVEL_INTRODUCTION,
    SMASH_SENTENCE_LEVEL_INTRODUCTION,
    SMASH_PARAGRAPH_LEVEL_INTRODUCTION,
]

INTRODUCTION_MODELS = [
    SMASH_WORD_LEVEL_INTRODUCTION,
    SMASH_SENTENCE_LEVEL_INTRODUCTION,
    SMASH_PARAGRAPH_LEVEL_INTRODUCTION,
]

COMPLETE_MODELS = [
    DOC2VEC_SIAMESE,
    WIKIPEDIA2VEC_SIAMESE,
    SMASH_WORD_LEVEL,
    SMASH_SENTENCE_LEVEL,
    SMASH_PARAGRAPH_LEVEL,
]

COMPLETE_MODELS_SAVE_CONFIG = [
    (
        PARAGRAPH_COUNT_COLUMN,
        "Source article length as paragraph count (%s equal-sized buckets)",
    ),
    (
        SENTENCE_COUNT_COLUMN,
        "Source article length as sentence count (%s equal-sized buckets)",
    ),
    (WORD_COUNT_COLUMN, "Source article length as word count (%s equal-sized buckets)"),
    (
        OUT_LINKS_COLUMN,
        "Number of links present in the source articles (%s equal-sized buckets)",
    ),
    (
        IN_LINKS_COLUMN,
        "Number of articles with links pointing to the source articles (%s equal-sized buckets)",
    ),
    (
        MISSING_WORDS_COLUMN,
        "Percentage of missing words in GloVe (%s equal-sized buckets)",
    ),
]

SMASH_MODELS = [
    SMASH_WORD_LEVEL,
    SMASH_SENTENCE_LEVEL,
    SMASH_PARAGRAPH_LEVEL,
]

SMASH_AND_INTRODUCTION_MODELS = [
    SMASH_WORD_LEVEL,
    SMASH_SENTENCE_LEVEL,
    SMASH_PARAGRAPH_LEVEL,
    SMASH_WORD_LEVEL_INTRODUCTION,
    SMASH_SENTENCE_LEVEL_INTRODUCTION,
    SMASH_PARAGRAPH_LEVEL_INTRODUCTION,
]

COSINE_MODELS = [
    DOC2VEC_SIAMESE,
    WIKIPEDIA2VEC_SIAMESE,
    DOC2VEC_COSINE,
    WIKIPEDIA2VEC_COSINE,
]

BEST_MODELS = [DOC2VEC_SIAMESE, SMASH_WORD_LEVEL, WIKIPEDIA2VEC_SIAMESE]

CLEAN_MODEL_NAMES = {
    DOC2VEC_SIAMESE: "Doc2Vec",
    DOC2VEC_COSINE: "Doc2Vec Cosine",
    WIKIPEDIA2VEC_SIAMESE: "Wikipedia2Vec",
    WIKIPEDIA2VEC_COSINE: "Wikipedia2Vec Cosine",
    SMASH_WORD_LEVEL: "SMASH RNN (P + S + W)",
    SMASH_SENTENCE_LEVEL: "SMASH RNN (P + S)",
    SMASH_PARAGRAPH_LEVEL: "SMASH RNN (P)",
    SMASH_WORD_LEVEL_INTRODUCTION: "SMASH RNN Introduction (P + S + W)",
    SMASH_SENTENCE_LEVEL_INTRODUCTION: "SMASH RNN Introduction (P + S)",
    SMASH_PARAGRAPH_LEVEL_INTRODUCTION: "SMASH RNN Introduction (P)",
}
NDCG_COLUMN = "ndcg"
MAP_COLUMN = "map"
PRECISION_COLUMN = "precision"
IS_IN_TOP_ARTICLES_COLUMN = "is_in_top_articles"
K_COLUMN = "k"
