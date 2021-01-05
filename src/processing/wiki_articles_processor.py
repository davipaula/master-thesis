import itertools
import json

import pandas as pd
from tqdm import tqdm

from ..utils.constants import WIKI_ARTICLES_DATASET_PATH, WIKI_ARTICLES_TOKENIZED_PATH
from ..utils.utils import clean_title


def create_wiki_articles_dataset():
    wiki_articles = process_wiki_tokenized()
    wiki_articles.to_csv(WIKI_ARTICLES_DATASET_PATH, index=False)


def process_wiki_tokenized() -> pd.DataFrame:
    """Processes the tokenized articles and generates a DataFrame

    Returns
    -------
    pd.DataFrame

    """
    text_ids = []
    text_string = []
    articles = []
    text_ids_intro = []

    with open(WIKI_ARTICLES_TOKENIZED_PATH, "r") as json_file:
        json_list = list(json_file)

    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        sections = result["tokenized_text"]
        raw_text = result["raw_text"]

        if not sections:
            continue

        # The original structure of a Wikipedia article is article <- sections <- paragraphs <- sentences <- words
        # This removes the `sections` dimension
        article_text_ids = list(itertools.chain.from_iterable(sections))
        article_raw_text = list(itertools.chain.from_iterable(raw_text))

        if not article_text_ids:
            continue

        if sections[0]:
            article_text_ids_intro = sections[0]
        else:
            article_text_ids_intro = [article_text_ids[0]]
            # Workaround for the cases where the introduction is null

        text_ids.append(article_text_ids)
        text_string.append(article_raw_text)
        articles.append(clean_title(result["title"]))
        text_ids_intro.append(article_text_ids_intro)

    return pd.DataFrame(
        list(zip(articles, text_ids, text_string, text_ids_intro)),
        columns=["article", "text_ids", "raw_text", "text_ids_intro"],
    )
