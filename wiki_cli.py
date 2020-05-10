
import json
import os
import fire
import logging
import gensim
import spacy
from preparation.wiki_articles_extractor import process_dump

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def process(wiki_dump_path: str, w2v_path: str, output_path: str, limit=0, override=False):
    if os.path.exists(output_path):
        if override:
            logger.info(f"Override existing output: {output_path}")
        else:
            logger.error(f"Output file exist already: {output_path}")
            exit(1)

    if not os.path.exists(wiki_dump_path):
        logger.error(f"Wiki dump does not exist at: {wiki_dump_path}")
        exit(1)

    spacy_model = "en_core_web_sm"

    nlp = spacy.load(spacy_model, disable=["tagger", "ner", "textcat"])  # disable the fancy and slow stuff
    logger.info(f"Spacy model loaded: {spacy_model}")

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)

    with open(output_path, "w") as f:
        for doc in process_dump(wiki_dump_path, nlp, w2v_model, limit):
            f.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    fire.Fire()
