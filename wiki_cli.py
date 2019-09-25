"""

You can use this script to obtain sections/paragraphs/sentences and their word indexes from a Wiki dump.

USAGE:

python wiki_cli.py process <wiki_dump_path> <word2vec_path> <output_path> [<max_doc_count>]

EXAMPLE:

python wiki_cli.py process /Volumes/data/repo/data/simplewiki-latest-pages-articles.xml \
    /Volumes/data/repo/data/glove.6B/glove.6B.200d.w2vformat.txt \
    simplewiki.jsonl 100

OUTPUT:

JSON-line file (one valid JSON per line)

{"title": "Contact network", "sections": [{"title": "Introduction", "text": "'''Contact network''' may mean:\n*Creative network\n*Social network\n*Power network", "paragraphs": [[[1871, 849, 107, 1702, 4069, 849, 659, 849, 268, 849]]]}]}

"""
import json
import os
import fire
import logging
import gensim
import spacy

from wiki_processing.processor import process_dump

logger = logging.getLogger(__name__)

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def process(wiki_dump_path: str, w2v_path: str, output_path: str, limit=0, override=False):

    if os.path.exists(output_path):
        if override:
            logger.info(f'Override existing output: {output_path}')
        else:
            logger.error(f'Output file exist already: {output_path}')
            exit(1)

    if not os.path.exists(wiki_dump_path):
        logger.error(f'Wiki dump does not exist at: {wiki_dump_path}')
        exit(1)

    spacy_model = 'en_core_web_sm'

    nlp = spacy.load(spacy_model, disable=['tagger', 'ner'])  # disable the fancy and slow stuff
    logger.info(f'Spacy model loaded: {spacy_model}')

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)

    with open(output_path, 'w') as f:
        for doc in process_dump(wiki_dump_path, nlp, w2v_model, limit):
            f.write(json.dumps(doc) + '\n')


if __name__ == '__main__':
    fire.Fire()