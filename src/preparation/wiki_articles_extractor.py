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
import multiprocessing
import sys
import os
from datetime import datetime

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import bz2
import re
import json
import torch
import logging
from gensim.corpora.wikicorpus import get_namespace
from gensim.scripts.segment_wiki import extract_page_xmls
from utils.extractor_utils import dropNested, replaceInternalLinks, replaceExternalLinks
from xml.etree import cElementTree
from gensim.scripts.segment_wiki import segment
from typing import List
from tqdm import tqdm

from utils.constants import WIKI_DUMP_PATH

from database import ArticlesDatabase

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def convert_to_plain_text(text):
    """
    Convert Wikitext to plain text

    :param text:
    :return: cleaned text
    """
    text = dropNested(text, r"{{", r"}}")
    text = dropNested(text, r"{\|", r"\|}")

    # replace internal links
    text = replaceInternalLinks(text)

    # replace external links
    text = replaceExternalLinks(text)

    return text.strip()


def process_text(text):
    """
    :param text: Wikitext
    :return: Sections dict(title, text, paragraphs[])
    """
    appendices_and_footers_sections = [
        "See also",
        "References",
        "Sources",
        "Further reading",
        "External links",
        "Notes",
        "Works",
        "Publications",
        "Discography",
        "Bibliography",
        "Filmography",
    ]
    headline_matches = list(re.finditer(r"^([=]{2,5})([^=]+?)([=]{2,5})", text, re.MULTILINE))
    sects = []

    # Extract sections from document text
    # TODO: there is a bug that is inserting empty paragraphs. The code below should be refactored, allowing to insert
    # TODO: a new sect only if sect[i]["paragraphs"] is not empty
    for i, match in enumerate(headline_matches):
        if len(sects) == 0:
            plain_sect_text = convert_to_plain_text(text[: match.start()])

            if plain_sect_text:
                sects.append({"title": "Introduction", "text": plain_sect_text})

        title = match.group(2).strip()

        # If text hits any of the appendices and footers sections, we should stop collecting data for the article
        # See:https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Layout#Standard_appendices_and_footers
        if title in appendices_and_footers_sections:
            break

        # ignore subsection hierarchy for now
        # level = len(match.group(1).strip()) - 1

        if i < len(headline_matches) - 1:
            next_match = headline_matches[i + 1]
            sect_text = text[match.end() : next_match.start()]
        else:
            # last section
            sect_text = text[match.end() :]

        plain_sect_text = convert_to_plain_text(sect_text)

        if plain_sect_text:
            sects.append(
                {"title": title, "text": plain_sect_text,}
            )

    # no sections found -> article consists of only a single section
    plain_text = convert_to_plain_text(text)
    if not plain_text:
        return

    if len(sects) == 0:
        sects.append({"title": "Introduction", "text": plain_text})

    return sects


def process_dump(page_xml):
    """
    :param log_every: Print process every X docs
    :param wiki_dump_path: Path to Wikipedia XML dump
    :param max_doc_count: limit the number of articles to be returned (0 = no limit)
    :return: Generator for processed docs (Wikipedia articles)
    """

    # Parse XML element
    elem = cElementTree.fromstring(page_xml)
    filter_namespaces = ("0",)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping
    ns_path = "./{%(ns)s}ns" % ns_mapping

    title = elem.find(title_path).text

    text = elem.find(text_path).text
    ns = elem.find(ns_path).text

    # Get the article name of the links in the document
    links_in_document = list(segment(page_xml, include_interlinks=True)[2])

    # Filter invalid namespaces (user pages, etc)
    if ns not in filter_namespaces:
        return

    # Ignore redirects
    if "#REDIRECT" in text or "#redirect" in text or "#Redirect" in text:
        return

    return {
        "title": title,
        "sections": process_text(text),
        "links": links_in_document,
    }


def extract_wiki_articles() -> None:
    """
    Extracts XML from wiki dump and stores it into the database
    :return:
    """
    if not os.path.exists(WIKI_DUMP_PATH):
        logger.error(f"Wiki dump does not exist at: {WIKI_DUMP_PATH}")
        exit(1)

    wiki_dump_xml = bz2.open(WIKI_DUMP_PATH, "rb")
    wiki_pages_xml = extract_page_xmls(wiki_dump_xml)

    logger.info("Articles loaded. Starting extraction")

    start = datetime.now()

    # To disable multiprocessing, use workers = 1
    workers = max(1, multiprocessing.cpu_count() - 1)

    pool = multiprocessing.Pool(workers)

    pool_output = pool.map(process_dump, wiki_pages_xml)

    pool.close()
    pool.join()

    time_elapsed = datetime.now() - start
    logger.info(f"Finished extracting. Time elapsed {time_elapsed}")

    articles_db = ArticlesDatabase()

    for page in pool_output:
        if page:
            articles_db.add_articles(page["title"], page["sessions"], page["links"])

    logger.info("Tokens saved")
