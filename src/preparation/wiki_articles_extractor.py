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
import sys
import os

from tqdm import tqdm

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import bz2
import re
import json
import torch
import logging
import gensim
import spacy
import pandas as pd
from itertools import compress
from gensim.corpora.wikicorpus import get_namespace
from gensim.scripts.segment_wiki import extract_page_xmls
from utils.extractor_utils import dropNested, replaceInternalLinks, replaceExternalLinks
from xml.etree import cElementTree
from gensim.scripts.segment_wiki import segment
from typing import List

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


def remove_special_characters(text):
    text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", text)
    text = re.sub(r"\'s", " 's ", text)
    text = re.sub(r"\'ve", " 've ", text)
    text = re.sub(r"n\'t", " 't ", text)
    text = re.sub(r"\'re", " 're ", text)
    text = re.sub(r"\'d", " 'd ", text)
    text = re.sub(r"\'ll", " 'll ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


def process_paragraph(text, w2v_model):
    """
    Split plain paragraph text into sentences and tokens, and find their word vectors (with Gensim)
    :param nlp:
    :param w2v_model:
    :param text:
    :return: sentences -> word indexes
    """

    tokenized_sentences = []
    normalized_sentences = []

    for sentence in text.sents:
        valid_tokens = [is_valid_token(token, w2v_model) for token in sentence]

        valid_words = list(compress(sentence, valid_tokens))
        if valid_words:
            sentence_embedding_ids = [w2v_model.vocab[word.lemma_.lower()].index for word in valid_words]
            valid_words = [word.lemma_.lower() for word in valid_words]

            tokenized_sentences.append(sentence_embedding_ids)
            normalized_sentences.append(valid_words)

    return tokenized_sentences, normalized_sentences


def is_valid_token(token, w2v_model):
    normalized_word = token.lemma_.lower()

    return token.is_alpha and not token.is_stop and normalized_word in w2v_model.vocab


def process_text(nlp, w2v_model, text):
    """

    :param nlp: Spacy model
    :param w2v_model: Gensim word2vec
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
        sects.append(
            {"title": "Introduction", "text": plain_text,}
        )

    # Tokenize + find word indexes for tokens
    for i, sect in enumerate(sects):
        sects[i]["paragraphs"] = []
        sects[i]["normalized_paragraphs"] = []

        tokenized_paragraphs = list(nlp.pipe(sect["text"].split("\n\n")))

        for paragraph in tokenized_paragraphs:
            paragraph_tokens, normalized_paragraph = process_paragraph(paragraph, w2v_model)

            if paragraph_tokens:
                sects[i]["paragraphs"].append(paragraph_tokens)
                sects[i]["normalized_paragraphs"].append(normalized_paragraph)

        # break  # returns only the introduction (the first section)

    return sects


def process_dump(
    wiki_dump_path: str,
    nlp,
    w2v_model,
    existing_articles: List[str],
    max_doc_count=0,
    filter_selected_articles=True,
    log_every=100,
):
    """

    :param existing_articles: List with already existing articles
    :param filter_selected_articles: Flag to filter only selected articles. If false all articles in dump file will be processed (not recommended)
    :param log_every: Print process every X docs
    :param wiki_dump_path: Path to Wikipedia XML dump
    :param nlp: Spacy NLP model
    :param w2v_model: Gensim Word2Vec model
    :param max_doc_count: limit the number of articles to be returned (0 = no limit)
    :return: Generator for processed docs (Wikipedia articles)
    """
    doc_counter = 0

    logger.info(f"Processing dump from: {wiki_dump_path}")

    if filter_selected_articles:
        selected_articles = pd.read_csv(SELECTED_ARTICLES_PATH)

    with bz2.open(wiki_dump_path, "rb") as xml_fileobj:
        page_xmls = extract_page_xmls(xml_fileobj)

        for i, page_xml in enumerate(tqdm(page_xmls)):
            # Parse XML element
            elem = cElementTree.fromstring(page_xml)
            filter_namespaces = ("0",)
            namespace = get_namespace(elem.tag)
            ns_mapping = {"ns": namespace}
            text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
            title_path = "./{%(ns)s}title" % ns_mapping
            ns_path = "./{%(ns)s}ns" % ns_mapping

            title = elem.find(title_path).text

            if filter_selected_articles:
                if title not in selected_articles["article"].to_list():
                    continue

            if title in existing_articles:
                continue

            text = elem.find(text_path).text
            ns = elem.find(ns_path).text

            # Get the article name of the links in the document
            links_in_document = list(segment(page_xml, include_interlinks=True)[2])

            # Filter invalid namespaces (user pages, etc)
            if ns not in filter_namespaces:
                continue

            # Ignore redirects
            if "#REDIRECT" in text or "#redirect" in text or "#Redirect" in text:
                continue

            yield {
                "title": title,
                "sections": process_text(nlp, w2v_model, text),
                "links": links_in_document,
            }

            doc_counter += 1

            if 0 < max_doc_count < doc_counter:
                break

    logger.info(f"Finished processing dump from: {wiki_dump_path}")


def load_existing_articles(output_path: str):
    with open(output_path, "r") as f:
        json_list = list(f)

    article_titles = []

    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        article_titles.append(result["title"])

    return article_titles


def extract_wiki_articles(wiki_dump_path: str, w2v_path: str, output_path: str, limit=0, append=True):

    if append:
        logger.info(f"Appending to output: {output_path}")
        open_mode = "a"
    else:
        logger.info(f"Creating new output file: {output_path}")
        open_mode = "w"

    if not os.path.exists(wiki_dump_path):
        logger.error(f"Wiki dump does not exist at: {wiki_dump_path}")
        exit(1)

    spacy_model = "en_core_web_sm"

    nlp = spacy.load(spacy_model, disable=["tagger", "ner", "textcat", "parser"])  # disable the fancy and slow stuff
    nlp.add_pipe(nlp.create_pipe("sentencizer"))

    logger.info(f"Spacy model loaded: {spacy_model}")

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)

    existing_articles = load_existing_articles(output_path)

    with open(output_path, open_mode) as f:
        for doc in process_dump(wiki_dump_path, nlp, w2v_model, existing_articles, limit):
            f.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    if torch.cuda.is_available():
        _w2v_path = "/home/dnascimento/thesis-davi/data/source/glove.6B.50d.w2vformat.txt"
        _wiki_dump_path = "/home/dnascimento/thesis-davi/data/source/enwiki-20200401-pages-articles.xml.bz2"
        _wiki_pre_processed_path = "/home/dnascimento/thesis-davi/data/processed/enwiki_entire_article.jsonl"
        SELECTED_ARTICLES_PATH = "/home/dnascimento/thesis-davi/data/processed/selected_articles.csv"

    else:
        _w2v_path = "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/glove.6B.50d.w2vformat.txt"
        _wiki_dump_path = "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/enwiki-20200401-pages-articles.xml.bz2"
        _wiki_pre_processed_path = (
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/enwiki_entire_article_2.jsonl"
        )
        SELECTED_ARTICLES_PATH = (
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/selected_articles.csv"
        )

    limit = 30

    extract_wiki_articles(
        wiki_dump_path=_wiki_dump_path, w2v_path=_w2v_path, output_path=_wiki_pre_processed_path, limit=limit
    )
