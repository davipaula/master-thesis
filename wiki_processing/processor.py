import logging
import re
import spacy
from xml.etree import cElementTree

from gensim.corpora.wikicorpus import get_namespace, filter_wiki
from gensim.scripts.segment_wiki import extract_page_xmls

from wiki_processing.extractor_utils import dropNested, replaceInternalLinks, replaceExternalLinks


logger = logging.getLogger(__name__)


def clean_text(text):
    """
    Convert Wikitext to plain text

    :param text:
    :return: cleaned text
    """
    text = dropNested(text, r'{{', r'}}')
    text = dropNested(text, r'{\|', r'\|}')

    # replace internal links
    text = replaceInternalLinks(text)

    # replace external links
    text = replaceExternalLinks(text)

    return text.strip()

def process_paragraph(nlp, w2v_model, text):
    """
    Split plain paragraph text into sentences and tokens, and find their word vectors (with Gensim)
    :param nlp:
    :param w2v_model:
    :param text:
    :return: sentences -> word indexes
    """
    pg = nlp(text)
    sentences = []

    for sent in pg.sents:
        token_ids = []

        for token in sent:
            if token.is_alpha:

                word = token.lemma_.lower()  # norm_
                if word in w2v_model.vocab:
                    token_ids.append(w2v_model.vocab[word].index)  # TODO: Add special index for UNKNOWN + PADDING

        sentences.append(token_ids)

    return sentences

def process_text(nlp, w2v_model, text):
    """

    :param nlp: Spacy model
    :param w2v_model: Gensim word2vec
    :param text: Wikitext
    :return: Sections dict(title, text, paragraphs[])
    """
    headline_matches = list(re.finditer(r'^([=]{2,5})([^=]+?)([=]{2,5})', text, re.MULTILINE))
    sects = []

    # Extract sections from document text
    for i, match in enumerate(headline_matches):
        if len(sects) == 0:
            sects.append({
                'title': 'Introduction',
                'text': clean_text(text[:match.start()])
            })
        title = match.group(2).strip()
        level = len(match.group(1).strip()) - 1  # ignore subsection hierarchy for now

        # print(f'{title} ({level})')

        if i < len(headline_matches) - 1:
            next_match = headline_matches[i + 1]

            # print(f'--- next: {headline_matches[i + 1].group(2)}')

            sect_text = text[match.end():next_match.start()]
        else:
            # last section
            sect_text = text[match.end():]

        sects.append({
            'title': title,
            'text': clean_text(sect_text),
        })

    # no sections found -> article consists of only a single section
    if len(sects) == 0:
        sects.append({
            'title': 'Introduction',
            'text': clean_text(text),
        })

    # Tokenize + find word indexes for tokens
    for i, sect in enumerate(sects):
        sects[i]['paragraphs'] = []

        for paragraph_text in sect['text'].split('\n\n'):
            paragraph_sents = process_paragraph(nlp, w2v_model, paragraph_text)

            sects[i]['paragraphs'].append(paragraph_sents)

    return sects


def process_dump(wiki_dump_path: str, nlp, w2v_model, max_doc_count=0, log_every = 1000):
    """

    :param log_every: Print process every X docs
    :param wiki_dump_path: Path to Wikipedia XML dump
    :param nlp: Spacy NLP model
    :param w2v_model: Gensim Word2Vec model
    :param max_doc_count: limit the number of articles to be returned (0 = no limit)
    :return: Generator for processed docs (Wikipedia articles)
    """
    doc_counter = 0

    logger.info(f'Processing dump from: {wiki_dump_path}')

    with open(wiki_dump_path, 'rb') as xml_fileobj:
        page_xmls = extract_page_xmls(xml_fileobj)
        i = 0

        for i, page_xml in enumerate(page_xmls):
            # s = segment(page_xml, include_interlinks=True)
            # print(page_xml)

            # Parse XML element
            elem = cElementTree.fromstring(page_xml)
            filter_namespaces = ('0',)
            namespace = get_namespace(elem.tag)
            ns_mapping = {"ns": namespace}
            text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
            title_path = "./{%(ns)s}title" % ns_mapping
            ns_path = "./{%(ns)s}ns" % ns_mapping

            title = elem.find(title_path).text
            text = elem.find(text_path).text
            ns = elem.find(ns_path).text

            # Filter invalid namespaces (user pages, etc)
            if ns not in filter_namespaces:
                continue

            # Ignore redirects
            if '#REDIRECT' in text or '#redirect' in text:
                continue

            yield {
                'title': title,
                # 'text': text,
                'sections': process_text(nlp, w2v_model, text),
            }

            doc_counter += 1

            if (doc_counter % log_every) == 0:
                logger.debug(f'Documents completed: {doc_counter}')

            if 0 < max_doc_count < doc_counter:

                break
