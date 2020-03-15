import logging
import json
import torch
import pandas as pd
from ast import literal_eval
from datetime import datetime
from torch.utils.data import TensorDataset, random_split
from src.smash_dataset import SMASHDataset
from src.utils import get_max_lengths


def get_click_stream_dump(click_stream_dump_path):
    rows = []

    with open(click_stream_dump_path, 'r') as f:
        for i, line in enumerate(f):
            cols = line.split('\t')

            if cols[2] == 'link':  # type
                rows.append([
                    cols[0],  # prev
                    cols[1],  # current
                    int(cols[3]),  # n
                ])

    # TODO use dumps from more months

    return pd.DataFrame(rows, columns=['prev', 'current', 'n'])


def get_available_titles(wiki_documents_path):
    # Load preprocessed
    title2sects = {}

    with open(wiki_documents_path, 'r') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            title = doc['title'].replace(' ', '_')
            title2sects[title] = [sect['paragraphs'] for sect in doc['sections']]

    print(f'Completed after {i} lines')

    return set(title2sects.keys())  # save as set (makes it faster)


def add_metrics(fdf):
    max_n = fdf.groupby(['prev']).agg({'n': 'max'})

    # Normalize click count with max value
    fdf['rel_n'] = 0.

    for idx, r in fdf.iterrows():
        fdf.at[idx, 'rel_n'] = r['n'] / max_n['n'][r['prev']]

    return fdf


def filter_available_titles(click_stream, available_titles):
    filtered_dataset = click_stream[
        (click_stream['prev'].isin(available_titles)) & (click_stream['current'].isin(available_titles))].copy()

    return filtered_dataset


def generate_click_stream_dataset(click_stream_dump_path, dataset_path):
    click_stream = get_click_stream_dump(click_stream_dump_path)
    available_titles = get_available_titles(dataset_path)

    click_stream_dataset = filter_available_titles(click_stream, available_titles)
    click_stream_dataset = add_metrics(click_stream_dataset)
    click_stream_dataset.columns = ['previous_article', 'current_article', 'number_of_clicks', 'click_rate']

    return click_stream_dataset


def combine_wiki_click_stream_datasets(click_stream_dataset, wiki_documents_dataset):
    combined_dataset = pd.merge(click_stream_dataset, wiki_documents_dataset, left_on=['previous_article'],
                                right_on=['article'])
    combined_dataset = pd.merge(combined_dataset, wiki_documents_dataset, left_on=['current_article'],
                                right_on=['article'])
    combined_dataset = combined_dataset.drop(columns=['article_x', 'article_y'])
    combined_dataset.columns = ['previous_article', 'current_article', 'number_of_clicks', 'click_rate',
                                'previous_article_text', 'current_article_text']
    combined_dataset = combined_dataset[
        ['previous_article', 'previous_article_text', 'current_article', 'current_article_text', 'number_of_clicks',
         'click_rate']]

    return combined_dataset


def get_wiki_documents_from_json(wiki_documents_path):
    texts, labels, articles = [], [], []

    with open(wiki_documents_path, 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        introduction_json = result['sections'][0]['paragraphs']
        # Cleaning empty paragraphs and sentences
        sentences_to_append = [filtered_sentence for filtered_sentence in
                               [[sentence for sentence in paragraph if sentence]
                                for paragraph in introduction_json if paragraph] if filtered_sentence]

        if sentences_to_append:
            texts.append(sentences_to_append)
            articles.append(result['title'])

    wiki_documents_dataset = pd.DataFrame(list(zip(articles, texts)), columns=['article', 'text'])

    return wiki_documents_dataset


def generate_dataset(click_stream_dump_path, wiki_documents_path):
    """
    Process:
        - Download Simple English data XML (not implemented)
        - Transform to JSON using wiki_cli.py (not implemented)
        - Download click stream (not implemented)
        - Generate click stream dataset for titles available on Simple English - ok
        - Combine click stream dataset with JSON dataset
        - Save result as CSV

    :return: void
    """
    print('Generating dataset')

    click_stream_dataset = generate_click_stream_dataset(click_stream_dump_path, wiki_documents_path)
    wiki_documents_dataset = get_wiki_documents_from_json(wiki_documents_path)

    combined_dataset = combine_wiki_click_stream_datasets(click_stream_dataset, wiki_documents_dataset)

    return combined_dataset


def create_tensor_dataset(dataset_path, word2vec_path, limit_rows_dataset):
    print('Getting max lengths', datetime.now())
    max_word_length, max_sent_length, max_paragraph_length = get_max_lengths(dataset_path, limit_rows_dataset)
    print('Finished getting max lengths', datetime.now())

    print('Loading complete dataset', datetime.now())
    complete_dataset = SMASHDataset(dataset_path, word2vec_path, max_sent_length,
                                    max_word_length,
                                    max_paragraph_length, limit_rows_dataset)
    print('Complete dataset loaded', datetime.now())

    print('Reading word ids from documents', datetime.now())
    words_ids_current_document = [literal_eval(text) for text in complete_dataset.current_article_text.values]
    words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document = get_padded_document_structures(
        words_ids_current_document)
    print('Transforming dataset into tensors', datetime.now())
    current_document_tensor = get_document_tensor(words_ids_current_document, max_word_length, max_sent_length,
                                                  max_paragraph_length)
    print('Finished Transforming dataset into tensors', datetime.now())

    print('Reading word ids from documents - previous document', datetime.now())
    words_ids_previous_document = [literal_eval(text) for text in
                                   complete_dataset.previous_article_text.values]
    words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document = get_padded_document_structures(
        words_ids_previous_document)
    print('Transforming dataset into tensors', datetime.now())
    previous_document_tensor = get_document_tensor(words_ids_previous_document, max_word_length, max_sent_length,
                                                   max_paragraph_length)
    print('Finished Transforming dataset into tensors', datetime.now())

    print('Transforming click_rate_tensor dataset into tensors', datetime.now())
    click_rate_tensor = torch.Tensor([click_rate for click_rate in complete_dataset.click_rate])
    print('Finished tensors creation', datetime.now())

    dataset = TensorDataset(current_document_tensor, words_per_sentence_current_document,
                            sentences_per_paragraph_current_document, paragraphs_per_document_current_document,
                            previous_document_tensor, words_per_sentence_previous_document,
                            sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document,
                            click_rate_tensor
                            )

    torch.save(dataset, './data/complete_dataset.pth')
    print('Saved TensorDataset {}'.format(datetime.now()))

    del current_document_tensor, words_per_sentence_current_document, \
        sentences_per_paragraph_current_document, paragraphs_per_document_current_document, \
        previous_document_tensor, words_per_sentence_previous_document, \
        sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, \
        click_rate_tensor

    print('Finished creating second dataset into tensors - previous', datetime.now())


def get_document_tensor(documents, max_word_length, max_sent_length, max_paragraph_length):
    __slot__ = ('document_placeholder')
    num_documents = len(documents)

    document_placeholder = torch.zeros((num_documents, max_paragraph_length, max_sent_length, max_word_length),
                                       dtype=int)

    print('Number of documents: {}'.format(num_documents))

    for document_idx, document in enumerate(documents):
        if (document_idx % 100) == 0:
            print('Finished document {} of {}. {}'.format(document_idx, num_documents, datetime.now()))
        for paragraph_idx, paragraph in enumerate(document):
            for sentence_idx, sentence in enumerate(paragraph):
                document_placeholder[document_idx, paragraph_idx, sentence_idx, 0:len(sentence)] = torch.LongTensor(
                    sentence)

    return document_placeholder


def get_padded_document_structures(self, words_ids_a):
    document_structures = [self.complete_dataset.get_document_structure(document) for document in words_ids_a]
    words_per_sentences_tensor = torch.LongTensor([
        SMASHDataset.get_padded_words_per_sentence(document_structure['words_per_sentence']) for document_structure in
        document_structures])
    sentences_per_paragraph_tensor = torch.LongTensor([
        SMASHDataset.get_padded_sentences_per_paragraph(document_structure['sentences_per_paragraph']) for
        document_structure in document_structures])
    paragraphs_per_document_tensor = torch.LongTensor(
        [document_structure['paragraphs_per_document'] for document_structure in document_structures])

    return words_per_sentences_tensor, sentences_per_paragraph_tensor, paragraphs_per_document_tensor


def split_dataset(dataset, train_split, batch_size, train_dataset_path='../data/training.pth',
                  validation_dataset_path='../data/validation.pth', test_dataset_path='../data/test.pth'):
    print('Beginning of dataset split')
    dataset_size = len(dataset)
    train_dataset_size = int(dataset_size * train_split)
    validation_dataset_size = int((dataset_size - train_dataset_size) / 2)
    test_dataset_size = dataset_size - train_dataset_size - validation_dataset_size

    train_dataset, validation_dataset, test_dataset = random_split(dataset,
                                                                   [train_dataset_size, validation_dataset_size,
                                                                    test_dataset_size])

    print('Datasets split. Starting saving them', datetime.now())

    training_params = {'batch_size': batch_size,
                       'shuffle': True,
                       'drop_last': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, **training_params)

    validation_and_test_params = {'batch_size': batch_size,
                                  'shuffle': True,
                                  'drop_last': False}
    validation_loader = torch.utils.data.DataLoader(validation_dataset, **validation_and_test_params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **validation_and_test_params)

    torch.save(train_loader, train_dataset_path)
    torch.save(validation_loader, validation_dataset_path)
    torch.save(test_loader, test_dataset_path)

    print('Datasets saved successfully')


def save_tensor_dataset(words_ids, name):
    words_per_sentence, sentences_per_paragraph, paragraphs_per_document = get_padded_document_structures(
        words_ids)
    document_tensor = get_document_tensor(words_ids)
    dataset = TensorDataset(document_tensor, words_per_sentence, sentences_per_paragraph, paragraphs_per_document)

    torch.save(dataset, name + '.pth')

    del document_tensor, words_per_sentence, sentences_per_paragraph, paragraphs_per_document, dataset


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    print('Running main')

    click_stream_dump_path = '../data/clickstream-enwiki-2019-08.tsv'
    wiki_documents_path = '../data/simplewiki.jsonl'

    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # get_wiki_documents_from_json(wiki_documents_path)

    # wiki_dataset = generate_dataset(click_stream_dump_path, wiki_documents_path)
    # wiki_dataset.to_csv('../data/wiki_df.csv',
    #                     index=False,
    #                     header=['previous_article', 'previous_article_text', 'current_article',
    #                             'current_article_text', 'number_of_clicks', 'click_rate'])

    # print(wiki_dataset.sample(n=10))

    # LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    # logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    #
    # cdf = read_click_stream_dump(cs_dump_path)
    #
    # print(f'Total click pairs: {len(cdf):,}')
    #
    # available_titles = get_available_titles(docs_path)
    #
    # fdf = normalize_dataset(cdf, available_titles)
    #
    # print(fdf.sample(n=10))
    #
    # fdf.to_csv('../data/click_pair.csv',
    #            index=False,
    #            header=['previous_article', 'current_article', 'number_of_clicks', 'click_rate'])
