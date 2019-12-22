import logging
import json
import pandas as pd


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

        if result['title'] == 'Augsburg':
            print('Augsburg')

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

    wiki_dataset = generate_dataset(click_stream_dump_path, wiki_documents_path)
    wiki_dataset.to_csv('../data/wiki_df.csv',
                        index=False,
                        header=['previous_article', 'previous_article_text', 'current_article',
                                'current_article_text', 'number_of_clicks', 'click_rate'])

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
