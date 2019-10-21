import logging
import json
import pandas as pd


def read_click_stream_dump(cs_dump_path):
    rows = []

    with open(cs_dump_path, 'r') as f:
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


def get_available_titles(docs_path):
    # Load preprocessed
    title2sects = {}

    with open(docs_path, 'r') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            title = doc['title'].replace(' ', '_')
            title2sects[title] = [sect['paragraphs'] for sect in doc['sections']]

    print(f'Completed after {i} lines')

    return set(title2sects.keys())  # save as set (makes it faster)


def normalize_dataset(cdf, available_titles):

    # Clicks for that we have matching articles
    fdf = cdf[(cdf['prev'].isin(available_titles)) & (cdf['current'].isin(available_titles))].copy()

    print(f'Click pairs with articles: {len(fdf):,}')

    max_n = fdf.groupby(['prev']).agg({'n': 'max'})

    # Normalize click count with max value
    fdf['rel_n'] = 0.

    for idx, r in fdf.iterrows():
        fdf.at[idx, 'rel_n'] = r['n'] / max_n['n'][r['prev']]

    return fdf


if __name__ == '__main__':

    print('Running main')

    cs_dump_path = '../data/clickstream-enwiki-2019-08.tsv'
    docs_path = '../data/simplewiki.jsonl'

    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

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



