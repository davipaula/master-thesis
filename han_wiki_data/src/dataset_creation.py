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


def add_articles_data(data_path, fdf):
    texts, labels, articles = [], [], []

    with open(data_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)

        texts.append(result['sections'][0]['paragraphs'])
        articles.append(result['title'])

    articles_df = pd.DataFrame(list(zip(articles, texts)), columns=['article', 'text'])
    click_stream_df = pd.read_csv(fdf)

    click_stream_df = click_stream_df.astype('object')

    merged_df = pd.merge(click_stream_df, articles_df, left_on='previous_article', right_on='article')
    merged_df = pd.merge(merged_df, articles_df, left_on=['current_article'], right_on=['article'])
    merged_df = merged_df.drop(columns=['article_x', 'article_y'])
    merged_df.columns = ['previous_article', 'current_article', 'number_of_clicks', 'click_rate',
                         'previous_article_text', 'current_article_text']
    merged_df = merged_df[
        ['previous_article', 'previous_article_text', 'current_article', 'current_article_text', 'number_of_clicks',
         'click_rate']]

    return merged_df


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    print('Running main')

    cs_dump_path = '../data/clickstream-enwiki-2019-08.tsv'
    docs_path = '../data/simplewiki.jsonl'

    wiki_df = add_articles_data('../data/simplewiki.jsonl', '../data/click_pair_small.csv')

    wiki_df.to_csv('../data/wiki_df_small.csv', index=False)

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
