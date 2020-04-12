import json
import torch
import pandas as pd
import random
from datetime import datetime
from torch.utils.data import TensorDataset, random_split
from src.utils import remove_special_characters, remove_special_characters_df


def get_click_stream_dump(click_stream_dump_path):
    df = pd.read_csv(click_stream_dump_path,
                     sep='\t',
                     quoting=3,
                     header=None,
                     names=['previous_article', 'current_article', 'type', 'n'],
                     dtype={'type': 'category',
                            'n': 'uint32'}
                     )

    df = df[df['type'] == 'link']
    df = df.drop(['type'], axis=1)

    return df


def get_available_titles(wiki_documents_path):
    # Load preprocessed
    title2sects = {}

    with open(wiki_documents_path, 'r') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            title = doc['title'].replace(' ', '_')
            title2sects[title] = [sect['paragraphs'] for sect in doc['sections']]

    return set(title2sects.keys())  # save as set (makes it faster)


def add_metrics(click_stream_dataset, available_titles=None):
    if available_titles is not None:
        click_stream_dataset = filter_available_titles(click_stream_dataset, available_titles)

    total_clicks = click_stream_dataset.groupby(['previous_article']).agg({'number_of_clicks': 'sum'})
    total_clicks.columns = ['total_clicks']

    click_stream_dataset = click_stream_dataset.merge(total_clicks, on='previous_article')
    click_stream_dataset['click_rate'] = round(
        click_stream_dataset['number_of_clicks'] / click_stream_dataset['total_clicks'], 6)
    click_stream_dataset = click_stream_dataset.drop(['total_clicks'], axis=1)

    return click_stream_dataset


def filter_available_titles(click_stream, available_titles):
    filtered_dataset = click_stream[
        (click_stream['previous_article'].isin(available_titles)) & (
            click_stream['current_article'].isin(available_titles))].copy()

    return filtered_dataset


def generate_click_stream_dataset(click_stream_dump_path, available_titles):
    click_stream_dataset = get_click_stream_dump(click_stream_dump_path)

    # TODO fix data issues in the processor.py
    click_stream_dataset['previous_article'] = remove_special_characters_df(click_stream_dataset['previous_article'])
    click_stream_dataset['current_article'] = remove_special_characters_df(click_stream_dataset['current_article'])

    click_stream_dataset['click_rate'] = 0

    click_stream_dataset.columns = ['previous_article', 'current_article', 'number_of_clicks', 'click_rate']

    return click_stream_dataset


def add_negative_sampling(combined_dataset, wiki_dataset):
    """
    :param combined_dataset:
    :param wiki_dataset:
    :return:
    """

    """
    - Get all pairs previous_article, previous_article_links
    - Explode previous_article_links
    - Find links that weren't visited - make a join with [previous_article, current_article]
    - Filter out visited links
    - Get data from non visited links
    
    """
    # Removing bad data. Should be fixed in the json parser
    wiki_dataset = wiki_dataset[wiki_dataset['text'].str.find('#Redirect') == -1].reset_index(drop=True)

    articles = combined_dataset[['previous_article', 'previous_article_links']].drop_duplicates(
        subset='previous_article')
    articles['previous_article_links'] = articles['previous_article_links'].astype(str).str.lower()
    articles = articles.explode('previous_article_links')

    # Checks if each link was visited. If it's NA, the link wasn't clicked
    articles_with_non_visited_links = pd.merge(articles,
                                               combined_dataset[
                                                   ['previous_article', 'current_article']],
                                               left_on=['previous_article', 'previous_article_links'],
                                               right_on=['previous_article', 'current_article'],
                                               how='left')

    # Keeps only links that weren't visited
    articles_with_non_visited_links = articles_with_non_visited_links[
        articles_with_non_visited_links['current_article'].isna()]

    # Adds the non visited links data
    negative_sampling = pd.merge(articles_with_non_visited_links,
                                 wiki_dataset,
                                 left_on=['previous_article_links'],
                                 right_on=['article'])
    negative_sampling = negative_sampling.drop(['current_article'], axis=1)
    negative_sampling.columns = ['previous_article', 'previous_article_links',
                                 'current_article', 'current_article_text', 'current_article_links']

    # Adds the previous_article text data. This is done in a separate step to reduce memory consumption
    negative_sampling = pd.merge(negative_sampling,
                                 wiki_dataset[['article', 'text']],
                                 how='left',
                                 left_on=['previous_article'],
                                 right_on=['article'])

    negative_sampling = negative_sampling.drop(['article'], axis=1)
    negative_sampling.rename({'text': 'previous_article_text'}, axis=1, inplace=True)

    # Reorganizing columns to match combined_dataset
    negative_sampling = negative_sampling[['previous_article', 'previous_article_text', 'previous_article_links',
                                           'current_article', 'current_article_text', 'current_article_links']]

    negative_sampling.insert(len(negative_sampling.columns), 'number_of_clicks', 0)
    negative_sampling.insert(len(negative_sampling.columns), 'click_rate', 0)

    combined_dataset = combined_dataset.append(negative_sampling, ignore_index=True)

    print(len(combined_dataset))

    return combined_dataset


def get_negative_sampling_data(non_visited_links_data, previous_article_data, columns_names):
    # Hack to perform cross-join with two dataframes. See https://github.com/pandas-dev/pandas/issues/5401
    non_visited_links_data.insert(len(non_visited_links_data.columns), 'key', 1)
    previous_article_data.insert(len(previous_article_data.columns), 'key', 1)
    negative_sampling = pd.merge(non_visited_links_data, previous_article_data, on='key').drop('key', axis=1)

    negative_sampling.insert(len(negative_sampling.columns), 'number_of_clicks', 0)
    negative_sampling.insert(len(negative_sampling.columns), 'click_rate', 0)

    negative_sampling.columns = columns_names

    return negative_sampling


def combine_wiki_click_stream_datasets(click_stream_dataset, wiki_documents_dataset):
    combined_dataset = pd.merge(click_stream_dataset,
                                wiki_documents_dataset,
                                left_on=['previous_article'],
                                right_on=['article'])

    combined_dataset = pd.merge(combined_dataset,
                                wiki_documents_dataset,
                                left_on=['current_article'],
                                right_on=['article'])

    combined_dataset = combined_dataset.drop(columns=['article_x', 'article_y'])

    combined_dataset.columns = ['previous_article', 'current_article', 'number_of_clicks', 'click_rate',
                                'previous_article_text', 'previous_article_links', 'current_article_text',
                                'current_article_links']

    # Rearranging columns
    combined_dataset = combined_dataset[
        ['previous_article', 'previous_article_text', 'previous_article_links', 'current_article',
         'current_article_text', 'current_article_links', 'number_of_clicks', 'click_rate']]

    return combined_dataset


def extract_documents_as_vectors_from_source(wiki_documents_path):
    texts, links, articles = [], [], []

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
            links.append(result['links'])

    wiki_documents_dataset = pd.DataFrame(list(zip(articles, texts, links)), columns=['article', 'text', 'links'])
    wiki_documents_dataset['article'] = remove_special_characters_df(wiki_documents_dataset['article'])

    return wiki_documents_dataset


def extract_documents_as_text_from_source(wiki_documents_path):
    texts, articles, links = [], [], []

    with open(wiki_documents_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        introduction_json = result['sections'][0]['text'].replace('\n', ' ').replace('\r', '')

        if introduction_json:
            texts.append(introduction_json)
            articles.append(result['title'])
            links.append(result['links'])

    wiki_documents_dataset = pd.DataFrame(list(zip(articles, texts, links)), columns=['article', 'text', 'links'])

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

    :return: pandas.DataFrame
    """
    print('Generating Wiki dataset')
    wiki_documents_dataset = extract_documents_as_vectors_from_source(wiki_documents_path)
    available_titles = wiki_documents_dataset['article'].drop_duplicates()
    # available_titles = None

    print('Generating click stream dataset')
    click_stream_dataset = generate_click_stream_dataset(click_stream_dump_path, available_titles)
    click_stream_dataset = add_metrics(click_stream_dataset, available_titles)

    print('Combining both datasets')
    combined_dataset = combine_wiki_click_stream_datasets(click_stream_dataset, wiki_documents_dataset)

    print('Adding negative sampling')
    combined_dataset = add_negative_sampling(combined_dataset, wiki_documents_dataset)

    print('Finished dataset creation')

    return combined_dataset


def generate_text_dataset(click_stream_dump_path, wiki_documents_path):
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
    wiki_documents_dataset = extract_documents_as_text_from_source(wiki_documents_path)

    combined_dataset = combine_wiki_click_stream_datasets(click_stream_dataset, wiki_documents_dataset)
    combined_dataset = add_negative_sampling(combined_dataset, wiki_documents_dataset)

    return combined_dataset


def split_dataset(dataset, train_split, batch_size, train_dataset_path='../data/training.pth',
                  validation_dataset_path='../data/validation.pth', test_dataset_path='../data/test.pth'):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

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


def get_dataset_sample(df, sample_size=0.1, destination_path='../data/wiki_df.csv'):
    random.seed(123)

    previous_articles = df['previous_article'].drop_duplicates().reset_index(drop=True)
    num_articles = len(previous_articles)

    sample_dataset_size = int(num_articles * sample_size)

    selected_indices = random.sample(range(num_articles), sample_dataset_size)

    selected_articles = previous_articles[previous_articles.index.isin(selected_indices)]

    dataset_sample = df[df['previous_article'].isin(selected_articles)].reset_index(drop=True)

    dataset_sample.to_csv(destination_path, index=False)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    click_stream_dump_path = '../data/clickstream-enwiki-2019-12.tsv'
    wiki_documents_path = '../data/simplewiki3.jsonl'

    # generate_click_stream_dataset(click_stream_dump_path, wiki_documents_path)

    # text_documents = generate_dataset(click_stream_dump_path, wiki_documents_path)
    # text_documents.to_csv('../data/wiki_df.csv',
    #                       index=False
    #                       )

    df = pd.read_csv('../data/wiki_df_complete.csv')
    get_dataset_sample(df, sample_size=0.05)
