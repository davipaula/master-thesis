import pandas as pd

from utils import clean_title


class ClickStreamProcessor:
    def __init__(self, click_stream_dump_path: str):
        self.__click_stream_dump_path = click_stream_dump_path

        # working_directory = os.getcwd()
        # save_path = os.path.join(working_directory, 'data', 'processed', 'click_stream.csv')
        self.__save_path = '/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/click_stream.csv'

        # if os.path.exists(save_path):
        #     self.__click_stream = pd.read_csv(save_path)
        # else:
        #     self.__click_stream = self.create_dataset(save_path)

    def create_dataset(self):
        click_stream = self.__extract_click_stream_data()
        click_stream = self.__add_metrics(click_stream)

        click_stream.to_csv(self.__save_path, index=False)

        return click_stream

    def __extract_click_stream_data(self):
        click_stream_dataset = pd.read_csv(self.__click_stream_dump_path,
                                           sep='\t',
                                           quoting=3,
                                           header=None,
                                           names=['source_article', 'target_article', 'type', 'n'],
                                           dtype={'source_article': 'unicode_',
                                                  'target_article': 'unicode_',
                                                  'type': 'category',
                                                  'n': 'uint32'}
                                           )

        click_stream_dataset = click_stream_dataset[click_stream_dataset['type'] == 'link']
        click_stream_dataset = click_stream_dataset.drop(['type'], axis=1)

        click_stream_dataset['source_article'] = click_stream_dataset['source_article'].map(clean_title)
        click_stream_dataset['target_article'] = click_stream_dataset['target_article'].map(clean_title)

        click_stream_dataset['click_rate'] = 0

        click_stream_dataset.columns = ['source_article', 'target_article', 'number_of_clicks', 'click_rate']

        return click_stream_dataset

    def __add_metrics(self, dataset):
        click_stream_dataset = dataset

        total_clicks = click_stream_dataset.groupby(['source_article']).agg({'number_of_clicks': 'sum'})
        total_clicks.columns = ['total_clicks']

        click_stream_dataset = click_stream_dataset.merge(total_clicks, on='source_article')
        click_stream_dataset['click_rate'] = round(
            click_stream_dataset['number_of_clicks'] / click_stream_dataset['total_clicks'], 6)
        click_stream_dataset = click_stream_dataset.drop(['total_clicks'], axis=1)

        return click_stream_dataset


if __name__ == '__main__':
    cs = ClickStreamProcessor('../data/clickstream-enwiki-2019-12.tsv')
    cs.create_dataset()
