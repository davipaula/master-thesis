import pandas as pd
from utils.utils import clean_title

CLICK_RATE_COLUMN = "click_rate"
NUMBER_OF_CLICKS_COLUMN = "number_of_clicks"
SOURCE_ARTICLE_COLUMN = "source_article"
TARGET_ARTICLE_COLUMN = "target_article"
TOTAL_CLICKS_COLUMN = "total_clicks"
TYPE_COLUMN = "type"


class ClickStreamExtractor:
    def __init__(self, click_stream_dump_path: str):
        self.__click_stream_dump_path = click_stream_dump_path

        # working_directory = os.getcwd()
        # save_path = os.path.join(working_directory, 'data', 'processed', 'click_stream.csv')
        self.__save_path = "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/click_stream.csv"

    def run(self):
        click_stream = self.__extract_click_stream_data()
        click_stream = self.__add_metrics(click_stream)

        click_stream.to_csv(self.__save_path, index=False)

        return click_stream

    def __extract_click_stream_data(self):
        click_stream_dataset = pd.read_csv(
            self.__click_stream_dump_path,
            sep="\t",
            quoting=3,
            header=None,
            names=[SOURCE_ARTICLE_COLUMN, TARGET_ARTICLE_COLUMN, TYPE_COLUMN, NUMBER_OF_CLICKS_COLUMN],
            dtype={
                SOURCE_ARTICLE_COLUMN: "unicode_",
                TARGET_ARTICLE_COLUMN: "unicode_",
                TYPE_COLUMN: "category",
                NUMBER_OF_CLICKS_COLUMN: "uint32",
            },
        )

        click_stream_dataset = click_stream_dataset[click_stream_dataset[TYPE_COLUMN] == "link"]
        click_stream_dataset = click_stream_dataset.drop([TYPE_COLUMN], axis=1)

        click_stream_dataset[SOURCE_ARTICLE_COLUMN] = click_stream_dataset[SOURCE_ARTICLE_COLUMN].map(clean_title)
        click_stream_dataset[TARGET_ARTICLE_COLUMN] = click_stream_dataset[TARGET_ARTICLE_COLUMN].map(clean_title)

        click_stream_dataset[CLICK_RATE_COLUMN] = 0

        click_stream_dataset.columns = [
            SOURCE_ARTICLE_COLUMN,
            TARGET_ARTICLE_COLUMN,
            NUMBER_OF_CLICKS_COLUMN,
            CLICK_RATE_COLUMN,
        ]

        return click_stream_dataset

    @staticmethod
    def __add_metrics(dataset):
        click_stream_dataset = dataset

        total_clicks = click_stream_dataset.groupby([SOURCE_ARTICLE_COLUMN]).agg({NUMBER_OF_CLICKS_COLUMN: "sum"})
        total_clicks.columns = [TOTAL_CLICKS_COLUMN]

        click_stream_dataset = click_stream_dataset.merge(total_clicks, on=SOURCE_ARTICLE_COLUMN)
        click_stream_dataset[CLICK_RATE_COLUMN] = round(
            click_stream_dataset[NUMBER_OF_CLICKS_COLUMN] / click_stream_dataset[TOTAL_CLICKS_COLUMN], 6
        )
        click_stream_dataset = click_stream_dataset.drop([TOTAL_CLICKS_COLUMN], axis=1)

        return click_stream_dataset


if __name__ == "__main__":
    cs = ClickStreamExtractor("../data/clickstream-enwiki-2020-12.tsv")
    cs.run()