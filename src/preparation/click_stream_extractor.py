import pandas as pd
from utils.utils import clean_title
from utils.constants import CLICK_STREAM_DUMP_PATH, CLICK_STREAM_PROCESSED_PATH

CLICK_RATE_COLUMN = "click_rate"
NUMBER_OF_CLICKS_COLUMN = "number_of_clicks"
SOURCE_ARTICLE_COLUMN = "source_article"
TARGET_ARTICLE_COLUMN = "target_article"
TOTAL_CLICKS_COLUMN = "total_clicks"
TYPE_COLUMN = "type"


class ClickStreamExtractor:
    def run(self):
        click_stream = self.__extract_click_stream_data()
        click_stream = self.__add_metrics(click_stream)

        click_stream.to_csv(CLICK_STREAM_PROCESSED_PATH, index=False)

        return click_stream

    @staticmethod
    def __extract_click_stream_data():
        click_stream_dataset = pd.read_csv(
            CLICK_STREAM_DUMP_PATH,
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
    cs = ClickStreamExtractor()
    cs.run()
