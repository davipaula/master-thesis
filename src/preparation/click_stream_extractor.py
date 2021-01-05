import pandas as pd

from ..utils.constants import (
    CLICK_RATE_COLUMN,
    CLICK_STREAM_DUMP_PATH,
    CLICK_STREAM_PROCESSED_PATH,
    NUMBER_OF_CLICKS_COLUMN,
    SOURCE_ARTICLE_COLUMN,
    TARGET_ARTICLE_COLUMN,
    TOTAL_CLICKS_COLUMN,
    TYPE_COLUMN,
)
from ..utils.utils import clean_title


def extract_click_stream_data():
    click_stream = _process_raw_click_stream()
    click_stream = _add_metrics(click_stream)

    click_stream.to_csv(CLICK_STREAM_PROCESSED_PATH, index=False)


def _add_metrics(dataset: pd.DataFrame) -> pd.DataFrame:
    """Adds the metric `click_through_rate` to the clickstream dataset

    Parameters
    ----------
    dataset :

    Returns
    -------
    pandas.core.frame.DataFrame

    """
    click_stream_dataset = dataset

    total_clicks = click_stream_dataset.groupby([SOURCE_ARTICLE_COLUMN]).agg(
        {NUMBER_OF_CLICKS_COLUMN: "sum"}
    )
    total_clicks.columns = [TOTAL_CLICKS_COLUMN]

    click_stream_dataset = click_stream_dataset.merge(
        total_clicks, on=SOURCE_ARTICLE_COLUMN
    )
    click_stream_dataset[CLICK_RATE_COLUMN] = round(
        click_stream_dataset[NUMBER_OF_CLICKS_COLUMN]
        / click_stream_dataset[TOTAL_CLICKS_COLUMN],
        6,
    )
    click_stream_dataset = click_stream_dataset.drop([TOTAL_CLICKS_COLUMN], axis=1)

    return click_stream_dataset


def _process_raw_click_stream() -> pd.DataFrame:
    """Returns a dataframe with the Click Stream Data

    Returns
    -------
    pandas.core.frame.DataFrame

    """
    click_stream_dataset = pd.read_csv(
        CLICK_STREAM_DUMP_PATH,
        sep="\t",
        quoting=3,
        header=None,
        names=[
            SOURCE_ARTICLE_COLUMN,
            TARGET_ARTICLE_COLUMN,
            TYPE_COLUMN,
            NUMBER_OF_CLICKS_COLUMN,
        ],
        dtype={
            SOURCE_ARTICLE_COLUMN: "unicode_",
            TARGET_ARTICLE_COLUMN: "unicode_",
            TYPE_COLUMN: "category",
            NUMBER_OF_CLICKS_COLUMN: "uint32",
        },
    )

    click_stream_dataset = click_stream_dataset[
        click_stream_dataset[TYPE_COLUMN] == "link"
    ]
    click_stream_dataset = click_stream_dataset.drop([TYPE_COLUMN], axis=1)

    click_stream_dataset[SOURCE_ARTICLE_COLUMN] = click_stream_dataset[
        SOURCE_ARTICLE_COLUMN
    ].map(clean_title)
    click_stream_dataset[TARGET_ARTICLE_COLUMN] = click_stream_dataset[
        TARGET_ARTICLE_COLUMN
    ].map(clean_title)

    click_stream_dataset[CLICK_RATE_COLUMN] = 0

    click_stream_dataset.columns = [
        SOURCE_ARTICLE_COLUMN,
        TARGET_ARTICLE_COLUMN,
        NUMBER_OF_CLICKS_COLUMN,
        CLICK_RATE_COLUMN,
    ]

    return click_stream_dataset
