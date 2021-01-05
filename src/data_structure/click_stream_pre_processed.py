import pandas as pd
from torch.utils.data.dataset import Dataset

from ..utils.constants import (
    CLICK_RATE_COLUMN,
    CLICK_STREAM_PROCESSED_PATH,
    NUMBER_OF_CLICKS_COLUMN,
    SOURCE_ARTICLE_COLUMN,
    TARGET_ARTICLE_COLUMN,
)


class ClickStreamPreProcessed(Dataset):
    def __init__(self):
        super(ClickStreamPreProcessed, self).__init__()

        self.dataset = pd.read_csv(
            CLICK_STREAM_PROCESSED_PATH,
            dtype={
                SOURCE_ARTICLE_COLUMN: "unicode_",
                TARGET_ARTICLE_COLUMN: "unicode_",
                NUMBER_OF_CLICKS_COLUMN: "uint32",
                CLICK_RATE_COLUMN: "float",
            },
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index]
