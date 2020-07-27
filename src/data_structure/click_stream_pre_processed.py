import pandas as pd
from torch.utils.data.dataset import Dataset

from utils.constants import CLICK_STREAM_PROCESSED_PATH

TARGET_ARTICLE_COLUMN = "target_article"
SOURCE_ARTICLE_COLUMN = "source_article"
NUMBER_OF_CLICKS_COLUMN = "number_of_clicks"
CLICK_RATE_COLUMN = "click_rate"


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

    def get_titles(self):
        titles = pd.Series.append(self.dataset[SOURCE_ARTICLE_COLUMN], self.dataset[TARGET_ARTICLE_COLUMN])
        return titles.unique()


if __name__ == "__main__":
    data = ClickStreamPreProcessed()
    print(data[10])
