import pandas as pd
from torch.utils.data.dataset import Dataset


class ClickStreamDataset(Dataset):
    def __init__(self, dataset):
        super(ClickStreamDataset, self).__init__()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return_item = {
            "source_article": self.dataset["source_article"].iloc[index],
            "target_article": self.dataset["target_article"].iloc[index],
            "click_rate": self.dataset["click_rate"].iloc[index],
        }
        return return_item

    def get_titles(self):
        titles = pd.Series.append(self.dataset["source_article"], self.dataset["target_article"])
        return titles.unique()
