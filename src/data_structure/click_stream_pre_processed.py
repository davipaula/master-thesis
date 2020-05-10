import pandas as pd
from torch.utils.data.dataset import Dataset


class ClickStreamPreProcessed(Dataset):
    def __init__(self):
        super(ClickStreamPreProcessed, self).__init__()

        dataset_path = "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/click_stream.csv"
        self.dataset = pd.read_csv(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index]

    def get_titles(self):
        titles = pd.Series.append(self.dataset["source_article"], self.dataset["target_article"])
        return titles.unique()


if __name__ == "__main__":
    data = ClickStreamPreProcessed()
    print(data[10])
