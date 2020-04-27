import pandas as pd
from torch.utils.data.dataset import Dataset


class ClickStreamPreProcessed(Dataset):
    def __init__(self, dataset_path: str):
        super(ClickStreamPreProcessed, self).__init__()

        self.dataset = pd.read_csv(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index]


if __name__ == "__main__":
    save_path = "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/click_stream.csv"
    data = ClickStreamPreProcessed(save_path)
    print(data[10])
