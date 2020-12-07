from pandas import DataFrame
import pandas as pd

from common.dataset import Dataset


class WineReviewsDataset(Dataset):

    def __init__(self, base_data_path="wine_reviews/data"):
        super(Dataset, self).__init__()
        self._base_data_path = base_data_path

    def _load_dataset(self) -> DataFrame:
        return pd.read_csv(f'{self._base_data_path}/winemag-data_first150k.csv')