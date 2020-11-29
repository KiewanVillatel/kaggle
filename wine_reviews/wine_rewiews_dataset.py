from pandas import DataFrame
import pandas as pd

from common.dataset import Dataset


class WineReviewsDataset(Dataset):
    def _load_dataset(self) -> DataFrame:
        return pd.read_csv('wine_reviews/data/winemag-data_first150k.csv')