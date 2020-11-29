from typing import Tuple, List

from pandas import DataFrame

from common.dataset import Dataset
from common.preprocessor import Preprocessor
from common.utils import preprocessing_helper


class WineReviewsPreprocessor(Preprocessor):

    def _apply(self, dataset: Dataset) -> Tuple[DataFrame, List[str], List[str]]:
        min_count = 1000
        df = dataset.dataframe

        df, province_columns = preprocessing_helper.one_hot(df, 'province', min_count=min_count)
        # df, province_columns = preprocessing_helper.one_hot(df, 'designation', min_count=min_count)
        # df, variety_columns = preprocessing_helper.one_hot(df, 'variety', min_count=min_count)
        # df, region_1_columns = preprocessing_helper.one_hot(df, 'region_1', min_count=min_count)
        # df, region_2_columns = preprocessing_helper.one_hot(df, 'region_2', min_count=min_count)
        # df, region_2_columns = preprocessing_helper.one_hot(df, 'winery', min_count=min_count)
        df = preprocessing_helper.normalize_column_names(df)
        df['price'] = df['price'].fillna(0)

        features = province_columns + ['price']

        return df, features, ['points']
