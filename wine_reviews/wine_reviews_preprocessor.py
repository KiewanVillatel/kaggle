from typing import Tuple, List

from pandas import DataFrame

from common.dataset import Dataset
from common.preprocessor import Preprocessor
from common.utils import preprocessing_helper


class WineReviewsPreprocessor(Preprocessor):

    def __init__(self, min_province, min_designation, min_variety, min_region_1, min_region_2, min_winery):
        self._min_province = min_province
        self._min_designation = min_designation
        self._min_variety = min_variety
        self._min_region_1 = min_region_1
        self._min_region_2 = min_region_2
        self._min_winery = min_winery

    def _apply(self, dataset: Dataset) -> Tuple[DataFrame, List[str], List[str]]:
        df = dataset.dataframe

        df, province_columns = preprocessing_helper.one_hot(df, 'province', min_count=self._min_province)
        df, designation_columns = preprocessing_helper.one_hot(df, 'designation', min_count=self._min_designation)
        df, variety_columns = preprocessing_helper.one_hot(df, 'variety', min_count=self._min_variety)
        df, region_1_columns = preprocessing_helper.one_hot(df, 'region_1', min_count=self._min_region_1)
        df, region_2_columns = preprocessing_helper.one_hot(df, 'region_2', min_count=self._min_region_2)
        df, winery_columns = preprocessing_helper.one_hot(df, 'winery', min_count=self._min_winery)
        df = preprocessing_helper.normalize_column_names(df)
        df['price'] = df['price'].fillna(0)

        features = province_columns + designation_columns + variety_columns + region_1_columns + region_2_columns + winery_columns + ['price']

        return df, features, ['points']
