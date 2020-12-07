from abc import ABC, abstractmethod
from .dataset import Dataset
from typing import Tuple, List
from pandas import DataFrame


class Preprocessor(ABC):

    @abstractmethod
    def _apply(self, dataset: Dataset) -> Tuple[DataFrame, List[str], List[str]]:
        pass

    def apply(self, dataset: Dataset) -> Dataset:
        df, features, target = self._apply(dataset)

        dataset.features = features
        dataset.labels = target
        dataset.dataframe = df

        return dataset
