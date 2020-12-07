from abc import ABC, abstractmethod
from pandas import DataFrame
from sklearn.model_selection import train_test_split


class Dataset(ABC):

    def __init__(self, dataframe: DataFrame = None):
        self._dataframe = dataframe
        self._features = []
        self._labels = []

    @abstractmethod
    def _load_dataset(self) -> DataFrame:
        pass

    def load_dataset(self):
        self._dataframe = self._load_dataset()

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, value):
        self._dataframe = value

    def get_training_test_sets(self):
        return train_test_split(self._dataframe[self._features], self._dataframe[self._labels])


