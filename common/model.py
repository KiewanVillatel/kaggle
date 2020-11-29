from abc import ABC, abstractmethod
from .dataset import Dataset


class Model(ABC):

    @abstractmethod
    def fit(self, dataset: Dataset):
        pass

    @abstractmethod
    def evaluate(self, dataset: Dataset):
        pass