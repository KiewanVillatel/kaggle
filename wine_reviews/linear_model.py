from sklearn.linear_model import LinearRegression

from common.dataset import Dataset
from common.model import Model


class LinearModel(Model):

    def __init__(self):
        self._model = LinearRegression(normalize=True)

    def fit(self, dataset: Dataset):
        X_train, _, y_train, _ = dataset.get_training_test_sets()
        self._model.fit(X_train, y_train)
        print(self._model.score(X_train, y_train))

    def evaluate(self, dataset: Dataset):
        _, X_test, _, y_test = dataset.get_training_test_sets()
        print(self._model.score(X_test, y_test))