import mlflow
from sklearn.linear_model import LinearRegression

from common.dataset import Dataset
from common.model import Model


class LinearModel(Model):

    def __init__(self, normalize):
        self._model = LinearRegression(normalize=normalize)

    def _test(self, x, y, prefix):
        score = self._model.score(x, y)
        mlflow.log_metric("{}_score".format(prefix), score)

    def fit(self, dataset: Dataset):
        x_train, _, y_train, _ = dataset.get_training_test_sets()
        self._model.fit(x_train, y_train)
        self._test(x_train, y_train, "train")
        mlflow.sklearn.log_model(self._model, "linear_model")

    def evaluate(self, dataset: Dataset):
        _, x_test, _, y_test = dataset.get_training_test_sets()
        self._test(x_test, y_test, "test")
        print(self._model.score(x_test, y_test))
