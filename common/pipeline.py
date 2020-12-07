from common.dataset import Dataset
from common.model import Model
from common.preprocessor import Preprocessor
import numpy as np


class Pipeline:

    def __init__(self, dataset: Dataset, preprocessor: Preprocessor, model: Model, seed: int):
        self._dataset = dataset
        self._preprocessor = preprocessor
        self._model = model
        np.random.seed(seed)

    def run(self):
        self._dataset.load_dataset()

        self._preprocessor.apply(self._dataset)

        self._model.fit(dataset=self._dataset)

        self._model.evaluate(dataset=self._dataset)
