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
        print("Loading dataset")
        self._dataset.load_dataset()
        print(f"Dataset loaded. \n"
              f"Nb rows: {len(self._dataset.dataframe.index)} \n"
              f"Nb columns: {len(self._dataset.dataframe.columns)} \n")

        print("Preprocessing")
        self._preprocessor.apply(self._dataset)
        print(f"Preprocessing done."
              f"Nb features: {len(self._dataset.features)} \n" 
              f"Nb labels: {len(self._dataset.labels)} \n")

        print("Training model")
        self._model.fit(dataset=self._dataset)

        print("Evaluating model")
        self._model.evaluate(dataset=self._dataset)
