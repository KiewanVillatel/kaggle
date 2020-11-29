from common.pipeline import Pipeline
from wine_reviews.wine_rewiews_dataset import WineReviewsDataset
from wine_reviews.wine_reviews_preprocessor import WineReviewsPreprocessor
from wine_reviews.linear_model import LinearModel

dataset = WineReviewsDataset()

preprocessor = WineReviewsPreprocessor()

pipeline = Pipeline(dataset=dataset, preprocessor=preprocessor, model=LinearModel())

pipeline.run()