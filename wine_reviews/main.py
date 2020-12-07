import click

from common.pipeline import Pipeline
from wine_reviews.wine_rewiews_dataset import WineReviewsDataset
from wine_reviews.wine_reviews_preprocessor import WineReviewsPreprocessor
from wine_reviews.linear_model import LinearModel


@click.command()
@click.option("--seed", default=0, help="Seed for the experiment")
@click.option("--normalize", default=True)
def main(seed, normalize):
    dataset = WineReviewsDataset()

    preprocessor = WineReviewsPreprocessor()

    pipeline = Pipeline(dataset=dataset, preprocessor=preprocessor, model=LinearModel(normalize), seed=seed)

    pipeline.run()


if __name__ == "__main__":
    main()
