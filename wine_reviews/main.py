import click

from common.pipeline import Pipeline
from wine_reviews.wine_rewiews_dataset import WineReviewsDataset
from wine_reviews.wine_reviews_preprocessor import WineReviewsPreprocessor
from wine_reviews.linear_model import LinearModel


@click.command()
@click.option("--seed", type=int, help="Seed for the experiment")
@click.option("--normalize", type=bool)
@click.option("--min_province", type=int)
@click.option("--min_designation", type=int)
@click.option("--min_variety", type=int)
@click.option("--min_region_1", type=int)
@click.option("--min_region_2", type=int)
@click.option("--min_winery", type=int)
def main(seed, normalize, min_province, min_designation, min_variety, min_region_1, min_region_2, min_winery):
    dataset = WineReviewsDataset()

    preprocessor = WineReviewsPreprocessor(min_province=min_province,
                                           min_designation=min_designation,
                                           min_variety=min_variety,
                                           min_region_1=min_region_1,
                                           min_region_2=min_region_2,
                                           min_winery=min_winery)

    pipeline = Pipeline(dataset=dataset, preprocessor=preprocessor, model=LinearModel(normalize), seed=seed)

    pipeline.run()


if __name__ == "__main__":
    main()
