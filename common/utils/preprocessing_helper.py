from pandas import DataFrame
from typing import Tuple, List
import pandas as pd


def one_hot(df: DataFrame, column: str,
            normalize_column_name: bool = True,
            min_count: int = 0,
            drop_original_column: bool = False) -> Tuple[DataFrame, List]:

    df[column] = df[column].fillna("NA")

    if min_count > 0:
        counts = df[column].to_frame().reset_index().groupby(column).count().add_suffix('_count').reset_index()
        df = pd.merge(df, counts, left_on=column, right_on=column)
        df.loc[df["index_count"] < min_count, column] = "dummy_value"
        df = df.drop('index_count', axis=1)

    dummies = pd.get_dummies(df[column], prefix=column)

    if normalize_column_name:
        dummies = normalize_column_names(dummies)

    if drop_original_column:
        df = df.drop(column, axis=1)

    return df.join(dummies), dummies.columns.tolist()


def normalize_column_names(df: DataFrame) -> DataFrame:
    columns_mapping = {c: c.lower().replace(" ", "_").replace("-", "_") for c in df.columns}
    df.rename(columns=columns_mapping, inplace=True)
    return df
