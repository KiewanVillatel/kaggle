from pandas import DataFrame
from typing import Tuple, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def one_hot(df: DataFrame,
            column: str,
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


def tf_idf(df: DataFrame,
           column: str,
           normalize_column_name: bool = True,
           min_df: float = 0,
           max_df: float = 1.0
           ) -> Tuple[DataFrame, List]:
    v = TfidfVectorizer(min_df=min_df, max_df=max_df)
    x = v.fit_transform(df[column])

    tfidf_df = pd.DataFrame(x.toarray(), columns=v.get_feature_names())

    if normalize_column_name:
        tfidf_df = normalize_column_names(tfidf_df)

    tfidf_df.columns = [f"{column}_{c}" for c in tfidf_df.columns]

    df = pd.concat([df, tfidf_df], axis=1)

    return df, tfidf_df.columns.to_list()


def normalize_column_names(df: DataFrame) -> DataFrame:
    columns_mapping = {c: c.lower().replace(" ", "_").replace("-", "_") for c in df.columns}
    df.rename(columns=columns_mapping, inplace=True)
    return df
