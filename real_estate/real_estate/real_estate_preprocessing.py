import pandas as pd
import numpy as np

from utils.preprocessing import Preprocessing, create_bag_of_words, filter_bag_of_words, create_and_prepare_bag_of_words


def prepare_bathrooms(df):
    """

    :param df:
    :return:
    """
    df['bathrooms'] = df['bathrooms'].str.replace('7+', '8')
    df['bathrooms'] = df['bathrooms'].str.replace('none', '0')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'])
    df['bathrooms'] = df['bathrooms'].fillna(0)
    df['bathrooms'] = df['bathrooms'].astype(np.int64)

    return df


def prepare_bedrooms(df):
    """

    :param df:
    :return:
    """
    df['bedrooms'] = df['bedrooms'].str.replace('7+', '8')
    df['bedrooms'] = df['bedrooms'].str.replace('studio', '9')
    df['bedrooms'] = pd.to_numeric(df['bedrooms'])
    df['bedrooms'] = df['bedrooms'].fillna(0)
    df['bedrooms'] = df['bedrooms'].astype(np.int64)

    return df


def prepare_size_min(df):
    """

    :param df:
    :return:
    """
    df['sizeMin'] = df['sizeMin'].str.replace(' sqft', '')
    df['sizeMin'] = df['sizeMin'].astype(np.int64)

    return df


class RealEstatePreprocessing(Preprocessing):
    """

    """

    def __init__(self, df, target_col):
        """

        :param df:
        :param target_col:
        """
        super().__init__(df, target_col)

    def start(self):
        """

        :return:
        """
        self.df = prepare_bathrooms(df=self.df)
        self.df = prepare_bedrooms(df=self.df)
        self.df = prepare_size_min(df=self.df)

        title_bag_of_words_prepared = create_and_prepare_bag_of_words(series=self.df["title"],
                                                                      col_sum_threshold=50)
        description_bag_of_words_prepared = create_and_prepare_bag_of_words(series=self.df["description"],
                                                                            col_sum_threshold=50)
        preprocessed_df = pd.concat([self.df, title_bag_of_words_prepared, description_bag_of_words_prepared],
                                    axis=1)
        y = preprocessed_df[self.target_col]
        x = preprocessed_df.drop(self.target_col, axis=1)

        return x, y
