import pandas as pd
import numpy as np

from utils.preprocessing import Preprocessing, create_and_prepare_bag_of_words, adapt_test_to_training_data


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

    def __init__(self, df, target_col, train_val_columns=None):
        """

        :param df:
        :param target_col:
        """
        super().__init__(df, target_col)
        self.columns_to_drop = [self.target_col, "title", "displayAddress", "type", "priceDuration", "index",
                                "addedOn", "furnishing", "description"]
        self.df_columns = self.df.columns
        self.train_val_columns = train_val_columns
        self.title_col_sum_threshold = 100
        self.description_col_sum_threshold = 200

    def start(self):
        """

        :return:
        """
        self.df = prepare_bathrooms(df=self.df)
        self.df = prepare_bedrooms(df=self.df)
        self.df = prepare_size_min(df=self.df)

        df_columns = self.df.columns.to_list()
        title_bag_of_words_prepared = (
            create_and_prepare_bag_of_words(series=self.df["title"],
                                            col_sum_threshold=self.title_col_sum_threshold,
                                            df_columns=df_columns,
                                            postfix="title"))
        preprocessed_df = pd.concat([self.df, title_bag_of_words_prepared], axis=1)

        description_bag_of_words_prepared = (
            create_and_prepare_bag_of_words(series=self.df["description"],
                                            col_sum_threshold=self.description_col_sum_threshold,
                                            df_columns=preprocessed_df.columns.to_list(),
                                            postfix="description"))
        preprocessed_df = pd.concat([preprocessed_df, description_bag_of_words_prepared],
                                    axis=1)

        x = preprocessed_df.drop(self.columns_to_drop, axis=1).fillna(0)
        y = self.df[self.target_col]

        if self.train_val_columns:
            x = adapt_test_to_training_data(test_df=x, train_val_columns=self.train_val_columns)
        else:
            x = x.reindex(sorted(x.columns), axis=1)
            # set default value for missing columns and Nan values
            x = x.fillna(0)

        return x, y
