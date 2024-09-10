import pandas as pd
import numpy as np

from utils.preprocessing import Preprocessing, create_and_prepare_bag_of_words, adapt_test_to_training_data, \
    encode_labels


class RealEstatePreprocessing(Preprocessing):
    """
    Represents all methods and variables that are needed for the preprocessing of the real estate input dataframe.
    """

    def __init__(self, df, target_col, title_col_sum_threshold=200, description_col_sum_threshold=1000,
                 train_val_columns=None):
        """
        :param df: input dataframe with mental health data
        :param target_col: column to predict
        :param title_col_sum_threshold: filter threshold wrt. title column sum
        :param description_col_sum_threshold: filter threshold wrt. description column sum
        :param train_val_columns: columns of dataframe that was used for modeling including bag of words columns
        """
        super().__init__(df, target_col)
        self.columns_to_drop = [self.target_col, "title", "displayAddress", "type", "priceDuration",
                                "addedOn", "description"]
        self.df_columns = self.df.columns
        self.title_col_sum_threshold = title_col_sum_threshold
        self.description_col_sum_threshold = description_col_sum_threshold
        # Todo: check replacing fillna with imputation method
        self.imputation_cols = ['bathrooms', 'bedrooms', 'furnishing']
        self.normalization_columns = ["price", "sizeMin"]
        self.bow_transformation = False
        self.train_val_columns = train_val_columns

    def prepare_bathrooms(self):
        """
        Prepare bathrooms column
        :return:
        """
        self.df['bathrooms'] = self.df['bathrooms'].str.replace('7+', '7')
        self.df['bathrooms'] = self.df['bathrooms'].str.replace('none', '1')
        self.df['bathrooms'] = pd.to_numeric(self.df['bathrooms'])
        self.df['bathrooms'] = self.df['bathrooms'].fillna(1)
        self.df['bathrooms'] = self.df['bathrooms'].astype(np.int64)

    def prepare_bedrooms(self):
        """
        Prepare bedrooms column
        :return:
        """
        self.df['bedrooms'] = self.df['bedrooms'].str.replace('7+', '7')
        self.df['bedrooms'] = self.df['bedrooms'].str.replace('studio', '8')
        self.df['bedrooms'] = pd.to_numeric(self.df['bedrooms'])
        self.df['bedrooms'] = self.df['bedrooms'].fillna(1)
        self.df['bedrooms'] = self.df['bedrooms'].astype(np.int64)

    def prepare_size_min(self):
        """
        Prepare sizeMin column
        :return:
        """
        self.df['sizeMin'] = self.df['sizeMin'].str.replace(' sqft', '')
        self.df['sizeMin'] = self.df['sizeMin'].astype(np.int64)

    def prepare_furnishing(self):
        """
        Prepare furnishing column
        :return:
        """
        self.df["furnishing"] = encode_labels(series=self.df["furnishing"])

    def prepare_bathroom_per_bedrooms(self):
        """
        Add new column bathroom per bedrooms
        see https://www.kaggle.com/code/bhavikrohit/uae-real-estate-market-analysis-2024
        :return:
        """
        self.df["bathroom_per_bedrooms"] = self.df['bedrooms'] / self.df['bathrooms']

    def prepare_price_per_sqft(self):
        """
        Add new column price per sqft
        see https://www.kaggle.com/code/bhavikrohit/uae-real-estate-market-analysis-2024
        :return:
        """
        self.df['price_per_sqft'] = self.df['price'] / (self.df['sizeMin'])

    def start(self):
        """
        Start real estate preprocessing
        :return:
        """
        self.prepare_bathrooms()
        self.prepare_bedrooms()
        self.prepare_size_min()
        self.prepare_furnishing()
        self.prepare_bathroom_per_bedrooms()
        self.prepare_price_per_sqft()

        if self.bow_transformation:
            title_bag_of_words_prepared = (
                create_and_prepare_bag_of_words(series=self.df["title"],
                                                col_sum_threshold=self.title_col_sum_threshold,
                                                columns=self.df.columns.to_list(),
                                                postfix="title"))
            self.df = pd.concat([self.df, title_bag_of_words_prepared], axis=1)

            description_bag_of_words_prepared = (
                create_and_prepare_bag_of_words(series=self.df["description"],
                                                col_sum_threshold=self.description_col_sum_threshold,
                                                columns=self.df.columns.to_list(),
                                                postfix="description"))
            self.df = pd.concat([self.df, description_bag_of_words_prepared], axis=1)

        # self.df[self.normalization_columns] = StandardScaler().fit_transform(self.df[self.normalization_columns])
        x = self.df.drop(self.columns_to_drop, axis=1)
        y = self.df[self.target_col]

        if self.train_val_columns:
            x = adapt_test_to_training_data(test_df=x, train_val_columns=self.train_val_columns)
        else:
            x = x.reindex(sorted(x.columns), axis=1)
            # set default value for missing columns and Nan values
            x = x.fillna(0)

        return x, y
