import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing import Preprocessing, prepare_text_with_regex, create_bag_of_words, filter_bag_of_words, \
    create_and_prepare_bag_of_words


class MentalHealthPreprocessing(Preprocessing):
    """
    Represents all methods and variables that are needed for the preprocessing of the mental health input dataframes.
    """
    def __init__(self, df, target_col, col_sum_threshold, train_val_columns=None):
        """
        :param df: input dataframe with mental health data
        :param train_val_columns: columns of dataframe that was used for modeling including bag of words columns
        """
        super().__init__(df=df, target_col=target_col)
        self.train_val_columns = train_val_columns
        # number of times word have to appear in statements
        self.col_sum_threshold = col_sum_threshold
        # irrelevant columns for modeling
        self.columns_to_drop = [self.target_col, "statement"]

    def adapt_test_to_training_data(self, test_df):
        """
        Adds missing training columns to test dataframe and removes columns that are missing in training dataframe.
        :param test_df: x data of test dataframe
        :return: adapted x data of test dataframe to training data
        """
        test_columns_set = set(test_df.columns.tolist())
        train_val_columns_set = set(self.train_val_columns)
        missing_train_val_columns = list(train_val_columns_set - test_columns_set)
        missing_test_columns = list(test_columns_set - train_val_columns_set)
        if missing_train_val_columns:
            # create missing columns
            df_missing_cols = pd.DataFrame(columns=missing_train_val_columns)
            test_df = pd.concat([test_df, df_missing_cols], axis=1)
        if missing_test_columns:
            # drop columns that were not present in the train bag of words matrix
            test_df = test_df.drop(missing_test_columns, axis=1)

        test_df = test_df.reindex(sorted(test_df.columns), axis=1)
        # set default value for missing columns and Nan values
        test_df = test_df.fillna(0)

        return test_df

    def start(self):
        """
        Start preprocessing of mental health data
        :return: preprocessed feature dataframe x and target column y
        """
        bag_of_words_prepared = create_and_prepare_bag_of_words(series=self.df["statement"],
                                                                col_sum_threshold=self.col_sum_threshold)
        preprocessed_df = pd.concat([self.df, bag_of_words_prepared], axis=1)
        y = preprocessed_df[self.target_col]
        # encode mental health status
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        x = preprocessed_df.drop(self.columns_to_drop, axis=1)
        # prepare bag of words matrix for test set
        if self.train_val_columns:
            x = self.adapt_test_to_training_data(test_df=x)
        else:
            x = x.reindex(sorted(x.columns), axis=1)
            # set default value for missing columns and Nan values
            x = x.fillna(0)

        return x, y
