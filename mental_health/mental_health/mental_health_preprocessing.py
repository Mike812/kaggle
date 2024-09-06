import pandas as pd

from utils.preprocessing import Preprocessing, create_and_prepare_bag_of_words, adapt_test_to_training_data, \
    encode_labels


class MentalHealthPreprocessing(Preprocessing):
    """
    Represents all methods and variables that are needed for the preprocessing of the mental health input dataframes.
    """
    def __init__(self, df, target_col, col_sum_threshold=150, train_val_columns=None):
        """
        :param df: input dataframe with mental health data
        :param train_val_columns: columns of dataframe that was used for modeling including bag of words columns
        """
        super().__init__(df=df, target_col=target_col)
        self.train_val_columns = train_val_columns
        # col sums: number of times word have to appear in statements
        self.col_sum_threshold = col_sum_threshold
        # irrelevant columns for modeling
        self.columns_to_drop = [self.target_col, "statement"]

    def start(self):
        """
        Start preprocessing of mental health data
        :return: preprocessed feature dataframe x and target column y
        """
        bag_of_words_prepared = create_and_prepare_bag_of_words(series=self.df["statement"],
                                                                col_sum_threshold=self.col_sum_threshold,
                                                                columns=self.df.columns.to_list(),
                                                                postfix="statement")
        preprocessed_df = pd.concat([self.df, bag_of_words_prepared], axis=1)
        # encode mental health status
        y = encode_labels(series=preprocessed_df[self.target_col])
        x = preprocessed_df.drop(self.columns_to_drop, axis=1)
        # prepare bag of words matrix for test set
        if self.train_val_columns:
            x = adapt_test_to_training_data(test_df=x, train_val_columns=self.train_val_columns)
        else:
            x = x.reindex(sorted(x.columns), axis=1)
            # set default value for missing columns and Nan values
            x = x.fillna(0)

        return x, y
