from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from stop_words import get_stop_words

from utils.preprocessing import Preprocessing, prepare_text_with_regex


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
        # have to contain characters
        self.filter_regex = '[A-Za-z]'
        # irrelevant columns for modeling
        self.columns_to_drop = [self.target_col, "statement"]

    def create_bag_of_words(self):
        """
        :return: bag of words dataframe
        """
        statements = self.df["statement"].fillna("").tolist()
        stop_words = get_stop_words("en")
        count_vec = CountVectorizer(stop_words=stop_words)
        count_vec_fitted = count_vec.fit_transform(statements)
        word_counts = count_vec_fitted.toarray()
        words = count_vec.get_feature_names_out()
        bag_of_words = pd.DataFrame(data=word_counts, columns=words)

        return bag_of_words

    def filter_bag_of_words(self, bag_of_words):
        """
        Filter dataframe by colsums and regex
        :param bag_of_words: dataframe
        :return: filtered bag_of_words dataframe
        """
        bag_of_words = bag_of_words.loc[:, bag_of_words.sum(axis=0) > self.col_sum_threshold]
        # bag_of_words = bag_of_words[list(bag_of_words.filter(regex=self.filter_regex))]

        return bag_of_words

    def start(self):
        """
        Start preprocessing of mental health data
        :return: preprocessed feature dataframe x and target column y
        """
        self.df['statement'] = self.df['statement'].apply(lambda x: prepare_text_with_regex(str(x)))
        bag_of_words = self.create_bag_of_words()
        bag_of_words = self.filter_bag_of_words(bag_of_words=bag_of_words)
        preprocessed_df = pd.concat([self.df, bag_of_words], axis=1)
        y = preprocessed_df[self.target_col]
        # encode mental health status
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        x = preprocessed_df.drop(self.columns_to_drop, axis=1)
        # prepare bag of words matrix for test set
        if self.train_val_columns:
            test_columns_set = set(x.columns.tolist())
            train_val_columns_set = set(self.train_val_columns)
            missing_train_val_columns = list(train_val_columns_set - test_columns_set)
            missing_test_columns = list(test_columns_set - train_val_columns_set)
            if missing_train_val_columns:
                # create missing columns
                df_missing_cols = pd.DataFrame(columns=missing_train_val_columns)
                x = pd.concat([x, df_missing_cols], axis=1)
            if missing_test_columns:
                # drop columns that were not present in the train bag of words matrix
                x = x.drop(missing_test_columns, axis=1)

        x = x.reindex(sorted(x.columns), axis=1)
        # set default value for missing columns and Nan values
        x = x.fillna(0)

        return x, y
