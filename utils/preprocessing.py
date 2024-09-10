import abc
import string
from abc import ABC
import re
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from stop_words import get_stop_words


def prepare_text_with_regex(text):
    """
    Prepare text with multiple regex expressions.
    Copied from https://www.kaggle.com/code/muhammadfaizan65/sentiment-analysis-for-mental-health-nlp
    :param text: string
    :return: prepared text
    """
    text = text.lower()  # Lowercase text
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text.strip()


def create_bag_of_words(series):
    """
    Create bag of words dataframe without known stop words
    :return: bag of words dataframe
    """
    statements = series.fillna("").tolist()
    stop_words = get_stop_words("en")
    count_vec = CountVectorizer(stop_words=stop_words)
    count_vec_fitted = count_vec.fit_transform(statements)
    word_counts = count_vec_fitted.toarray()
    words = count_vec.get_feature_names_out()
    bag_of_words = pd.DataFrame(data=word_counts, columns=words)

    return bag_of_words


def filter_bag_of_words(bag_of_words, col_sum_threshold):
    """
    Filter dataframe by summing up the bag of words boolean columns and compare with a threshold
    :param bag_of_words: dataframe
    :param col_sum_threshold:
    :return: filtered bag_of_words dataframe
    """
    bag_of_words = bag_of_words.loc[:, bag_of_words.sum(axis=0) > col_sum_threshold]

    return bag_of_words


def rename_bow_columns(df, columns, postfix):
    """
    Rename bag of words columns
    :param df: dataframe
    :param columns: list of columns to rename if in df
    :param postfix: postfix to rename the column
    :return:
    """
    columns_to_drop = []
    for col in columns:
        if col in df:
            df[col + "_in_" + postfix] = df[col]
            columns_to_drop.append(col)
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)

    return df


def create_and_prepare_bag_of_words(series, col_sum_threshold, columns=None, postfix=None):
    """
    Create bag of words from series, filter and rename columns
    :param series: dataframe column that is used for bow transformation
    :param col_sum_threshold: filter threshold wrt. column sum
    :param columns: list of columns to check for renaming
    :param postfix: postfix to rename the column
    :return: prepared bag of words
    """
    series = series.apply(lambda x: prepare_text_with_regex(str(x)))
    bag_of_words = create_bag_of_words(series=series)
    bag_of_words_prepared = filter_bag_of_words(bag_of_words=bag_of_words, col_sum_threshold=col_sum_threshold)
    if columns:
        bag_of_words_prepared = rename_bow_columns(df=bag_of_words_prepared, columns=columns, postfix=postfix)

    return bag_of_words_prepared.reset_index(drop=True)


def adapt_test_to_training_data(test_df, train_val_columns):
    """
    Adds missing training columns to test dataframe and removes columns that are missing in training dataframe.
    :param test_df: x data of test dataframe
    :param train_val_columns:
    :return: adapted x data of test dataframe to training data
    """
    test_columns_set = set(test_df.columns.tolist())
    train_val_columns_set = set(train_val_columns)
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


def encode_labels(series):
    """
    Encode distinct string values of a series to numerical labels, e.g. {'NO': 0, 'YES': 1, 'PARTLY': 2}
    :param series:
    :return:
    """
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(series)


class Preprocessing(ABC):
    """
    Abstract class that contains variables and methods for preprocessing of input data
    """

    def __init__(self, df, target_col):
        """
        :param df: train, test or validation dataframe
        :param target_col: column that will be predicted
        """
        self.df = df
        self.target_col = target_col

    @abc.abstractmethod
    def start(self):
        """
        Starts preprocessing pipeline
        :return: preprocessed feature dataframe x and target column y
        """
        return
