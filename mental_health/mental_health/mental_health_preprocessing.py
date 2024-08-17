from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MentalHealthPreprocessing:
    """

    """
    def __init__(self, df, train_val_columns=None):
        """

        :param df:
        """
        # train, test or validation dataframe
        self.df = df
        # column that will be predicted
        self.target_col = "status"
        # irrelevant columns for modeling
        self.columns_to_drop = [self.target_col, "statement"]
        # number of times word have to appear in statements
        self.col_sum_threshold = 100
        # have to contain characters
        self.filter_regex = '[A-Za-z]'
        self.train_val_columns = train_val_columns

    def create_bag_of_words(self):
        """

        :return:
        """
        statements = self.df["statement"].fillna("").tolist()
        count_vec = CountVectorizer()
        count_vec_fitted = count_vec.fit_transform(statements)
        word_counts = count_vec_fitted.toarray()
        words = count_vec.get_feature_names_out()
        bag_of_words = pd.DataFrame(data=word_counts, columns=words)

        return bag_of_words

    def filter_bag_of_words(self, bag_of_words):
        """

        :param bag_of_words:
        :return:
        """
        bag_of_words = bag_of_words.loc[:, bag_of_words.sum(axis=0) > self.col_sum_threshold]
        bag_of_words = bag_of_words[list(bag_of_words.filter(regex=self.filter_regex))]

        return bag_of_words

    def start(self):
        """

        :return:
        """
        bag_of_words = self.create_bag_of_words()
        bag_of_words = self.filter_bag_of_words(bag_of_words=bag_of_words)
        preprocessed_df = pd.concat([self.df, bag_of_words], axis=1)
        y = preprocessed_df[self.target_col]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        x = preprocessed_df.drop(self.columns_to_drop, axis=1)
        # Todo: debug and check NaN values
        if self.train_val_columns:
            test_columns_set = set(x.columns.tolist())
            train_val_columns_set = set(self.train_val_columns)
            missing_train_val_columns = list(train_val_columns_set - test_columns_set)
            missing_test_columns = list(test_columns_set - train_val_columns_set)

            if missing_train_val_columns:
                for col in missing_train_val_columns:
                    x[col] = 0
            if missing_test_columns:
                x = x.drop(missing_test_columns, axis=1)

        x = x.reindex(sorted(x.columns), axis=1)
        x = x.fillna(0)
        return x, y
