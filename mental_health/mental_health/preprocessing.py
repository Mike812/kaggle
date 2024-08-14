from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


class MentalHealthPreprocessing:
    def __init__(self, df):
        # train, test or validation dataframe
        self.df = df
        # column that will be predicted
        self.target_col = "status"
        # irrelevant columns for modeling
        self.columns_to_drop_train = [self.target_col, "statement"]
        # number of times word have to appear in statements
        self.col_sum_threshold = 100
        # have to contain characters
        self.filter_regex = '[A-Za-z]'

    def create_bag_of_words(self):
        statements = self.df["statement"].fillna("").tolist()
        count_vec = CountVectorizer()
        count_vec_fitted = count_vec.fit_transform(statements)
        word_counts = count_vec_fitted.toarray()
        words = count_vec.get_feature_names_out()
        bag_of_words = pd.DataFrame(data=word_counts, columns=words)

        return bag_of_words

    def filter_bag_of_words(self, bag_of_words):
        bag_of_words = bag_of_words.loc[:, bag_of_words.sum(axis=0) > self.col_sum_threshold]
        bag_of_words = bag_of_words[list(bag_of_words.filter(regex=self.filter_regex))]

        return bag_of_words

    def start(self):
        bag_of_words = self.create_bag_of_words()
        bag_of_words = self.filter_bag_of_words(bag_of_words=bag_of_words)
        preprocessed_df = pd.concat([self.df, bag_of_words], axis=1)
        y_train = preprocessed_df[self.target_col]
        x_train = preprocessed_df.drop(self.columns_to_drop_train, axis=1)
        return x_train, y_train
