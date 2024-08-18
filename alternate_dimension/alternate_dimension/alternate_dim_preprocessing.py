import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class AlternateDimPreprocessing:
    """
    Represents all methods and variables that are needed for the preprocessing of the alternate dimension
    input dataframes. Inspired by https://www.kaggle.com/learn/intermediate-machine-learning and
    https://www.kaggle.com/code/vaasubisht/eda-statisticaltests-gradient-boosting-shap learning content.
    """
    def __init__(self, df, test=False):
        """

        :param df:
        :param test:
        """
        # train, test or validation dataframe
        self.df = df
        # flag for test data set
        self.test = test
        # column that will be predicted
        self.target_col = "Transported"
        # Categorical columns that can be one hot encoded (less than 10 distinct values)
        self.one_hot_cols = ['HomePlanet', 'Destination']
        # Columns that will be imputed due to missing values
        self.imputation_cols = ["Age", "RoomService", "CryoSleep", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        # Inbalanced columns that will be normalized, [col for col in self.imputation_cols if (self.df[col].skew() > 1)]
        self.normalization_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        # irrelevant columns for modeling
        self.columns_to_drop_train = ["Name", "Cabin", "PassengerId", "VIP", self.target_col]
        self.columns_to_drop_test = ["Name", "Cabin", "PassengerId", "VIP"]

    def apply_one_hot_encoding(self):
        """
        Function to apply one hot encoding to specific columns of a dataframe
        :return:
        """
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        one_hot_data = pd.DataFrame(one_hot_encoder.fit_transform(self.df[self.one_hot_cols]))
        one_hot_data.index = self.df.index
        self.df = self.df.drop(self.one_hot_cols, axis=1)
        self.df = pd.concat([self.df, one_hot_data], axis=1)
        return self.df

    def apply_imputation(self):
        """
        Function to apply imputation to specific columns of a dataframe
        :return:
        """
        imputer = SimpleImputer()
        imputed_data = pd.DataFrame(imputer.fit_transform(self.df[self.imputation_cols]))
        self.df = self.df.drop(self.imputation_cols, axis=1)
        imputed_data.index = self.df.index
        imputed_data.columns = self.imputation_cols
        self.df = pd.concat([self.df, imputed_data], axis=1)
        return self.df

    def apply_log_normalization(self):
        """
        Apply log2 transformation to inbalanced columns
        :return:
        """
        for col in self.normalization_columns:
            # add 0.1 to values to avoid - inf values in log function; np.seterr(divide='ignore') not needed
            self.df[col] += 0.1
            self.df[col] = np.where(self.df[col] > 0, np.log2(self.df[col]), 0)
        return self.df

    def start(self):
        """
        Starts preprocessing
        :return:
        """
        self.df = self.apply_imputation()
        self.df = self.apply_one_hot_encoding()
        self.df = self.apply_log_normalization()

        if self.test:
            x_train = self.df.drop(self.columns_to_drop_test, axis=1)
            return x_train
        else:
            y_train = self.df["Transported"]
            x_train = self.df.drop(self.columns_to_drop_train, axis=1)
            return x_train, y_train
