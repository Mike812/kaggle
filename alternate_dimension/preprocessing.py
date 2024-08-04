import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class Preprocessing:
    def __init__(self, df, test):
        self.df = df
        self.test = test

    # Define one hot encoded and imputation columns
    one_hot_cols = ['HomePlanet', 'CryoSleep', 'Destination']
    imputation_cols = ["Age", "RoomService", "VIP", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    def apply_one_hot_encoding(self):
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        one_hot_data = pd.DataFrame(one_hot_encoder.fit_transform(self.df[self.one_hot_cols]))
        one_hot_data.index = self.df.index
        self.df = self.df.drop(self.one_hot_cols, axis=1)
        self.df = pd.concat([self.df, one_hot_data], axis=1)
        return self.df

    def apply_imputation(self):
        imputer = SimpleImputer()
        imputed_data = pd.DataFrame(imputer.fit_transform(self.df[self.imputation_cols]))
        self.df = self.df.drop(self.imputation_cols, axis=1)
        imputed_data.index = self.df.index
        imputed_data.columns = self.imputation_cols
        self.df = pd.concat([self.df, imputed_data], axis=1)
        return self.df

    # seems to have no effect on the mae
    def prepare_numerical_id(self):
        self.df['PassengerId'] = self.df.PassengerId.str.replace('_', '').astype(float)
        return self.df

    def start(self):
        self.df = self.apply_one_hot_encoding()
        self.df = self.apply_imputation()
        if self.test:
            x_train = self.df.drop(["Name", "Cabin", "PassengerId"], axis=1)
            return x_train
        else:
            y_train = self.df["Transported"]
            x_train = self.df.drop(["Transported", "Name", "Cabin", "PassengerId"], axis=1)
            return x_train, y_train
