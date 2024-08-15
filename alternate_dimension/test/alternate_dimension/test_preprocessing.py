import math

import pandas as pd
import pytest

from alternate_dimension.alternate_dimension.alternate_dim_preprocessing import AlternateDimPreprocessing

df_small = pd.DataFrame({"PassengerId": [1, 2, 3], "HomePlanet": ["Earth", "Mars", "Saturn"], "CryoSleep": [1, 0, 1],
                         "Cabin": ["A", "B", "C"], "Destination": ["Galaxis1", "Galaxis2", "Galaxis1"],
                         "Age": [12, 21, None], "VIP": [1, 0, 1], "RoomService": [0, 3.9, 31.9],
                         "FoodCourt": [21, None, 90], "ShoppingMall": [None, 12, 21], "Spa": [12, 21, 12],
                         "VRDeck": [100, 0, 0], "Name": ["A", "B", "C"], "Transported": [1, 0, 1]})

df_test = pd.read_csv("../data/test.csv")


# Test all preprocessing methods
class TestPreprocessing:

    def test_apply_imputation(self):
        df1 = df_small.copy()
        assert df1.isnull().values.any()
        preprocessing = AlternateDimPreprocessing(df=df1, test=False)
        df1 = preprocessing.apply_imputation()
        assert not df1.isnull().values.any()

        df2 = df_test.copy()
        assert df2.isnull().values.any()
        preprocessing = AlternateDimPreprocessing(df=df2, test=True)
        df2 = preprocessing.apply_imputation()
        assert not df2[preprocessing.imputation_cols].isnull().values.any()

    def test_apply_one_hot_encoding(self):
        df1 = df_small.copy()
        preprocessing = AlternateDimPreprocessing(df=df1, test=False)
        assert len(df1["HomePlanet"].unique()) == 3
        assert len(df1["Destination"].unique()) == 2
        assert df1.shape[1] == 14

        df1 = preprocessing.apply_one_hot_encoding()
        assert df1.shape[1] == 17
        with pytest.raises(KeyError):
            assert df1["HomePlanet"]
        with pytest.raises(KeyError):
            assert df1["Destination"]

    def test_apply_log_normalization(self):
        df1 = df_small.copy()
        preprocessing = AlternateDimPreprocessing(df=df1, test=False)
        df1 = preprocessing.apply_log_normalization()
        assert not df1["RoomService"][0] == -math.inf
        assert df1["RoomService"][1] == 2.0
        assert df1["RoomService"][2] == 5.0

    def test_start(self):
        # there are 5 one hot encoded columns
        feature_columns = ["Age", "RoomService", "CryoSleep", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", 0, 1, 2, 3, 4]
        df1 = df_small.copy()
        preprocessing = AlternateDimPreprocessing(df=df1, test=False)
        x, y = preprocessing.start()
        assert x.columns.tolist() == feature_columns
        assert not x.isnull().values.any()
        assert x.shape[0] == 3
        assert y.size == 3
        assert y.name == "Transported"

        df2 = df_test.copy()
        no_one_hot_cols = df2["HomePlanet"].unique().size + df2["Destination"].unique().size
        one_hot_cols = list(range(no_one_hot_cols))
        feature_columns = (["Age", "RoomService", "CryoSleep", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
                           + one_hot_cols)

        assert df2.isnull().values.any()
        preprocessing = AlternateDimPreprocessing(df=df2, test=True)
        x = preprocessing.start()
        assert x.columns.tolist() == feature_columns
        assert not x.isnull().values.any()
        assert x.shape[0] == df_test.shape[0]

