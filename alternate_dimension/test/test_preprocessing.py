import math

import pandas as pd
import pytest

from predict_transport.preprocessing import Preprocessing

df_small = pd.DataFrame({"PassengerId": [1, 2, 3], "HomePlanet": ["Earth", "Mars", "Saturn"], "CryoSleep": [1, 0, 1],
                         "Cabin": ["A", "B", "C"], "Destination": ["Galaxis1", "Galaxis2", "Galaxis1"],
                         "Age": [12, 21, None], "VIP": [1, 0, 1], "RoomService": [0, 3.9, 31.9],
                         "FoodCourt": [21, None, 90], "ShoppingMall": [None, 12, 21], "Spa": [12, 21, 12],
                         "VRDeck": [100, 0, 0], "Name": ["A", "B", "C"]})

df_test = pd.read_csv("../data/test.csv")


# Test all preprocessing methods
class TestPreprocessing:

    def test_apply_imputation(self):
        df1 = df_small.copy()
        assert df1.isnull().values.any()
        preprocessing = Preprocessing(df=df1, test=False)
        df1 = preprocessing.apply_imputation()
        assert not df1.isnull().values.any()

        df2 = df_test.copy()
        assert df2.isnull().values.any()
        preprocessing = Preprocessing(df=df2, test=False)
        df2 = preprocessing.apply_imputation()
        assert not df2[preprocessing.imputation_cols].isnull().values.any()

    def test_apply_one_hot_encoding(self):
        df1 = df_small.copy()
        preprocessing = Preprocessing(df=df1, test=False)
        assert len(df1["HomePlanet"].unique()) == 3
        assert len(df1["Destination"].unique()) == 2
        assert df1.shape[1] == 13

        df1 = preprocessing.apply_one_hot_encoding()
        assert df1.shape[1] == 16
        with pytest.raises(KeyError):
            assert df1["HomePlanet"]
        with pytest.raises(KeyError):
            assert df1["Destination"]

    def test_apply_log_normalization(self):
        df1 = df_small.copy()
        preprocessing = Preprocessing(df=df1, test=False)
        df1 = preprocessing.apply_log_normalization()
        assert not df1["RoomService"][0] == -math.inf
        assert df1["RoomService"][1] == 2.0
        assert df1["RoomService"][2] == 5.0
