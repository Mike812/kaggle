import pandas as pd

from predict_transport.preprocessing import Preprocessing

df_small = pd.DataFrame({"PassengerId": [1, 2, 3], "HomePlanet": ["Earth", "Mars", "Saturn"], "CryoSleep": [1, 0, 1],
                         "Cabin": ["A", "B", "C"], "Destination": ["Galaxis1", "Galaxis2", "Galaxis1"],
                         "Age": [12, 21, None], "VIP": [1, 0, 1], "RoomService": [11, 23, None],
                         "FoodCourt": [21, None, 90], "ShoppingMall": [None, 12, 21], "Spa": [12, 21, 12],
                         "VRDeck": [100, 0, 0], "Name": ["A", "B", "C"]})

df_test = pd.read_csv("../data/test.csv")


def test_apply_imputation():
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
