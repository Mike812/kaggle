import os

import pandas as pd

from real_estate.real_estate.real_estate_preprocessing import RealEstatePreprocessing


df = pd.DataFrame({
    "title": ["Great ROI I High Floor", "Nice balcony"],
    "displayAddress": ["Binghatti Canal, Business Bay, Dubai", "Dubai"],
    "bathrooms": ["2", "1"],
    "bedrooms": ["4", "2"],
    "addedOn": ["2024-08-14T12:02:53Z", "2024-09-14T12:02:53Z"],
    "type": ["Residential for Sale", "Residential for Sale"],
    "price": [2500000, 1000000],
    "verified": [True, False],
    "priceDuration": ["sell", "sell"],
    "sizeMin": ["1000 sqft", "1000 sqft"],
    "furnishing": ["NO", "YES"],
    "description": ["MNA Properties is delighted", "MNA Properties"]})

x_expected = pd.DataFrame({
    "bathroom_per_bedrooms": [2.0, 2.0],
    "bathrooms": [2, 1],
    "bedrooms": [4, 2],
    "furnishing": [0, 1],
    "price_per_sqft": [2500.0, 1000.0],
    "sizeMin": [1000, 1000],
    "verified": [True, False],
})

real_estate_data = pd.read_csv("/Users/michaelkrug/git/kaggle/real_estate/data/uae_real_estate_2024.csv")
target_col = "price"


class TestRealEstatePreprocessing:

    def test_start_preprocessing(self):
        real_estate_preprocessing = RealEstatePreprocessing(
            df=df, target_col=target_col, title_col_sum_threshold=0, description_col_sum_threshold=0
        )
        x, y = real_estate_preprocessing.start()
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)

        assert x.equals(x_expected)
        assert y.equals(pd.Series([2500000, 1000000]))

        real_estate_preprocessing = RealEstatePreprocessing(
            df=real_estate_data, target_col=target_col
        )
        x, y = real_estate_preprocessing.start()

        assert x.shape[1] == 7
