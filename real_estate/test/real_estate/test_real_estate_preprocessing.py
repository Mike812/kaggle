import os

import pandas as pd

from real_estate.real_estate.real_estate_preprocessing import RealEstatePreprocessing

# print file names in data path
data_path = "../../data/"
for dir_name, _, file_names in os.walk(data_path):
    for filename in file_names:
        print(os.path.join(dir_name, filename))

df = pd.DataFrame({
    "title": ["Great ROI I High Floor"],
    "displayAddress": ["Binghatti Canal, Business Bay, Dubai"],
    "bathrooms": ["2"],
    "bedrooms": ["4"],
    "addedOn": ["2024-08-14T12:02:53Z"],
    "type": ["Residential for Sale"],
    "price": [2500000],
    "verified": [True],
    "priceDuration": ["sell"],
    "sizeMin": ["1000 sqft"],
    "furnishing": ["NO"],
    "description": ["MNA Properties is delighted"]})

x_expected = pd.DataFrame({
    "bathroom_per_bedrooms": [2.0],
    "bathrooms": [2],
    "bedrooms": [4],
    "furnishing": [0],
    "price_per_sqft": [2500.0],
    "sizeMin": [1000],
    "verified": [True],
})

real_estate_data = pd.read_csv(data_path + "uae_real_estate_2024.csv")
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
        assert y.equals(pd.Series(2500000))

        real_estate_preprocessing = RealEstatePreprocessing(
            df=real_estate_data, target_col=target_col
        )
        x, y = real_estate_preprocessing.start()

        assert x.shape[1] == 7
