import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os

from real_estate.real_estate.real_estate_preprocessing import RealEstatePreprocessing

# print file names in data path
data_path = "../data/"
for dir_name, _, file_names in os.walk(data_path):
    for filename in file_names:
        print(os.path.join(dir_name, filename))


def main():
    # Read data from kaggle as dataframe and define variables

    real_estate_data = pd.read_csv(data_path+"uae_real_estate_2024.csv")
    target_col = "price"
    model = XGBRegressor(n_estimators=500, learning_rate=0.1, early_stopping_rounds=5)
    cv_splits = 3
    test_size = 0.3

    train_val_data, test_data = train_test_split(real_estate_data, test_size=test_size, random_state=42)
    x, y = RealEstatePreprocessing(df=train_val_data, target_col=target_col).start()
    print(x.head())
    print(y)


if __name__ == "__main__":
    main()
