import pandas as pd
from xgboost import XGBClassifier
import os

# print file names in data path
data_path = "../data/"
for dir_name, _, file_names in os.walk(data_path):
    for filename in file_names:
        print(os.path.join(dir_name, filename))


def main():
    # Read data from kaggle as dataframe and define variables

    real_estate_data = pd.read_csv(data_path+"uae_real_estate_2024.csv")
    target_col = "price"
    model = XGBClassifier(n_estimators=500, learning_rate=0.1, early_stopping_rounds=5)
    cv_splits = 3
    test_size = 0.3

    print(real_estate_data.head())


if __name__ == "__main__":
    main()
