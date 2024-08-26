import os

import pandas as pd
from sklearn.model_selection import train_test_split

from mental_health.mental_health.mental_health_preprocessing import MentalHealthPreprocessing

# print file names in data path
data_path = "../../data/"
for dir_name, _, file_names in os.walk(data_path):
    for filename in file_names:
        print(os.path.join(dir_name, filename))

df_small = pd.DataFrame({"statement": ["I think that I am bipolar or suffer from another mental disease",
                                       "I think I feel normal."],
                         "status": ["bipolar", "normal"]})

df_test = pd.read_csv(data_path + "combined_data.csv")
target_col = "status"

expected_bow = pd.DataFrame({"another": [1, 0], "bipolar": [1, 0], "disease": [1, 0], "feel": [0, 1],
                             "mental": [1, 0], "normal": [0, 1], "suffer": [1, 0], "think": [1, 1]})


class TestMentalHealthPreprocessing:
    def test_create_bag_of_words(self):
        mental_health_preprocessing = MentalHealthPreprocessing(
            df=df_small, target_col=target_col,
            col_sum_threshold=1
        )
        bag_of_words = mental_health_preprocessing.create_bag_of_words()
        print("Compare bag of words dataframes")
        assert bag_of_words.equals(expected_bow)

        filtered_bag_of_words = mental_health_preprocessing.filter_bag_of_words(bag_of_words)
        expected_filtered_bow = pd.DataFrame({"think": [1, 1]})
        print(filtered_bag_of_words.head())
        print(expected_filtered_bow.head())
        assert filtered_bag_of_words.equals(expected_filtered_bow)

    def test_start_preprocessing(self):
        mental_health_preprocessing = MentalHealthPreprocessing(
            df=df_small, target_col=target_col,
            col_sum_threshold=0
        )
        x, y = mental_health_preprocessing.start()
        assert x.equals(expected_bow)
        # check label encoding
        assert y[0] == 0
        assert y[1] == 1

        train_val_data, test_data = train_test_split(df_test, test_size=0.3, random_state=42)

        mental_health_preprocessing = MentalHealthPreprocessing(
            df=test_data, target_col=target_col,
            col_sum_threshold=50
        )
        x, y = mental_health_preprocessing.start()

        assert x.shape[1] < 3000
