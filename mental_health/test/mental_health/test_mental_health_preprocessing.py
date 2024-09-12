import pandas as pd
from sklearn.model_selection import train_test_split

from mental_health.mental_health.mental_health_preprocessing import MentalHealthPreprocessing


df_train = pd.DataFrame({"statement": ["I think that I am bipolar or suffer from another mental disease",
                                       "I think I feel normal."],
                         "status": ["bipolar", "normal"]})
df_test = pd.DataFrame({"statement": ["I am bipolar or suffer from another mental sickness",
                                      "I feel normal."],
                        "status": ["bipolar", "normal"]})

df_expected_bow_train = pd.DataFrame({"another": [1, 0], "bipolar": [1, 0], "disease": [1, 0], "feel": [0, 1],
                                      "mental": [1, 0], "normal": [0, 1], "suffer": [1, 0], "think": [1, 1]})

df_combined = pd.read_csv("/Users/michaelkrug/git/kaggle/mental_health/data/combined_data.csv")
target_col = "status"


class TestMentalHealthPreprocessing:

    def test_start_preprocessing(self):
        mental_health_preprocessing = MentalHealthPreprocessing(
            df=df_train, target_col=target_col, col_sum_threshold=0
        )
        x, y = mental_health_preprocessing.start()
        assert x.equals(df_expected_bow_train)
        # check label encoding
        assert y[0] == 0
        assert y[1] == 1

        train_val_data, test_data = train_test_split(df_combined, test_size=0.3, random_state=42)

        mental_health_preprocessing = MentalHealthPreprocessing(
            df=test_data, target_col=target_col, col_sum_threshold=50
        )
        x, y = mental_health_preprocessing.start()

        assert x.shape[1] < 3000
