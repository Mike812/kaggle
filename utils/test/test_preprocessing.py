import pandas as pd

from utils.preprocessing import prepare_text_with_regex, create_bag_of_words, filter_bag_of_words


df_train = pd.DataFrame({"statement": ["I think that I am bipolar or suffer from another mental disease",
                                       "I think I feel normal."],
                         "status": ["bipolar", "normal"]})

df_expected_bow_train = pd.DataFrame({"another": [1, 0], "bipolar": [1, 0], "disease": [1, 0], "feel": [0, 1],
                                      "mental": [1, 0], "normal": [0, 1], "suffer": [1, 0], "think": [1, 1]})


class TestPreprocessing:
    def test_prepare_text_with_regex(self):
        test_string = "Abc Def [hij] 20s."
        prepared_string = prepare_text_with_regex(test_string)

        assert prepared_string == "abc def"

    def test_create_bag_of_words(self):
        bag_of_words = create_bag_of_words(series=df_train["statement"])
        print("Compare bag of words dataframes")
        assert bag_of_words.equals(df_expected_bow_train)

        filtered_bag_of_words = filter_bag_of_words(bag_of_words, col_sum_threshold=1)
        expected_filtered_bow = pd.DataFrame({"think": [1, 1]})
        print(filtered_bag_of_words.head())
        print(expected_filtered_bow.head())
        assert filtered_bag_of_words.equals(expected_filtered_bow)
