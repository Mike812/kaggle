import pandas as pd
from sklearn.model_selection import train_test_split

from mental_health.mental_health.preprocessing import MentalHealthPreprocessing

# Read data from kaggle as dataframes
combined_data = pd.read_csv("../data/combined_data.csv")
train_val_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

x_test, y_test = MentalHealthPreprocessing(df=test_data).start()

print(y_test.shape)
