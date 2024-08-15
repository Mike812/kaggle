import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils.model_evaluation import ModelEvaluation
from mental_health.mental_health.mental_health_preprocessing import MentalHealthPreprocessing

# Read data from kaggle as dataframes
combined_data = pd.read_csv("../data/combined_data.csv", index_col=0)
train_val_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)
model = XGBClassifier(n_estimators=500, learning_rate=0.1, early_stopping_rounds=5)
cross_validation_result = ModelEvaluation(train_val_data=train_val_data.reset_index(),
                                          preprocesser=MentalHealthPreprocessing,
                                          model=model).cross_validate()

print("\nFinal model:")
statements = test_data["statement"]
# Start preprocessing of test data
x_test = MentalHealthPreprocessing(df=test_data).start()
# Pick best model from cross validation
mse_results = cross_validation_result.mse_results
best_model_index = mse_results.index(min(mse_results))
print("Mean squared error of best model: " + str(mse_results[best_model_index]))
print("Accuracy of best model: " + str(cross_validation_result.acc_results[best_model_index]))
print("Classification report of best model:\n" + str(cross_validation_result.reports[best_model_index]))
# Predict if passenger will be transported or not
y_test = cross_validation_result.xgb_models[best_model_index].predict(x_test)
submission_result = pd.DataFrame({"PassengerId": statements.values, "Transported": y_test})
print("Submission result:\n")
print(submission_result)
