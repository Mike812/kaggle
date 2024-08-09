import pandas as pd

from predict_transport.model_evaluation import ModelEvaluation
from predict_transport.preprocessing import Preprocessing

# Read data from kaggle as dataframes
train_val_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

cross_validation_result = ModelEvaluation(train_val_data=train_val_data).cross_validate()

print("\nFinal model:")
passenger_ids = test_data["PassengerId"]
# Start preprocessing of test data
x_test = Preprocessing(df=test_data, test=True).start()
# Pick best model from cross validation
mse_results = cross_validation_result.mse_results
best_model_index = mse_results.index(min(mse_results))
print("Mean squared error of best model: " + str(mse_results[best_model_index]))
print("Accuracy of best model: " + str(cross_validation_result.acc_results[best_model_index]))
print("Classification report of best model:\n" + str(cross_validation_result.reports[best_model_index]))
# Predict if passenger will be transported or not
y_test = cross_validation_result.xgb_models[best_model_index].predict(x_test)
submission_result = pd.DataFrame({"PassengerId": passenger_ids.values, "Transported": y_test})
print("Submission result:\n")
print(submission_result)
# write final result to data folder
submission_result.to_csv("../data/submission_result.csv", index=False)
