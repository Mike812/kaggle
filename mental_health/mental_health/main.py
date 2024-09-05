import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import os
import numpy

from utils.model_evaluation import ModelEvaluation
from mental_health.mental_health.mental_health_preprocessing import MentalHealthPreprocessing
from utils.cross_validation_result import print_cv_result
from utils.io_utils import write_to_csv

# print file names in data path
data_path = "../data/"
for dir_name, _, file_names in os.walk(data_path):
    for filename in file_names:
        print(os.path.join(dir_name, filename))


def main():
    # Read data from kaggle as dataframe and define variables
    combined_data = pd.read_csv(data_path + "combined_data.csv", index_col=0)
    target_col = "status"
    model = XGBClassifier(n_estimators=500, learning_rate=0.1, early_stopping_rounds=5)
    cv_splits = 3
    test_size = 0.3

    train_val_data, test_data = train_test_split(combined_data, test_size=test_size, random_state=42)
    cv_result = ModelEvaluation(train_val_data=train_val_data.reset_index(drop=True),
                                preprocessor=MentalHealthPreprocessing,
                                target_col=target_col,
                                model=model,
                                splits=cv_splits,
                                bow=True).cross_validate_classification()

    print("\nFinal model:")
    # Pick best model from cross validation
    best_model_index = cv_result.mse_results.index(min(cv_result.mse_results))
    xgb_final_model = cv_result.xgb_models[best_model_index]
    train_val_columns = cv_result.train_val_columns
    # print results of best model
    print_cv_result(cv_result=cv_result, best_model_index=best_model_index)

    # Start preprocessing of test data
    x_test, y_test = MentalHealthPreprocessing(df=test_data.reset_index(drop=True), target_col=target_col,
                                               train_val_columns=train_val_columns,
                                               col_sum_threshold=50).start()

    # predict and evaluate final results
    y_pred = xgb_final_model.predict(x_test)
    mse = mean_squared_error(y_pred, y_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_true=y_test, y_pred=y_pred)
    print("Mean squared error: " + str(mse))
    print("Model accuracy: " + str(acc))
    print("Classification report:\n " + str(report))

    # save final model and prepared data for jupyter notebook
    pickle.dump(xgb_final_model, open(data_path + "xgb_mental_health.pkl", "wb"))
    write_to_csv(file=data_path + "train_val_columns.csv", data=train_val_columns)
    numpy.savetxt(data_path + "y_test.csv", y_test, delimiter=",")
    numpy.savetxt(data_path + "y_pred.csv", y_pred, delimiter=",")


if __name__ == "__main__":
    main()
