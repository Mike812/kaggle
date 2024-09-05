import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os

from utils.cross_validation_result import print_cv_result
from utils.model_evaluation import ModelEvaluationBagOfWords
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
    cv_result = ModelEvaluationBagOfWords(train_val_data=train_val_data.reset_index(),
                                          preprocessor=RealEstatePreprocessing,
                                          target_col=target_col,
                                          model=model,
                                          splits=cv_splits).cross_validate_classification(transform_to_sparse=False)

    print("\nFinal model:")
    # Pick best model from cross validation
    best_model_index = cv_result.mse_results.index(min(cv_result.mse_results))
    xgb_final_model = cv_result.xgb_models[best_model_index]
    train_val_columns = cv_result.train_val_columns
    # print results of best model
    print_cv_result(cv_result=cv_result, best_model_index=best_model_index)

    x_test, y_test = RealEstatePreprocessing(df=test_data.reset_index(), target_col=target_col,
                                             train_val_columns=train_val_columns).start()

    # predict and evaluate final results
    y_pred = xgb_final_model.predict(x_test)
    mse = mean_squared_error(y_pred, y_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_true=y_test, y_pred=y_pred)
    print("Mean squared error: " + str(mse))
    print("Model accuracy: " + str(acc))
    print("Classification report:\n " + str(report))


if __name__ == "__main__":
    main()
