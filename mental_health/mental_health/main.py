import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from mental_health.mental_health.model_evaluation_bag_of_words import ModelEvaluationBagOfWords
from mental_health.mental_health.mental_health_preprocessing import MentalHealthPreprocessing
from utils.cross_validation_result import print_cv_result


def main():
    # Read data from kaggle as dataframe and define variables
    combined_data = pd.read_csv("../data/combined_data.csv", index_col=0)
    target_col = "status"
    model = XGBClassifier(n_estimators=500, learning_rate=0.1, early_stopping_rounds=5)
    cv_splits = 3
    test_size = 0.3
    col_sum_threshold_train = 150
    col_sum_threshold_test = 50

    train_val_data, test_data = train_test_split(combined_data, test_size=test_size, random_state=42)
    cv_result = ModelEvaluationBagOfWords(train_val_data=train_val_data.reset_index(),
                                          preprocessor=MentalHealthPreprocessing,
                                          target_col=target_col,
                                          col_sum_threshold=col_sum_threshold_train,
                                          model=model,
                                          splits=cv_splits).cross_validate()

    print("\nFinal model:")
    # Pick best model from cross validation
    best_model_index = cv_result.mse_results.index(min(cv_result.mse_results))
    train_val_columns = cv_result.train_val_columns
    print_cv_result(cv_result=cv_result, best_model_index=best_model_index)

    # Start preprocessing of test data
    x_test, y_test = MentalHealthPreprocessing(df=test_data.reset_index(), target_col=target_col,
                                               train_val_columns=train_val_columns,
                                               col_sum_threshold=col_sum_threshold_test).start()

    # predict and evaluate final results
    y_pred = cv_result.xgb_models[best_model_index].predict(x_test)
    mse = mean_squared_error(y_pred, y_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_true=y_test, y_pred=y_pred)
    print("Mean squared error: " + str(mse))
    print("Model accuracy: " + str(acc))
    print("Classification report:\n " + str(report))


if __name__ == "__main__":
    main()
