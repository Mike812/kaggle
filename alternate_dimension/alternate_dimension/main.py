import pandas as pd
from xgboost import XGBClassifier

from alternate_dimension.alternate_dimension.alternate_dim_preprocessing import AlternateDimPreprocessing
from utils.cross_validation_result import print_cv_result
from utils.model_evaluation import ModelEvaluation


def main():
    # Read data from kaggle as dataframes
    train_val_data = pd.read_csv("../data/train.csv")
    test_data = pd.read_csv("../data/test.csv")

    model = XGBClassifier(n_estimators=500, learning_rate=0.1, early_stopping_rounds=5)
    cv_result = ModelEvaluation(train_val_data=train_val_data,
                                preprocessor=AlternateDimPreprocessing,
                                model=model).cross_validate()

    print("\nFinal model:")
    passenger_ids = test_data["PassengerId"]
    # Start preprocessing of test data
    x_test = AlternateDimPreprocessing(df=test_data, test=True).start()
    # Pick best model from cross validation
    best_model_index = cv_result.mse_results.index(min(cv_result.mse_results))
    print_cv_result(cv_result=cv_result, best_model_index=best_model_index)
    # Predict if passenger will be transported or not
    y_test = cv_result.xgb_models[best_model_index].predict(x_test)
    submission_result = pd.DataFrame({"PassengerId": passenger_ids.values, "Transported": y_test})
    print("Submission result:\n")
    print(submission_result)
    # write final result to data folder
    submission_result.to_csv("../data/submission_result.csv", index=False)


if __name__ == "__main__":
    main()
