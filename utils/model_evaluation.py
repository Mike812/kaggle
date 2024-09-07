from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report, r2_score
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
import time

from utils.cross_validation_result import CVResultClassificationBoW, CVResultClassification, CVResultRegressionBow


def print_classification_results(mse, acc, report):
    """
    Prints classification metrics to console
    :param mse:
    :param acc:
    :param report:
    :return:
    """
    print("Mean squared error: " + str(mse))
    print("Model accuracy: " + str(acc))
    print("Classification report: " + str(report))
    print()


def print_regression_results(mse, mae, r2_result):
    """
    Prints regression metrics to console
    :param mse:
    :param mae:
    :param r2_result:
    :return:
    """
    print("Mean squared error: " + str(mse))
    print("Mean absolute error: " + str(mae))
    print("R2 Score: " + str(r2_result))
    print()


class ModelEvaluation:
    """
    Consists of all methods and variables that are needed for model evaluation
    """

    def __init__(self, train_val_data, preprocessor, target_col, model, splits, bow=False):
        """
        :param train_val_data: dataframe with training and validation data
        :param preprocessor: preprocessor class
        :param target_col: column that will be predicted
        :param model: machine learning model
        :param splits: number of cross validation rounds
        :param bow: flag for bag of words preprocessing
        """
        self.train_val_data = train_val_data
        self.preprocessor = preprocessor
        self.target_col = target_col
        self.model = model
        self.splits = splits
        self.bow = bow

    def cross_validate_classification(self, x_to_sparse_matrix=True):
        """
        Cross validate a classification model
        :return: CVResultClassification or CrossValidationResultBoW if a bag of words transformation is used
        """
        # Prepare cross validation of model predictions
        kf = KFold(n_splits=self.splits, shuffle=True, random_state=42)
        counter = 1
        xgb_models = []
        mse_results = []
        acc_results = []
        reports = []
        x_train_val, y_train_val = self.preprocessor(df=self.train_val_data, target_col=self.target_col).start()
        train_val_columns = x_train_val.columns.tolist()
        print(f'Train data: {x_train_val.shape}')
        print(f'Validation data: {y_train_val.shape}')
        # Start cross validation
        for train_index, val_index in (kf.split(self.train_val_data)):
            print(f'\nFold: {counter}')
            start_time = time.time()
            if x_to_sparse_matrix:
                # transform bag of words matrix to sparse matrix to speed up training
                x_train = csr_matrix(x_train_val.loc[train_index].values)
                x_val = csr_matrix(x_train_val.loc[val_index].values)
            else:
                x_train = x_train_val.loc[train_index]
                x_val = x_train_val.loc[val_index]
            y_train = y_train_val[train_index]
            y_val = y_train_val[val_index]
            # train model
            self.model.fit(x_train, y_train, verbose=False, eval_set=[(x_val, y_val)])
            # predict and evaluate model quality
            y_predictions = self.model.predict(x_val)
            mse = mean_squared_error(y_true=y_val, y_pred=y_predictions)
            acc = accuracy_score(y_true=y_val, y_pred=y_predictions)
            report = classification_report(y_true=y_val, y_pred=y_predictions)
            xgb_models.append(self.model)
            mse_results.append(mse)
            acc_results.append(acc)
            reports.append(report)
            end_time = time.time()
            print_classification_results(mse=mse, acc=acc, report=report)
            print("Elapsed time: " + str(round(end_time - start_time, 2)) + " seconds")
            counter += 1

        if self.bow:
            result = CVResultClassificationBoW(mse_results=mse_results, xgb_models=xgb_models, acc_results=acc_results,
                                               reports=reports, train_val_columns=train_val_columns)
        else:
            result = CVResultClassification(mse_results=mse_results, xgb_models=xgb_models, acc_results=acc_results,
                                            reports=reports)

        return result

    def cross_validate_regression(self, x_to_sparse_matrix=True):
        """
        Cross validate a regression model
        :return: CVResultClassification or CrossValidationResultBoW if a bag of words transformation is used
        """
        # Prepare cross validation of model predictions
        kf = KFold(n_splits=self.splits, shuffle=True, random_state=42)
        counter = 1
        xgb_models = []
        mse_results = []
        mae_results = []
        r2_results = []
        x_train_val, y_train_val = self.preprocessor(df=self.train_val_data, target_col=self.target_col).start()
        train_val_columns = x_train_val.columns.to_list()
        print(f'Train data: {x_train_val.shape}')
        print(f'Validation data: {y_train_val.shape}')
        # Start cross validation
        for train_index, val_index in (kf.split(self.train_val_data)):
            print(f'\nFold: {counter}')
            start_time = time.time()
            if x_to_sparse_matrix:
                # transform bag of words matrix to sparse matrix to speed up training
                x_train = csr_matrix(x_train_val.loc[train_index].values)
                x_val = csr_matrix(x_train_val.loc[val_index].values)
            else:
                x_train = x_train_val.loc[train_index]
                x_val = x_train_val.loc[val_index]
            y_train = y_train_val[train_index]
            y_val = y_train_val[val_index]
            # train model
            self.model.fit(x_train, y_train, verbose=False, eval_set=[(x_val, y_val)])
            # predict and evaluate model quality
            y_predictions = self.model.predict(x_val)
            mse = mean_squared_error(y_true=y_val, y_pred=y_predictions)
            mae = mean_absolute_error(y_true=y_val, y_pred=y_predictions)
            r2_result = r2_score(y_true=y_val, y_pred=y_predictions)
            xgb_models.append(self.model)
            mse_results.append(mse)
            mae_results.append(mae)
            r2_results.append(r2_result)
            end_time = time.time()
            print_regression_results(mse=mse, mae=mae, r2_result=r2_result)
            print("Elapsed time: " + str(round(end_time - start_time, 2)) + " seconds")
            counter += 1

        return CVResultRegressionBow(xgb_models=xgb_models, mse_results=mse_results, mae_results=mae_results,
                                     r2_results=r2_results, train_val_columns=train_val_columns)

