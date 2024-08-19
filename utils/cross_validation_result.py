
def print_cv_result(cv_result, best_model_index):
    """
    Prints the cross validation results to console
    :param cv_result: cross validation result
    :param best_model_index: index of best model
    """
    print("Mean squared error of best model: " + str(cv_result.mse_results[best_model_index]))
    print("Accuracy of best model: " + str(cv_result.acc_results[best_model_index]))
    print("Classification report of best model:\n" + str(cv_result.reports[best_model_index]))


class CrossValidationResult:
    """
    Represents result object of cross validation
    """
    def __init__(self, mse_results, xgb_models, acc_results, reports):
        """
        :param mse_results: mean squared error result list
        :param xgb_models: xg boost model list
        :param acc_results: accuracy result list
        :param reports: classification report list
        """
        self.mse_results = mse_results
        self.xgb_models = xgb_models
        self.acc_results = acc_results
        self.reports = reports


class CrossValidationResultBoW(CrossValidationResult):
    """
    Represents result object of cross validation
    """
    def __init__(self, mse_results, xgb_models, acc_results, reports, train_val_columns):
        """
        :param mse_results: mean squared error result list
        :param xgb_models: xg boost model list
        :param acc_results: accuracy result list
        :param reports: classification report list
        :param train_val_columns: columns of dataframe that was used for modeling including bag of words columns
        """
        super().__init__(mse_results=mse_results, xgb_models=xgb_models, acc_results=acc_results,
                         reports=reports)
        self.train_val_columns = train_val_columns
