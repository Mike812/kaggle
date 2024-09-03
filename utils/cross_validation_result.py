
def print_cv_result(cv_result, best_model_index):
    """
    Prints the cross validation results to console
    :param cv_result: cross validation result
    :param best_model_index: index of best model
    """
    print("Mean squared error of best model: " + str(cv_result.mse_results[best_model_index]))
    print("Accuracy of best model: " + str(cv_result.acc_results[best_model_index]))
    print("Classification report of best model:\n" + str(cv_result.reports[best_model_index]))


class CVResultClassification:
    """
    Represents the result object of a cross validation for a classification model
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


class CVResultRegression:
    """
    Represents the result object of a cross validation for a regression model
    """
    def __init__(self, mse_results, r2_results):
        """
        :param mse_results: mean squared error result list
        :param r2_results:
        """
        self.mse_results = mse_results
        self.xgb_models = r2_results


class CVResultClassificationBoW(CVResultClassification):
    """
    Represents the result object of a cross validation for a classification model using bag of words columns
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
