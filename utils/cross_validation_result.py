
def print_cv_classification_result(cv_result, best_model_index):
    """
    Prints the cross validation classification results to console
    :param cv_result: cross validation result
    :param best_model_index: index of best model
    """
    print("Mean squared error of best model: " + str(cv_result.mse_results[best_model_index]))
    print("Accuracy of best model: " + str(cv_result.acc_results[best_model_index]))
    print("Classification report of best model:\n" + str(cv_result.reports[best_model_index]))
    print()


def print_cv_regression_result(cv_result, best_model_index):
    """
    Prints the cross validation regression results to console
    :param cv_result: cross validation result
    :param best_model_index: index of best model
    """
    print("Mean squared error of best model: " + str(cv_result.mse_results[best_model_index]))
    print("Mean absolute error of best model: " + str(cv_result.mae_results[best_model_index]))
    print("R2 score of best model: " + str(cv_result.r2_results[best_model_index]))
    print()


class CVResult:
    """
    Represents the result object of a cross validation
    """
    def __init__(self, xgb_models, mse_results,):
        """
        :param xgb_models: xg boost model list
        :param mse_results: mean squared error result list
        """
        self.xgb_models = xgb_models
        self.mse_results = mse_results


class CVResultClassification(CVResult):
    """
    Represents the result object of a cross validation for a classification model
    """
    def __init__(self, xgb_models, mse_results, acc_results, reports):
        """
        :param xgb_models: xg boost model list
        :param mse_results: mean squared error result list
        :param acc_results: accuracy result list
        :param reports: classification report list
        """
        super().__init__(xgb_models, mse_results)
        self.acc_results = acc_results
        self.reports = reports


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


class CVResultRegression(CVResult):
    """
    Represents the result object of a cross validation for a regression model
    """
    def __init__(self, xgb_models, mse_results, mae_results, r2_results):
        """
        :param xgb_models: xg boost model list
        :param mse_results: mean squared error result list
        :param mae_results: mean absolute error result list
        :param r2_results: r2 score list
        """
        super().__init__(xgb_models, mse_results)
        self.mae_results = mae_results
        self.r2_results = r2_results


class CVResultRegressionBow(CVResultRegression):
    """
    Represents the result object of a cross validation for a regression model using bag of words columns
    """
    def __init__(self, xgb_models, mse_results, mae_results, r2_results, train_val_columns):
        """
        :param xgb_models: xg boost model list
        :param mse_results: mean squared error result list
        :param mae_results: mean absolute error result list
        :param r2_results: r2 score list
        :param train_val_columns: columns of dataframe that was used for modeling including bag of words columns
        """
        super().__init__(xgb_models, mse_results, mae_results, r2_results)
        self.train_val_columns = train_val_columns
