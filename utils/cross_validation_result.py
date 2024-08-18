
def print_cv_result(cv_result, best_model_index):
    """

    :param cv_result:
    :param best_model_index:
    :return:
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

        :param mse_results:
        :param xgb_models:
        :param acc_results:
        :param reports:
        """
        self.mse_results = mse_results
        self.xgb_models = xgb_models
        self.acc_results = acc_results
        self.reports = reports
