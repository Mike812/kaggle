from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
import time


class CrossValidationResultBoW:
    """
    Represents result object of cross validation
    """
    def __init__(self, mse_results, xgb_models, acc_results, reports, train_val_columns):
        """

        :param mse_results:
        :param xgb_models:
        :param acc_results:
        :param reports:
        """
        self.mse_results = mse_results
        self.xgb_models = xgb_models
        self.acc_results = acc_results
        self.reports = reports,
        self.train_val_columns = train_val_columns


class ModelEvaluationBagOfWords:
    """
    Consists of all methods and variables that are needed for model evaluation
    """

    def __init__(self, train_val_data, preprocessor, model, splits):
        """

        :param train_val_data:
        :param preprocessor:
        :param model:
        :param splits:
        """
        self.train_val_data = train_val_data
        self.preprocessor = preprocessor
        self.model = model
        self.splits = splits

    def cross_validate(self):
        """

        :return:
        """
        # Prepare cross validation of model predictions
        kf = KFold(n_splits=self.splits, shuffle=True, random_state=42)
        counter = 1
        mse_results = []
        xgb_models = []
        acc_results = []
        reports = []
        x_train_val, y_train_val = self.preprocessor(df=self.train_val_data).start()
        train_val_columns = x_train_val.columns.tolist()
        print(f'Train data: {x_train_val.shape}')
        print(f'Validation data: {y_train_val.shape}')
        # Start cross validation
        for train_index, val_index in (kf.split(self.train_val_data)):
            print(f'\nFold: {counter}')
            start_time = time.time()
            x_train = csr_matrix(x_train_val.loc[train_index].values)
            y_train = y_train_val[train_index]
            x_val = csr_matrix(x_train_val.loc[val_index].values)
            y_val = y_train_val[val_index]
            # Define and train model
            self.model.fit(x_train, y_train, verbose=False, eval_set=[(x_val, y_val)])
            # Predict and evaluate model quality
            y_predictions = self.model.predict(x_val)
            mse = mean_squared_error(y_predictions, y_val)
            acc = accuracy_score(y_val, y_predictions)
            report = classification_report(y_true=y_val, y_pred=y_predictions)
            print("Mean squared error: " + str(mse))
            print("Model accuracy: " + str(acc))
            print("Classification report:\n " + str(report))
            mse_results.append(mse)
            acc_results.append(acc)
            reports.append(report)
            xgb_models.append(self.model)
            end_time = time.time()
            print("Elapsed time: " + str(round(end_time - start_time, 2)) + " seconds")
            counter += 1

        return CrossValidationResultBoW(mse_results=mse_results, xgb_models=xgb_models, acc_results=acc_results,
                                        reports=reports, train_val_columns=train_val_columns)
