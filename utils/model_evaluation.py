from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import KFold

from utils.cross_validation_result import CrossValidationResult


class ModelEvaluation:
    """
    Consists of all methods and variables that are needed for model evaluation
    """
    def __init__(self, train_val_data, preprocessor, model):
        """

        :param train_val_data:
        :param preprocessor:
        :param model:
        """
        self.train_val_data = train_val_data
        self.preprocessor = preprocessor
        self.model = model

    def cross_validate(self):
        """

        :return:
        """
        # Prepare cross validation of model predictions
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        counter = 1
        mse_results = []
        xgb_models = []
        acc_results = []
        reports = []
        # Start cross validation
        for train_index, val_index in (kf.split(self.train_val_data)):
            print(f'\nFold: {counter}')
            train_data = self.train_val_data.loc[train_index]
            val_data = self.train_val_data.loc[val_index]
            print(f'Train data: {train_data.shape}')
            print(f'Validation data: {train_data.shape}')
            # Start preprocessing of data needed for modeling
            x_train, y_train = self.preprocessor(df=train_data).start()
            x_val, y_val = self.preprocessor(df=val_data).start()
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
            xgb_models.append(self.model)
            reports.append(report)
            counter += 1

        return CrossValidationResult(mse_results=mse_results, xgb_models=xgb_models, acc_results=acc_results,
                                     reports=reports)
