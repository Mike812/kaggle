from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
import time

from utils.cross_validation_result import CVResultClassificationBoW
from utils.model_evaluation import ModelEvaluation


class ModelEvaluationBagOfWords(ModelEvaluation):
    """
    Consists of all methods and variables that are needed for model evaluation
    """
    def __init__(self, train_val_data, preprocessor, target_col, model, splits):
        """
        :param train_val_data: dataframe with training and validation data
        :param preprocessor: preprocessor class
        :param target_col: column that will be predicted
        :param model: machine learning model
        :param splits: number of cross validation rounds
        """
        super().__init__(train_val_data, preprocessor, target_col, model, splits)

    def cross_validate(self, transform_to_sparse=True):
        """
        :return: CrossValidationResultBoW
        """
        # Prepare cross validation of model predictions
        kf = KFold(n_splits=self.splits, shuffle=True, random_state=42)
        counter = 1
        mse_results = []
        xgb_models = []
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
            if transform_to_sparse:
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

        return CVResultClassificationBoW(mse_results=mse_results, xgb_models=xgb_models, acc_results=acc_results,
                                         reports=reports, train_val_columns=train_val_columns)
