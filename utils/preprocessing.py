import abc
from abc import ABC


class Preprocessing(ABC):
    """
    Abstract class that contains variables and methods for preprocessing of input data
    """
    def __init__(self, df, target_col):
        """
        :param df: train, test or validation dataframe
        :param target_col: column that will be predicted
        """
        self.df = df
        self.target_col = target_col

    @abc.abstractmethod
    def start(self):
        """
        Starts preprocessing pipeline
        :return: preprocessed feature dataframe x and target column y
        """
        return
