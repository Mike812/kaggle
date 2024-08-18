import abc
from abc import ABC


class Preprocessing(ABC):
    """

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

        :return:
        """
        return
