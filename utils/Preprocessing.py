import abc
from abc import ABC


class Preprocessing(ABC):
    def __init__(self, df):
        self.df = df

    @abc.abstractmethod
    def start(self):
        return
