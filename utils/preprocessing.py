import abc
import string
from abc import ABC
import re


def prepare_text_with_regex(text):
    """
    Prepare text with multiple regex expressions.
    Copied from https://www.kaggle.com/code/muhammadfaizan65/sentiment-analysis-for-mental-health-nlp
    :param text: string
    :return: prepared text
    """
    text = text.lower()  # Lowercase text
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text.strip()


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
