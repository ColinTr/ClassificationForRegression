"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract class.
    This is the base model from which the classifiers used to extract features will inherit.
    """

    @abstractmethod
    def fit(self, X_train, Y_train):
        """
        Fit the model to the given data
        :param X_train: dataframe or numpy array, the descriptive variables of the data.
        :param Y_train: dataframe or numpy array, the goal variable of each instance of the data.
        """
        pass

    @abstractmethod
    def extract_features(self, X, Y):
        """
        Extracts the features of the given dataset by using the model.
        :param X: dataframe or numpy array, the descriptive variables of the data.
        :param Y: dataframe or numpy array, the goal variable of each instance of the data.
        :return:
        """
        pass
