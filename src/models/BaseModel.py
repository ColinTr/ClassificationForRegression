"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    TODO
    """

    @abstractmethod
    def fit(self, X_train, Y_train):
        """
        TODO
        """
        pass

    @abstractmethod
    def extract_features(self, X_test, Y_test):
        """
        TODO
        """
        pass
