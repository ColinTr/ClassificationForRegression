"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from abc import ABC, abstractmethod


class CustomClassGenerator(ABC):
    """
    TODO
    """

    @abstractmethod
    def fit(self, thresholds_list):
        """
        TODO
        """
        pass

    @abstractmethod
    def transform(self, Y_train, Y_test):
        """
        TODO
        """
        pass