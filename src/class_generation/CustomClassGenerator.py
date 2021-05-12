"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from abc import ABC, abstractmethod


class CustomClassGenerator(ABC):
    """
    Abstract class.
    Process of discretization.
    Using a list of thresholds, the user will be able to define classes for a goal variable.
    """

    @abstractmethod
    def fit(self, thresholds_list):
        """
        Fit the CustomClassGenerator the thresholds list.
        """
        pass

    @abstractmethod
    def transform(self, Y_train, Y_test):
        """
        Discretize the given goal variable (train and test) using the fitted thresholds list.
        """
        pass
