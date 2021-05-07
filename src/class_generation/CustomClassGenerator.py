"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from abc import ABC, abstractmethod


class CustomClassGenerator(ABC):
    @abstractmethod
    def fit(self, thresholds_list):
        pass

    @abstractmethod
    def transform(self, Y_train, Y_test):
        pass
