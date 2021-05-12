"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from abc import ABC, abstractmethod


class StepsEncoder(ABC):
    """
    Abstract class.
    Classes that inherit this class will generate thresholds to split the given into a number of bins.
    """

    @abstractmethod
    def generate_steps(self, Y, n_bins):
        """
        The splitting method that generates thresholds.
        """
        pass
