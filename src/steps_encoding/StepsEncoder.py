"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from abc import ABC, abstractmethod


class StepsEncoder(ABC):
    """
    TODO
    """

    @abstractmethod
    def generate_steps(self, Y, n_bins):
        """
        TODO
        """
        pass
