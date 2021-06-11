"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from . import StepsEncoder
import numpy as np


class EqualWidthStepsEncoder(StepsEncoder.StepsEncoder):
    """
    Generates thresholds to split the goal variable into n_bins equally wide bins,
    distributed between the min and the max of the goal variable.
    """

    def __init__(self):
        super(EqualWidthStepsEncoder, self).__init__()
        self.thresholds_list = None

    def generate_steps(self, Y, n_bins):
        """
        Create thresholds to split the goal variable into n_bins equally wide bins,
        distributed between the min and the max of the goal variable.
        :param Y: np.array or list : The goal variable
        :param n_bins: int : The number of bins to create
        :return: list[int] : The (n_bins - 1) thresholds
        """
        Y_min = np.min(Y)
        Y_max = np.max(Y)

        step = Y_max - Y_min / n_bins

        thresholds_list = [(Y_min + index * step) for index in range(1, n_bins)]

        return thresholds_list
