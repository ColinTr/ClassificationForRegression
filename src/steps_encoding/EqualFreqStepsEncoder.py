"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from . import StepsEncoder
import numpy as np


class EqualFreqStepsEncoder(StepsEncoder.StepsEncoder):
    """
    Generates thresholds to split the goal variable into n_bins containing the same number of goal variables.
    """

    def __init__(self):
        super(EqualFreqStepsEncoder, self).__init__()
        self.thresholds_list = None

    def generate_steps(self, Y, n_bins):
        """
        Create thresholds to split the goal variable into n_bins containing the same number of goal variables.
        :param Y: np.array or list : The goal variable
        :param n_bins: int : The number of bins to create
        :return: list[int] : The (n_bins - 1) thresholds
        """
        thresholds_list = []

        # IMPORTANT : Now using only the UNIQUE values to define the thresholds
        # When many data had the same value, some thresholds could fall between these same value data
        #   which would result in classes never showing up
        split_sorted_Y = np.array_split(sorted(np.unique(Y)), n_bins)

        # We define the thresholds as the mean between the last element of a list and the first of the next list
        for index in range(len(split_sorted_Y)):
            if index < len(split_sorted_Y) - 1:  # We wont have to compute a threshold for the last bin
                threshold = split_sorted_Y[index][-1] + split_sorted_Y[index + 1][0] / 2.0
                thresholds_list.append(threshold)

        return thresholds_list
