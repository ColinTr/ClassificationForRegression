"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from . import CustomClassGenerator
import numpy as np


class BelowThresholdClassGenerator(CustomClassGenerator.CustomClassGenerator):
    """
    Associates a class to the values of a goal variable based on a threshold_list.
    """

    def __init__(self):
        super(BelowThresholdClassGenerator, self).__init__()
        self.thresholds_list = None

    def fit(self, thresholds_list):
        self.thresholds_list = thresholds_list

    def transform(self, Y_train, Y_test):
        train_discretized_classes = self.below_threshold_class_gen(Y_train)
        test_discretized_classes = self.below_threshold_class_gen(Y_test)
        return train_discretized_classes, test_discretized_classes

    def below_threshold_class_gen(self, Y):
        """
        Associate a class to the values of Y based on the threshold_list.
        (number_of_thresholds - 1) classes will be created.
        Each class is associated to a threshold and for each value of Y, if it is below the threshold of a class,
        1 will be put, 0 otherwise.
        :param Y: np.array or list : The goal variable
        :return: list[list[int]] : The classes
        """
        classes = np.zeros((len(Y), len(self.thresholds_list)), dtype=int)
        for threshold_index, threshold in zip(range(len(self.thresholds_list)), self.thresholds_list):
            for i in range(len(Y)):
                if Y[i] < threshold:
                    classes[i][threshold_index] = 1

        return classes
