"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from . import CustomClassGenerator
import numpy as np


class InsideBinClassGenerator(CustomClassGenerator.CustomClassGenerator):
    """
    Associates a class to the values of a goal variable based on a threshold_list.
    """

    def __init__(self):
        super(InsideBinClassGenerator, self).__init__()
        self.thresholds_list = None

    def fit(self, thresholds_list):
        self.thresholds_list = thresholds_list

    def transform(self, Y_train, Y_test):
        train_discretized_classes = self.inside_bin_class_gen(Y_train)
        test_discretized_classes = self.inside_bin_class_gen(Y_test)
        return train_discretized_classes, test_discretized_classes

    def inside_bin_class_gen(self, Y):
        """
        Associate a class to the values of Y based on the threshold_list.
        For each value of Y, the position inside the pairs of thresholds will be defined as the class number.
        Note that the min and max value of Y consist the first element of the pair
        and the last element of the last pair respectively
        :param Y: np.array or list : The goal variable
        :return: list[list[int]] : The classes
        """
        Y = list(Y.reshape(-1))
        Y_max = np.max(Y)
        Y_min = np.min(Y)

        thresholds_pairs = [[Y_min, self.thresholds_list[0]]]
        for threshold_index in range(0, len(self.thresholds_list) - 1):
            thresholds_pairs.append([self.thresholds_list[threshold_index], self.thresholds_list[threshold_index + 1]])
        thresholds_pairs.append([self.thresholds_list[-1], Y_max])

        num_thresholds = len(thresholds_pairs)

        classes = []
        for value in Y:
            class_of_instance = -1
            # If the value is the maximum of the range, its class is the last one
            if value >= Y_max:
                class_of_instance = num_thresholds - 1
            elif value <= Y_min:
                class_of_instance = 0
            else:
                # Otherwise we iterate through the bins :
                for class_index in range(0, num_thresholds):
                    if thresholds_pairs[class_index][0] <= value < thresholds_pairs[class_index][1]:
                        class_of_instance = class_index

            classes.append([class_of_instance])

        # for class_nb in np.unique(classes):
        #     print(classes.count(class_nb))
        # print('===================')

        return np.array(classes)
