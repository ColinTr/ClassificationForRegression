"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from . import CustomClassGenerator
import numpy as np


class InsideBinClassGenerator(CustomClassGenerator.CustomClassGenerator):
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
        :return: list[int] : The classes
        """
        thresholds_pairs = [[np.min(Y), self.thresholds_list[0]]]
        for threshold_index in range(0, len(self.thresholds_list) - 1):
            thresholds_pairs.append([self.thresholds_list[threshold_index], self.thresholds_list[threshold_index + 1]])
        thresholds_pairs.append([self.thresholds_list[-1], np.max(Y)])

        classes = []
        for value in np.array(Y):
            # If the value is the maximum of the range, its class is the last one
            if value == np.max(Y):
                classes.append(len(thresholds_pairs) - 1)
            else:
                # Otherwise we iterate through the bins :
                for class_index in range(0, len(thresholds_pairs)):
                    # Notice the <= for the first element, which is why we needed to check for the maximum value earlier
                    if thresholds_pairs[class_index][0] <= value < thresholds_pairs[class_index][1]:
                        classes.append(class_index)

        return classes
