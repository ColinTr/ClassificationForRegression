"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import numpy as np
import random


def one_hot_encode():
    # TODO
    return None


def equal_width_split(Y, n_bins):
    """
    Create thresholds to split the goal variable into n_bins equally wide bins,
    distributed between the min and the max of the goal variable.
    :param Y: np.array or list : The goal variable
    :param n_bins: int : The number of bins to create
    :return: list[int] : The (n_bins - 1) thresholds
    """
    Y_min = np.min(Y)
    Y_max = np.max(Y)
    step = (Y_max - Y_min) / n_bins

    thresholds_list = [(Y_min + index * step) for index in range(1, n_bins)]

    return thresholds_list


def equal_freq_split(Y, n_bins):
    """
    Create thresholds to split the goal variable into n_bins containing the same number of goal variables.
    :param Y: np.array or list : The goal variable
    :param n_bins: int : The number of bins to create
    :return: list[int] : The (n_bins - 1) thresholds
    """
    thresholds_list = []

    split_sorted_Y = np.array_split(sorted(Y), n_bins)

    # We define the thresholds as the mean between the last element of a list and the first of the next list
    for index in range(len(split_sorted_Y)):
        if index < len(split_sorted_Y) - 1:  # We wont have to compute a threshold for the last bin
            threshold = (split_sorted_Y[index][-1] + split_sorted_Y[index + 1][0]) / 2
            thresholds_list.append(threshold)

    return thresholds_list


def below_threshold_class_gen(Y, thresholds_list):
    """
    Associate a class to the values of Y based on the threshold_list.
    (number_of_thresholds - 1) classes will be created.
    Each class is associated to a threshold and for each value of Y, if it is below the threshold of a class,
    1 will be put, 0 otherwise.
    :param Y: np.array or list : The goal variable
    :param thresholds_list: list[int] : The thresholds
    :return: list[list[int]] : The classes
    """
    classes = np.zeros((len(Y), len(thresholds_list)), dtype=int)
    for threshold_index, threshold in zip(range(len(thresholds_list)), thresholds_list):
        for i in range(len(Y)):
            if Y[i] < threshold:
                classes[i][threshold_index] = 1

    return classes


def inside_bin_class_gen(Y, thresholds_list):
    """
    Associate a class to the values of Y based on the threshold_list.
    For each value of Y, the position inside the pairs of thresholds will be defined as the class number.
    Note that the min and max value of Y consist the first element of the pair
    and the last element of the last pair respectively
    :param Y: np.array or list : The goal variable
    :param thresholds_list: list[int] : The thresholds
    :return: list[int] : The classes
    """

    thresholds_pairs = [[np.min(Y), thresholds_list[0]]]
    for threshold_index in range(0, len(thresholds_list) - 1):
        thresholds_pairs.append([thresholds_list[threshold_index], thresholds_list[threshold_index + 1]])
    thresholds_pairs.append([thresholds_list[-1], np.max(Y)])

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


def kfold_train_test_split(n_indexes, k_folds, seed=None):
    """
    Generate indexes and randomly puts them in k_folds lists.
    :param n_indexes: int : The number of indexes to use
    :param k_folds: int : The number of lists to use
    :param seed: int : The seed for the random shuffling
    :return: list[list[int]] : The shuffled indexes inside their k_folds lists
    """
    random.seed(seed)

    indexes_list = list(range(0, n_indexes))

    random.shuffle(indexes_list)

    return np.array_split(indexes_list, k_folds)
