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


def apply_box_cox():
    # TODO
    return None


def standardize():
    # TODO
    return None


def equal_width_split(Y, n_bins):
    """
    TODO
    :param Y:
    :param n_bins:
    :return:
    """
    Y_min = np.min(Y)
    Y_max = np.max(Y)
    step = (Y_max - Y_min) / n_bins

    thresholds_list = [(Y_min + index * step) for index in range(1, n_bins)]

    return thresholds_list


def equal_freq_split(Y, n_bins):
    """
    TODO
    :param Y:
    :param n_bins:
    :return:
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
    TODO
    :param Y: numpy array
    :param thresholds_list:
    :return:
    """
    classes = np.zeros((len(Y), len(thresholds_list)), dtype=int)
    for threshold_index, threshold in zip(range(len(thresholds_list)), thresholds_list):
        for i in range(len(Y)):
            if Y[i] < threshold:
                classes[i][threshold_index] = 1

    return classes


def inside_bin_class_gen(Y, thresholds_list):
    """
    TODO
    :param Y:
    :param thresholds_list:
    :return:
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
    random.seed(seed)

    indexes_list = list(range(0, n_indexes))

    random.shuffle(indexes_list)

    return np.array_split(indexes_list, k_folds)
