"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from . import StepsEncoder
import numpy as np
import logging


def put_first_item_from_list_to_other_list(list_to_remove_element_from, list_to_add_element_into):
    """

    :param list_to_remove_element_from:
    :param list_to_add_element_into:
    :return:
    """
    tmp_value = list_to_remove_element_from[0]
    list_to_remove_element_from = np.delete(list_to_remove_element_from, 0)
    list_to_add_element_into.append(tmp_value)
    return list_to_remove_element_from, list_to_add_element_into


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

        # Simple buggy version that doesn't handle overlapping edges
        # split_sorted_Y = np.array_split(sorted(np.array(Y).ravel()), n_bins)

        values_stock = sorted(np.concatenate(Y).ravel())  # We will remove values one by one from this list
        split_sorted_Y = []
        bins_left = n_bins
        ideal_bin_length = int(len(values_stock) / n_bins)
        switch_cond = True  # True = Allow to go above ideal_bin_length, False don't allow

        for bin_number in range(n_bins):
            split_sorted_Y.append([])

            # While we are not exceeding the size of the ideal bin and we still have values to add...
            while len(split_sorted_Y[bin_number]) < ideal_bin_length and len(values_stock) > 0:
                # Take the first element from values_stock and add it to the current bin
                values_stock, split_sorted_Y[bin_number] = put_first_item_from_list_to_other_list(values_stock, split_sorted_Y[bin_number])

                # Then count the number of values from the stock that are equal to the current last value
                equal_data_index = 0
                while len(values_stock) > equal_data_index and values_stock[equal_data_index] == split_sorted_Y[bin_number][-1]:
                    equal_data_index += 1

                # If there are identical values...
                if equal_data_index > 0:
                    # Either switch_cond is True and we will add them to the current bin, disregarding if its size will exceed the ideal_bin_length
                    # Either switch_cond is False and we will only add them if it doesn't make the size of the current bin exceed the ideal_bin_length
                    # And the last possible case is when switch_cond is False and the number of equal value data is greater than the ideal bin length
                    if (switch_cond == True) or (switch_cond == False and (len(split_sorted_Y[bin_number]) + equal_data_index) <= ideal_bin_length) or (equal_data_index >= ideal_bin_length):
                        while equal_data_index > 0:
                            values_stock, split_sorted_Y[bin_number] = put_first_item_from_list_to_other_list(values_stock, split_sorted_Y[bin_number])
                            equal_data_index -= 1
                        switch_cond = not switch_cond  # Inverse switch_cond

            bins_left -= 1

        # Since ideal_bin_length is rounded to the floor value, we might still have values left in the stock
        # So we simply add the remaining values to the last bin
        while len(values_stock) > 0:
            values_stock, split_sorted_Y[-1] = put_first_item_from_list_to_other_list(values_stock, split_sorted_Y[-1])

        empty_bins = 0
        for bin_number in range(n_bins):
            if len(split_sorted_Y[bin_number]) == 0:
                empty_bins += 1
        if empty_bins > 0:
            logging.warning(str(empty_bins) + " bin" + ('s are' if empty_bins > 1 else ' is') + " empty because of the bin generation strategy.")

        # We can now define the thresholds as the mean between the last element of a list and the first of the next list
        for index in range(n_bins):
            if index < len(split_sorted_Y) - 1 and len(split_sorted_Y[index + 1]) > 0:
                threshold = (split_sorted_Y[index][-1] + split_sorted_Y[index + 1][0]) / 2.0
                thresholds_list.append(threshold)

        return thresholds_list
