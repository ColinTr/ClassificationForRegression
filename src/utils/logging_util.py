"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import logging
import os


def setup_logging_level(level):
    """
    Sets up the logging level.
    :param level: String, the logging level.
    """
    if level == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    elif level == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif level == 'warning':
        logging.getLogger().setLevel(logging.WARNING)
    else:
        raise ValueError('Unknown parameter for log_lvl.')


def find_index_in_list(split_path_list, element_list):
    """
    Returns the index of any element of a list inside another list.
    :param split_path_list: The list where we want to find an item.
    :param element_list: The list of items to test.
    :return: Index of the element or None if not found.
    """
    index = None
    for element in element_list:
        try:
            index = split_path_list.index(element)
        except ValueError:
            pass
    return index


def generate_output_path(input_path, elements_to_replace, new_element):
    """
    Inside a String path, replace one of the given possible elements by the new element.
    :param input_path: String, the input path to modify.
    :param elements_to_replace: A list of elements that can be replaced.
    :param new_element: The new element to use.
    :return: The modified path.
    """
    if os.path.isfile(input_path):
        input_path = os.path.dirname(input_path)  # Remove the file name from the path
    path = os.path.normpath(input_path)  # Clean the path if needed
    split_path = path.split(os.sep)  # Split the path with the system separator ('/' or '\')

    index_to_replace = find_index_in_list(split_path, elements_to_replace)
    if index_to_replace is None:
        raise ValueError('Unable to generate an output path, please define explicitly the parameter --output_path')

    split_path[index_to_replace] = new_element

    output_path = os.path.join(*split_path)

    return output_path
