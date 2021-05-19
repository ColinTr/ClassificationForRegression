"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import logging
import os


def setup_logging_level(level):
    if level == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    elif level == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif level == 'warning':
        logging.getLogger().setLevel(logging.WARNING)
    else:
        raise ValueError('Unknown parameter for log_lvl.')


def find_index_in_list(split_path_list, element_list):
    index = None
    for element in element_list:
        try:
            index = split_path_list.index(element)
        except ValueError:
            pass
    return index


def generate_output_path(input_path, elements_to_replace, new_element):
    path = os.path.dirname(input_path)  # Remove the file name from the path
    path = os.path.normpath(path)  # Clean the path if needed
    split_path = path.split(os.sep)  # Split the path with the system separator ('/' or '\')

    index_to_replace = find_index_in_list(split_path, elements_to_replace)
    if index_to_replace is None:
        raise ValueError('Unable to generate an output path, please define explicitly the parameter --output_path')

    split_path[index_to_replace] = new_element

    output_path = os.path.join(*split_path)

    return output_path
