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


def split_path(path):
    path = os.path.dirname(path)  # Remove the file name from the path
    path = os.path.normpath(path)  # Clean the path if needed
    return path.split(os.sep)  # Split the path with the system separator ('/' or '\')


def find_index_in_list(split_path_list, element_list):
    index = None
    for element in element_list:
        try:
            index = split_path_list.index(element)
        except ValueError:
            pass
    return index
