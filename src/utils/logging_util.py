"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import logging


def setup_logging_level(level):
    if level == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    elif level == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif level == 'warning':
        logging.getLogger().setLevel(logging.WARNING)
    else:
        raise ValueError('Unknown parameter for log_lvl.')
