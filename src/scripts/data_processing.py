"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import argparse


def argument_parser():
    parser = argparse.ArgumentParser(usage='\n python3 data_processing.py [dataset_path] [output_path] [split_method] '
                                           '[output_classes]',
                                     description="This program allows to process datasets to be used later.")

    parser.add_argument('--dataset_path',
                        type=str,
                        help='The dataset to process')

    parser.add_argument('--output_path',
                        type=str,
                        default='../data/processed/',
                        help='The folder where the result will be written')

    parser.add_argument('--split_method',
                        type=str,
                        default="equal_width",
                        choices=["equal_width", "equal_fred", "kmeans"],
                        help='The splitting method to use')

    parser.add_argument('--output_classes',
                        type=str,
                        default="below_threshold",
                        choices=["below_threshold", "inside_bin"],
                        help='The method of class generation')

    return parser.parse_args()


if __name__ == "__main__":
    """
    """
    args = argument_parser()

    pass
