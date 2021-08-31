"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import argparse
import time
import os


def argument_parser():
    """
    A parser to allow user to easily generate the results of any number of classifiers using our method.
    """

    parser = argparse.ArgumentParser(usage='\n python runner.py [dataset_name] [goal_index] [classifiers]+',
                                     description="This program allows to run all the scripts necessary to generate the"
                                                 " final figures on any number of classifiers for a given dataset.")

    parser.add_argument('--n_bins',
                        type=int,
                        nargs='+',
                        help='The number of bins',
                        required=True)

    parser.add_argument('--dataset_name',
                        type=str,
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    """
    Pre-processes all the datasets.
    """
    args = argument_parser()

    split_method = 'equal_freq'
    output_classes = 'below_threshold'
    n_jobs = 16

    cmd_list = []
    for bins in args.n_bins:
        cmd_list.append("python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --n_bins=\"{}\" --output_classes=\"{}\" --split_method=\"{}\" --log_lvl info".format(args.dataset_name, bins, output_classes, split_method))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))