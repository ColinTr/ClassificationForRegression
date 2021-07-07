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
                        help='The number of bins',
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

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'cleaned')) if f.is_dir()]
    datasets_names = [dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    cmd_list = []
    for dataset_name in datasets_names:
        cmd_list.append("python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --n_bins=\"{}\" --output_classes=\"{}\" --split_method=\"{}\" --log_lvl warning".format(dataset_name, args.n_bins, output_classes, split_method))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))