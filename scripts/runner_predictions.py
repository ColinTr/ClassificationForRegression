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
    Extract features for all the datasets.
    """
    args = argument_parser()

    regressors = ['RandomForest', 'LinearRegression', 'XGBoost', 'Khiops', 'DecisionTree']
    split_method = 'equal_freq'
    output_classes = 'inside_bin'
    log_lvl = 'warning'
    n_bins = [2, 4, 8, 16, 32]
    n_jobs = 16
    grid_search = 'True'

    datasets_directories = [f.path for f in os.scandir('../data/cleaned/') if f.is_dir()]
    datasets_names = [dataset_directory.split('/')[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    cmd_list = []
    for dataset_name in datasets_names:
        for regressor in regressors:
            for bins in n_bins:
                cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/processed/{}/{}_bins_{}_{}/\" --regressor=\"{}\" --log_lvl=\"{}\" --n_jobs={} --grid_search {}"
                                .format(dataset_name, bins, split_method, output_classes, regressor, log_lvl, n_jobs, grid_search))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
