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
    parser.add_argument('--regressor',
                        type=str,
                        help='The regressor model to use',
                        choices=["RandomForest", "LinearRegression", "XGBoost", "GaussianNB", "Khiops", "DecisionTree"],
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    split_method = 'equal_freq'
    output_classes = 'below_threshold'
    log_lvl = 'warning'
    n_bins = [2, 4, 8, 16, 32]
    n_jobs = 16
    grid_search = 'True'

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'predictions')) if f.is_dir()]
    datasets_names = sorted([dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories])

    cmd_list = []
    for dataset_name in datasets_names:
        bins_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'predictions', dataset_name)) if f.is_dir()]
        bins_level_directories = sorted([bins_level_directory.split(os.path.sep)[-1] for bins_level_directory in bins_level_directories])

        n_metrics_computed = 0

        for bins_level_directory in bins_level_directories:
            classifier_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'predictions', dataset_name, bins_level_directory)) if f.is_dir()]
            classifier_level_directories = sorted([classifier_level_directory.split(os.path.sep)[-1] for classifier_level_directory in classifier_level_directories])

            for classifier_level_directory in classifier_level_directories:
                regressor_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'predictions', dataset_name, bins_level_directory, classifier_level_directory)) if f.is_dir()]
                regressor_level_directories = sorted([regressor_level_directory.split(os.path.sep)[-1] for regressor_level_directory in regressor_level_directories])

                for regressor_level_directory in regressor_level_directories:
                    if regressor_level_directory.split('_')[0] == args.regressor:
                        cmd_list.append('python compute_metrics.py --predictions_folder=\"{}\"'
                                        .format(os.path.join('..', 'data', 'predictions', dataset_name, bins_level_directory, classifier_level_directory, regressor_level_directory)))
                        n_metrics_computed += 1

        if n_metrics_computed > 0:
            cmd_list.append("python visualisation.py --parent_folder=\"{}\" --metric=\"RMSE\""
                            .format(os.path.join('..', 'data', 'metrics', dataset_name)))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
