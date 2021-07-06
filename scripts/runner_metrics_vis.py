"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import time
import os


if __name__ == "__main__":
    """
    Extract features for all the datasets.
    """

    regressors = ['Khiops']
    split_method = 'equal_freq'
    output_classes = 'below_threshold'
    log_lvl = 'warning'
    n_bins = [2, 4, 8, 16, 32]
    n_jobs = 16
    grid_search = 'True'

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'predictions')) if f.is_dir()]
    datasets_names = [dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    cmd_list = []
    for dataset_name in datasets_names:
        bins_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'predictions', dataset_name)) if f.is_dir()]
        bins_level_directories = [bins_level_directory.split(os.path.sep)[-1] for bins_level_directory in bins_level_directories]

        for bins_level_directory in bins_level_directories:
            classifier_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'predictions', dataset_name, bins_level_directory)) if f.is_dir()]
            classifier_level_directories = [classifier_level_directory.split(os.path.sep)[-1] for classifier_level_directory in classifier_level_directories]

            for classifier_level_directory in classifier_level_directories:
                regressor_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'predictions', dataset_name, bins_level_directory, classifier_level_directory)) if f.is_dir()]
                regressor_level_directories = [regressor_level_directory.split(os.path.sep)[-1] for regressor_level_directory in regressor_level_directories]

                for regressor_level_directory in regressor_level_directories:
                    cmd_list.append('python compute_metrics.py --predictions_folder=\"{}\"'
                                    .format(os.path.join('..', 'data', 'predictions', dataset_name, bins_level_directory, classifier_level_directory, regressor_level_directory)))

        cmd_list.append("python visualisation.py --parent_folder=\"{}\" --metric=\"RMSE\""
                                    .format(os.path.join('..', 'data', 'metrics', dataset_name)))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
