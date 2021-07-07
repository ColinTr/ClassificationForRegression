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

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'cleaned')) if f.is_dir()]
    datasets_names = [dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    cmd_list = []
    for dataset_name in datasets_names:
        for regressor in regressors:
            for bins in n_bins:
                cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/{}_bins_{}_{}/{}/\" --regressor=\"{}\" --log_lvl=\"{}\" --n_jobs={} --grid_search {}"
                                .format(dataset_name, bins, split_method, output_classes, 'RandomForest_classifier', regressor, log_lvl, n_jobs, grid_search))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
