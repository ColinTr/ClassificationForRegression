"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import os

if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'predictions')) if f.is_dir()]
    datasets_names = [dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    regressors = ['RandomForest', 'XGBoost', 'DecisionTree', 'LinearRegression']
    split_method = 'equal_freq'
    output_classes = 'below_threshold'
    log_lvl = 'warning'
    grid_search = 'True'
    n_jobs = 16
    cmd_list = []

    for dataset_name in datasets_names:
        for regressor in regressors:
            cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/processed/{}/2_bins_{}_{}/\" --regressor=\"{}\" --log_lvl=\"{}\" --grid_search=\"{}\" --n_jobs={}"
                            .format(dataset_name, split_method, output_classes, regressor, log_lvl, grid_search, n_jobs))
