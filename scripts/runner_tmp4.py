"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import os
import time

if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """

    split_method = 'equal_freq'
    output_classes = 'below_threshold'
    models = ['RandomForest', 'XGBoost', 'DecisionTree']
    log_lvl = 'warning'
    use_hyperparam_file = 'True'
    n_jobs = 16

    datasets_directories = [f.path for f in os.scandir('../data/cleaned/') if f.is_dir()]
    datasets_names = [dataset_directory.split('/')[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    cmd_list = []
    for dataset_name in datasets_names:
        # cmd_list.append("python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --n_bins=\"{}\" --output_classes=\"{}\" --split_method=\"{}\" --log_lvl=\"{}\"".format(dataset_name, 2, output_classes, split_method, log_lvl))
        for model in models:
            cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/processed/{}/{}_bins_{}_{}/\" --regressor=\"{}\" --log_lvl=\"{}\" --n_jobs={} --use_hyperparam_file {}".format(dataset_name, 2, split_method, output_classes, model, log_lvl, n_jobs, use_hyperparam_file))
            cmd_list.append("python compute_metrics.py  --predictions_folder=\"../data/predictions/{}/{}_bins_{}_{}/Standard/{}_regressor\" --log_lvl=\"{}\"".format(dataset_name, 2, split_method, output_classes, model, 'info'))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
