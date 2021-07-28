"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import time
import os

if __name__ == "__main__":
    regressors = ['Khiops']
    split_method = 'equal_freq'
    output_classes = 'below_threshold'
    log_lvl = 'warning'
    n_bins = [2, 4, 8, 16, 32]
    n_jobs = 16
    grid_search = 'True'

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics')) if f.is_dir()]
    datasets_names = sorted([dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories])

    cmd_list = []
    for dataset_name in datasets_names:
        cmd_list.append("python visualisation_fused.py --parent_folder=\"{}\" --metric=\"RMSE\"".format(os.path.join('..', 'data', 'metrics', dataset_name)))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
