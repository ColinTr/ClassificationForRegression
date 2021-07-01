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

    split_method = 'equal_freq'
    output_classes = 'inside_bin'
    classifier = 'RandomForest'
    log_lvl = 'warning'
    n_bins = [2, 4, 8, 16, 32]
    n_jobs = 16

    datasets_directories = [f.path for f in os.scandir('../data/cleaned/') if f.is_dir()]
    datasets_names = [dataset_directory.split('/')[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    cmd_list = []
    for dataset_name in datasets_names:
        for bins in n_bins:
            cmd_list.append(
                "python feature_extraction.py --dataset_folder=\"" + os.path.join('..', 'data', 'processed', dataset_name, str(bins) + '_bins_' + str(split_method) + '_' + str(output_classes)) + "\" --classifier=\"{}\" --log_lvl=\"{}\" --n_jobs={}"
                .format(classifier, log_lvl, n_jobs))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
