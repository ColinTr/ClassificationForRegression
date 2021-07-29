"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import os


if __name__ == "__main__":
    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics')) if f.is_dir()]
    datasets_names = sorted([dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories])

    for dataset_name in datasets_names:
        os.system("python visualisation_fused.py --parent_folder {}".format(os.path.join('..', 'data', 'metrics', dataset_name)))
