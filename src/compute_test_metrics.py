"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from utils.logging_util import setup_logging_level
from sklearn.ensemble import RandomForestRegressor
from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np
import argparse
import logging
import time
import os


def argument_parser():
    """
    TODO
    """

    parser = argparse.ArgumentParser(usage='\n python3 compute_test_metrics.py [dataset_folder] |regressor] [log_lvl]'
                                           '\n Example : python scripts/compute_test_metrics.py TODO...',
                                     description="This program allows to compute the mean metrics of a regressor"
                                                 "across all datasets named with TEST inside a folder.")

    parser.add_argument('--dataset_folder',
                        type=str,
                        help='The folder where the TEST k-fold datasets are stored',
                        required=True)

    parser.add_argument('--regressor',
                        type=str,
                        help='The regression model to use',
                        choices=["RandomForests", "LogisticRegression", "XGBoost", "GaussianNB", "Khiops"],
                        required=True)

    parser.add_argument('--log_lvl',
                        type=str,
                        default='warning',
                        choices=["debug", "info", "warning"],
                        help='Change the log display level')

    return parser.parse_args()


def detect_class_columns(header):
    col_index = 0
    regression_goal_var_index = -1
    class_cols_indexes = []
    for column_name in header:
        if column_name.split('_')[0] == 'class':
            class_cols_indexes.append(col_index)
        elif column_name == 'reg_goal_var':
            regression_goal_var_index = col_index
        col_index = col_index + 1
    return class_cols_indexes, regression_goal_var_index


if __name__ == "__main__":
    args = argument_parser()

    dataset_folder = args.dataset_folder

    # Setup the logging level
    setup_logging_level(args.log_lvl)

    directory_files = [f for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]

    X_train_datasets = {}
    Y_train_datasets = {}
    X_test_datasets = {}
    Y_test_datasets = {}
    for filename in directory_files:
        logging.debug('Reading file : ' + os.path.join(dataset_folder, filename) + '...')
        reading_start_time = time.time()
        train_dataframe = pd.read_csv(os.path.join(dataset_folder, filename))
        logging.debug("Dataset imported ({0:.2f}".format(time.time() - reading_start_time) + "sec)")

        # We keep all the columns except the goal variable ones
        class_columns_indexes, reg_goal_var_index = detect_class_columns(list(train_dataframe.columns.values))
        X_cols_to_drop = class_columns_indexes.copy()
        X_cols_to_drop.append(reg_goal_var_index)

        X = train_dataframe.drop(train_dataframe.columns[X_cols_to_drop], axis=1)
        Y = train_dataframe[train_dataframe.columns[reg_goal_var_index]]  # This time, Y is the regression goal variable

        logging.debug("Dataset's first 3 rows :")
        logging.debug('X :\n' + str(X.head(3)))
        logging.debug('Y :\n' + str(Y.head(3)))

        if 'Extended' in filename.split('_'):
            fold_num = filename.split('_')[2]
        else:
            fold_num = filename.split('_')[1]

        if 'TRAIN' in filename.split('_'):
            X_train_datasets[fold_num] = X
            Y_train_datasets[fold_num] = Y
        elif 'TEST' in filename.split('_'):
            X_test_datasets[fold_num] = X
            Y_test_datasets[fold_num] = Y
        else:
            raise ValueError('Unsupported file found : ' + filename)

    # Safety checks
    if len(X_train_datasets.keys()) == 0:
        raise ValueError('No TRAIN dataset found in train_folder')
    if len(X_train_datasets.keys()) != len(X_test_datasets.keys()):
        raise ValueError('Train and test number of files don\t match')

    test_accuracies = []
    train_accuracies = []
    for X_train_key, Y_train_key, X_test_key, Y_test_key in zip(sorted(X_train_datasets.keys()),
                                                                sorted(Y_train_datasets.keys()),
                                                                sorted(X_test_datasets.keys()),
                                                                sorted(Y_test_datasets.keys())):
        X_train = X_train_datasets[X_train_key]
        Y_train = Y_train_datasets[Y_train_key]
        X_test = X_test_datasets[X_test_key]
        Y_test = Y_test_datasets[Y_test_key]

        logging.debug("X_train :")
        logging.debug('\n' + str(X_train.head(3)))
        logging.debug("Y_train :")
        logging.debug('\n' + str(Y_train.head(3)))
        logging.debug("X_test:")
        logging.debug('\n' + str(X_test.head(3)))
        logging.debug("Y_test :")
        logging.debug('\n' + str(Y_test.head(3)))

        if args.regressor == "RandomForests":
            model = RandomForestRegressor()

            # Fit the model on the TRAINING data
            model.fit(X_train, Y_train)

            # Compute the metrics
            test_accuracy = model.score(X_test, Y_test)
            train_accuracy = model.score(X_train, Y_train)
            test_accuracies.append(test_accuracy)
            train_accuracies.append(train_accuracy)
            logging.info('Split ' + X_train_key + ' accuracy : train = {0:.2f}'.format(train_accuracy) + ' & test = {0:.2f}'.format(test_accuracy))
        elif args.regressor == "LogisticRegression":
            # TODO
            pass
        elif args.regressor == "XGBoost":
            # TODO
            pass
        elif args.regressor == "GaussianNB":
            # TODO
            pass
        elif args.regressor == "Khiops":
            # TODO
            pass

    logging.info('Mean accuracy : train =  {0:.4f}'.format(np.mean(train_accuracies)) + ' & test =  {0:.4f}'.format(np.mean(test_accuracies)))
