"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.DataProcessingUtils import detect_class_columns
from src.utils.logging_util import setup_logging_level
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from os.path import isfile, join
from src.utils.Metrics import *
from os import listdir
import pandas as pd
import numpy as np
import argparse
import logging
import time
import os


def argument_parser():
    """
    A parser to allow user to easily compute many metrics on the datasets of the given folder.
    """

    parser = argparse.ArgumentParser(usage='\n python compute_test_metrics.py [dataset_folder] [regressor] [log_lvl]',
                                     description="This program allows to compute the mean metrics of a regressor"
                                                 "across all datasets named with TEST inside a folder.")

    parser.add_argument('--dataset_folder',
                        type=str,
                        help='The folder where the test and train k-fold datasets are stored',
                        required=True)

    parser.add_argument('--regressor',
                        type=str,
                        help='The regression model to use',
                        choices=["RandomForest", "LogisticRegression", "XGBoost", "GaussianNB", "Khiops"],
                        required=True)

    parser.add_argument('--log_lvl',
                        type=str,
                        default='info',
                        choices=["debug", "info", "warning"],
                        help='Change the log display level')

    return parser.parse_args()


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

        # We keep all the columns except the goal variable ones and the class ones
        class_columns_indexes, reg_goal_var_index = detect_class_columns(list(train_dataframe.columns.values))

        # It would be cheating to use the class_X columns since they are the goal variable encoded
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

    train_metrics_list = []
    test_metrics_list = []
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

        # We fit the model on the TRAINING data
        # Before predicting on both training and testing data to compute the metrics
        model = None
        if args.regressor == "RandomForest":
            model = RandomForestRegressor()
            model.fit(X_train, Y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        elif args.regressor == "LogisticRegression":
            # TODO
            pass
        elif args.regressor == "XGBoost":
            model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
            model.fit(X_train, Y_train)

            y_train_pred = model.predict(np.ascontiguousarray(X_train))
            y_test_pred = model.predict(np.ascontiguousarray(X_test))
            pass
        elif args.regressor == "GaussianNB":
            # TODO
            pass
        elif args.regressor == "Khiops":
            # TODO
            pass
        else:
            raise ValueError('Unknown parameter for regressor')

        # Compute the metrics
        train_metrics = compute_all_metrics(y_train_pred, Y_train, n=len(Y_train), p=X_train.shape[1])
        test_metrics = compute_all_metrics(y_test_pred, Y_test, n=len(Y_test), p=X_test.shape[1])

        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)
        logging.info('Split ' + X_train_key + ' R² score : train = {0:.2f}'.format(train_metrics["r_squared"]) +
                     ' & test = {0:.2f}'.format(test_metrics["r_squared"]))

    logging.info('Mean R² score : train =  {0:.4f}'.format(
        np.mean([metrics_dict['r_squared'] for metrics_dict in train_metrics_list]))
                 + ' & test =  {0:.4f}'.format(
        np.mean([metrics_dict['r_squared'] for metrics_dict in test_metrics_list])))
