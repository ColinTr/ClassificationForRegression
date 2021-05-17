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
import gc


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

    train_filename_list = [e for e in directory_files if 'TRAIN' in e.split('_')]
    train_filename_list.sort()
    test_filename_list = [e for e in directory_files if 'TEST' in e.split('_')]
    test_filename_list.sort()

    # Safety checks
    if len(train_filename_list) == 0:
        raise ValueError('No TRAIN dataset found in train_folder')
    if len(train_filename_list) != len(test_filename_list):
        raise ValueError('Train and test number of files don\'t match')

    train_metrics_list = []
    test_metrics_list = []

    for train_filename, test_filename in zip(train_filename_list, test_filename_list):
        # Get the fold_num for pretty logging
        if 'Extended' in train_filename.split('_'):
            fold_num = train_filename.split('_')[2]
            test_fold_num = test_filename.split('_')[2]
        else:
            fold_num = train_filename.split('_')[1]
            test_fold_num = test_filename.split('_')[1]

        if fold_num != test_fold_num:
            raise ValueError('Train and test files number don\'t match')

        logging.debug('Reading training file : ' + os.path.join(dataset_folder, train_filename) + '...')
        reading_start_time = time.time()
        train_dataframe = pd.read_csv(os.path.join(dataset_folder, train_filename))
        logging.debug("Dataset imported ({0:.2f}".format(time.time() - reading_start_time) + "sec)")

        # We keep all the columns except the goal variable ones and the class ones
        class_columns_indexes, reg_goal_var_index = detect_class_columns(list(train_dataframe.columns.values))

        # It would be cheating to use the class_X columns since they are the goal variable encoded
        X_cols_to_drop = class_columns_indexes.copy()
        X_cols_to_drop.append(reg_goal_var_index)

        X_train = train_dataframe.drop(train_dataframe.columns[X_cols_to_drop], axis=1)
        # This time, Y is the regression goal variable
        Y_train = train_dataframe[train_dataframe.columns[reg_goal_var_index]]

        logging.debug("Train dataset's first 3 rows :")
        logging.debug('X :\n' + str(X_train.head(3)))
        logging.debug('Y :\n' + str(Y_train.head(3)))

        logging.debug('Reading testing file : ' + os.path.join(dataset_folder, test_filename) + '...')
        reading_start_time = time.time()
        test_dataframe = pd.read_csv(os.path.join(dataset_folder, test_filename))
        logging.debug("Dataset imported ({0:.2f}".format(time.time() - reading_start_time) + "sec)")

        X_test = test_dataframe.drop(test_dataframe.columns[X_cols_to_drop], axis=1)
        # This time, Y is the regression goal variable
        Y_test = test_dataframe[test_dataframe.columns[reg_goal_var_index]]

        logging.debug("Test dataset's first 3 rows :")
        logging.debug('X :\n' + str(X_test.head(3)))
        logging.debug('Y :\n' + str(Y_test.head(3)))

        # We fit the model on the TRAINING data
        # Before predicting on both training and testing data to compute the metrics
        model = None
        if args.regressor == "RandomForest":
            model = RandomForestRegressor(n_jobs=-1)
            model.fit(X_train, Y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        elif args.regressor == "LogisticRegression":
            # TODO
            pass
        elif args.regressor == "XGBoost":
            model = XGBRegressor(n_jobs=-1, n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
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
        logging.info('Split ' + fold_num + ' R² score : train = {0:.2f}'.format(train_metrics["r_squared"]) +
                     ' & test = {0:.2f}'.format(test_metrics["r_squared"]))

        # Expressly free the variables from the memory
        del train_dataframe, test_dataframe, X_train, X_test, Y_train, Y_test, y_train_pred, y_test_pred

        # Call python's garbage collector
        gc.collect()

    logging.info('Mean R² score : train =  {0:.4f}'.format(
        np.mean([metrics_dict['r_squared'] for metrics_dict in train_metrics_list]))
                 + ' & test =  {0:.4f}'.format(
        np.mean([metrics_dict['r_squared'] for metrics_dict in test_metrics_list])))
