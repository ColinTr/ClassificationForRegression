"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import pandas as pd
import numpy as np
import argparse
import logging
import time
import sys
import os
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.DataProcessingUtils import get_real_class_predicted_probas
from src.utils.DataProcessingUtils import detect_class_columns
from src.utils.logging_util import generate_output_path
from src.utils.logging_util import setup_logging_level
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from os.path import isfile, join
from xgboost import XGBRegressor
from os import listdir


def argument_parser():
    """
    A parser to allow user to easily compute many metrics on the datasets of the given folder.
    """

    parser = argparse.ArgumentParser(usage='\n python generate_predictions.py [dataset_folder] [output_path]'
                                           '[regressor] [log_lvl]',
                                     description="This program allows to compute the mean metrics of a regressor"
                                                 "across all datasets named with TEST inside a folder.")

    parser.add_argument('--dataset_folder',
                        type=str,
                        help='The folder where the test and train k-fold datasets are stored',
                        required=True)

    parser.add_argument('--output_path',
                        type=str,
                        help='The folder where the results will be saved')

    parser.add_argument('--regressor',
                        type=str,
                        help='The regression model to use',
                        choices=["RandomForest", "LinearRegression", "XGBoost", "GaussianNB", "Khiops"],
                        required=True)

    parser.add_argument('--log_lvl',
                        type=str,
                        default='info',
                        choices=["debug", "info", "warning"],
                        help='Change the log display level')

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    # Setup the logging level
    setup_logging_level(args.log_lvl)

    dataset_folder = args.dataset_folder
    output_path = args.output_path

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

    if 'Extended' in train_filename_list[0].split('_'):
        extended = True
    else:
        extended = False

    # If no value was given for the 'output_path', we will generate it automatically
    if output_path is None:
        output_path = generate_output_path(dataset_folder, ['processed', 'extracted_features'], 'predictions')

        if not extended:
            output_path = os.path.join(output_path, 'Standard')

        # Add the regressor name to the path so we can distinguish different predictions
        output_path = os.path.join(output_path, args.regressor + '_regressor')

        logging.info('Generated output path : ' + output_path)

    train_predictions_list, test_predictions_list = [],  []

    for train_filename, test_filename in zip(train_filename_list, test_filename_list):
        # Get the fold_num for pretty logging
        if extended:
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
        logging.debug('X_train :\n' + str(X_train.head(3)))
        logging.debug('Y_train :\n' + str(Y_train.head(3)))

        logging.debug('Reading testing file : ' + os.path.join(dataset_folder, test_filename) + '...')
        reading_start_time = time.time()
        test_dataframe = pd.read_csv(os.path.join(dataset_folder, test_filename))
        logging.debug("Dataset imported ({0:.2f}".format(time.time() - reading_start_time) + "sec)")

        X_test = test_dataframe.drop(test_dataframe.columns[X_cols_to_drop], axis=1)
        # This time, Y is the regression goal variable
        Y_test = test_dataframe[test_dataframe.columns[reg_goal_var_index]]

        logging.debug("Test dataset's first 3 rows :")
        logging.debug('X_test :\n' + str(X_test.head(3)))
        logging.debug('Y_test :\n' + str(Y_test.head(3)))

        # We also extract the predicted probability of the real class of every classifier
        #     so we can compute the accuracy of the classifiers later
        train_df_predicted_probas, test_df_predicted_probas = None, None
        if extended:
            train_df_predicted_probas = get_real_class_predicted_probas(train_dataframe)
            test_df_predicted_probas = get_real_class_predicted_probas(test_dataframe)

        # We fit the model on the TRAINING data
        # Before predicting on both training and testing data to compute the metrics
        model, Y_train_pred, Y_test_pred = None, None, None
        if args.regressor == "RandomForest":
            model = RandomForestRegressor(n_jobs=-1)
            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)
        elif args.regressor == "LinearRegression":
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

            # print('Test R² : ' + str(r2_score(Y_train, Y_train_pred)))
        elif args.regressor == "XGBoost":
            model = XGBRegressor(n_jobs=-1, n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(np.ascontiguousarray(X_train))
            Y_test_pred = model.predict(np.ascontiguousarray(X_test))
            pass
        elif args.regressor == "GaussianNB":
            # TODO
            raise ValueError('This regressor hasn\'t been implemented yet')
        elif args.regressor == "Khiops":
            # TODO
            raise ValueError('This regressor hasn\'t been implemented yet')
        else:
            raise ValueError('Unknown parameter for regressor')

        # We need to save the number of attributes of X for future metrics computation
        tmp_array = np.zeros(len(Y_train))
        tmp_array[:] = X_train.shape[1]

        train_prediction_dataset = pd.DataFrame({'Y_train_pred': Y_train_pred,
                                                 'Y_train': Y_train,
                                                 'X_nb_attributes': tmp_array})
        test_prediction_dataset = pd.DataFrame({'Y_test_pred': Y_test_pred,
                                                'Y_test': Y_test})

        if extended:
            train_prediction_dataset = pd.concat([train_prediction_dataset, train_df_predicted_probas], axis=1)
            test_prediction_dataset = pd.concat([test_prediction_dataset, test_df_predicted_probas], axis=1)

        # Save the extended datasets in a CSV file
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        train_prediction_dataset.to_csv(path_or_buf=os.path.join(output_path, train_filename), index=False)
        test_prediction_dataset.to_csv(path_or_buf=os.path.join(output_path, test_filename), index=False)

        logging.info("Split " + str(fold_num) + " predictions predictions saved.")

        # Expressly free the variables from the memory
        del train_dataframe, test_dataframe, X_train, X_test, Y_train, Y_test, Y_train_pred, Y_test_pred

        # Call python's garbage collector
        gc.collect()

    logging.info('All predictions saved in folder ' + output_path + ' !')
