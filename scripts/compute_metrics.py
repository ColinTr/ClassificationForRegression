"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np
import argparse
import logging
import sys
import os
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.DataProcessingUtils import compute_log_losses
from src.utils.logging_util import generate_output_path
from src.utils.logging_util import setup_logging_level
from src.utils.Metrics import compute_all_metrics
from sklearn.metrics import r2_score

def argument_parser():
    """
    A parser to allow user to easily compute many metrics on the datasets of the given folder.
    """

    parser = argparse.ArgumentParser(usage='\n python compute_metrics.py [results_folder] [output_path] [log_lvl]',
                                     description="This program allows to compute the metrics of all the"
                                                 "result files inside a folder.")

    parser.add_argument('--predictions_folder',
                        type=str,
                        help='The folder where the prediction results are stored',
                        required=True)

    parser.add_argument('--output_path',
                        type=str,
                        help='The folder where the computed metrics will be saved')

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

    predictions_folder = args.predictions_folder
    output_path = args.output_path

    # If no value was given for the 'output_path', we will generate it automatically
    if output_path is None:
        output_path = generate_output_path(predictions_folder, ['predictions'], 'metrics')
        logging.info('Generated output path : ' + output_path)

    directory_files = [f for f in listdir(predictions_folder) if isfile(join(predictions_folder, f))]

    train_predictions_filename_list = [e for e in directory_files if 'TRAIN' in e.split('_')]
    train_predictions_filename_list.sort()
    test_predictions_filename_list = [e for e in directory_files if 'TEST' in e.split('_')]
    test_predictions_filename_list.sort()

    # Safety checks
    if len(train_predictions_filename_list) == 0:
        raise ValueError('No TRAIN dataset found in train_folder')
    if len(train_predictions_filename_list) != len(test_predictions_filename_list):
        raise ValueError('Train and test number of files don\'t match')

    if 'Extended' in train_predictions_filename_list[0].split('_'):
        extended = True
    else:
        extended = False

    train_metrics_list, test_metrics_list, metrics_dataframe_list = [], [], []

    for train_filename, test_filename in zip(train_predictions_filename_list, test_predictions_filename_list):
        # Get the fold_num for pretty logging
        if extended:
            train_fold_num = train_filename.split('_')[2]
            test_fold_num = test_filename.split('_')[2]
        else:
            train_fold_num = train_filename.split('_')[1]
            test_fold_num = test_filename.split('_')[1]

        if train_fold_num != test_fold_num:
            raise ValueError('Train and test files number don\'t match')

        train_predictions_dataframe = pd.read_csv(os.path.join(predictions_folder, train_filename))
        logging.debug('TRAIN prediction file first 3 rows :\n' + str(train_predictions_dataframe.head(3)))

        test_predictions_dataframe = pd.read_csv(os.path.join(predictions_folder, test_filename))
        logging.debug('TEST prediction file first 3 rows :\n' + str(test_predictions_dataframe.head(3)))

        Y_train_pred = train_predictions_dataframe['Y_train_pred']
        Y_train = train_predictions_dataframe['Y_train']

        Y_test_pred = test_predictions_dataframe['Y_test_pred']
        Y_test = test_predictions_dataframe['Y_test']

        X_nb_attributes = train_predictions_dataframe['X_nb_attributes'][0]

        # Compute the metrics
        train_metrics = compute_all_metrics(Y_train, Y_train_pred, n=len(Y_train), p=X_nb_attributes)
        test_metrics = compute_all_metrics(Y_test, Y_test_pred, n=len(Y_test), p=X_nb_attributes)

        split_metrics_df = pd.DataFrame({'train_mean_absolute_error': [train_metrics["mean_absolute_error"]],
                                         'test_mean_absolute_error': [test_metrics["mean_absolute_error"]],
                                         'train_mean_squared_error': [train_metrics["mean_squared_error"]],
                                         'test_mean_squared_error': [test_metrics["mean_squared_error"]],
                                         'train_root_mean_squared_error': [train_metrics["root_mean_squared_error"]],
                                         'test_root_mean_squared_error': [test_metrics["root_mean_squared_error"]],
                                         'train_r_squared': [train_metrics["r_squared"]],
                                         'test_r_squared': [test_metrics["r_squared"]],
                                         'train_adjusted_r_squared': [train_metrics["adjusted_r_squared"]],
                                         'test_adjusted_r_squared': [test_metrics["adjusted_r_squared"]]})

        # Add the mean log losses of the classifiers to the metrics dataframe
        train_log_loss = compute_log_losses(train_predictions_dataframe)
        test_log_loss = compute_log_losses(test_predictions_dataframe)
        split_metrics_df = pd.concat([split_metrics_df, pd.DataFrame({'train_mean_log_loss': [train_log_loss],
                                                                      'test_mean_log_loss': [test_log_loss]})], axis=1)

        metrics_dataframe_list.append(split_metrics_df)

        logging.debug('Computed metrics : \n' + str(split_metrics_df))

        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)
        logging.info('Split ' + train_fold_num + ' R² score : train = {0:.2f}'.format(train_metrics["r_squared"]) +
                     ' & test = {0:.2f}'.format(test_metrics["r_squared"]))

        # Expressly free the variables from the memory
        del Y_train_pred, Y_train, Y_test_pred, Y_test, train_predictions_dataframe, test_predictions_dataframe

        # Call python's garbage collector
        gc.collect()

    metrics_dataframe = pd.concat(metrics_dataframe_list, ignore_index=True, axis=0)

    # Save the extended datasets in a CSV file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = None
    if extended:
        filename = 'metrics_extracted_features.csv'
    else:
        filename = 'metrics_normal.csv'

    metrics_dataframe.to_csv(path_or_buf=os.path.join(output_path, filename), index=False)

    logging.info('Mean R² score : train =  {0:.4f}'.format(np.mean(metrics_dataframe['train_r_squared']))
                 + ' & test =  {0:.4f}'.format(np.mean(metrics_dataframe['test_r_squared'])))
