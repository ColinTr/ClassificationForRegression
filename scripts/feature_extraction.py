"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from os.path import isfile, join
from os import listdir
import numpy as np
import pandas as pd
import argparse
import platform
import logging
import time
import sys
import gc
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.DataProcessingUtils import detect_class_columns
from src.models.LogisticRegressionC import LogisticRegressionC
from src.utils.logging_util import generate_output_path
from src.utils.logging_util import setup_logging_level
from src.models.DecisionTreeC import DecisionTreeC
from src.models.RandomForestC import RandomForestC
from src.models.GaussianNBC import GaussianNBC
from src.models.XGBoostC import XGBoostC

if platform.system() == "Windows":
    if os.environ.get('KhiopsHome') is not None:
        import importlib
        khiops_spec = importlib.util.find_spec("PyKhiopsC")
        if khiops_spec is not None:
            from src.models.PyKhiopsC import PyKhiopsC
else:
    if os.path.exists(os.path.join(os.environ["HOME"], "pykhiops", "lib")):
        from src.models.PyKhiopsC import PyKhiopsC


def argument_parser():
    """
    A parser to allow user to easily extract features of any folder of datasets with the classifier of his choice.
    """
    parser = argparse.ArgumentParser(usage='\n python feature_extraction.py [dataset_folder] [output_path]'
                                           '|classifier] [class_cols] [log_lvl]',
                                     description="This program allows to extract features from a dataset.")

    parser.add_argument('--dataset_folder',
                        type=str,
                        help='The folder where the k-fold datasets are stored',
                        required=True)

    parser.add_argument('--output_path',
                        type=str,
                        help='The folder where the results will be saved')

    parser.add_argument('--classifier',
                        type=str,
                        help='The classifier model to use',
                        choices=["RandomForest", "LogisticRegression", "XGBoost", "GaussianNB", "Khiops", "DecisionTree"],
                        required=True)

    parser.add_argument('--n_jobs',
                        type=int,
                        help='The number of cores to use',
                        default=-1)

    parser.add_argument('--log_lvl',
                        type=str,
                        default='info',
                        choices=["debug", "info", "warning"],
                        help='Change the log display level')

    return parser.parse_args()


def sort_dict(dictionary):
    sorted_dict = {}
    for key in sorted(dictionary.keys()):
        sorted_dict[key] = dictionary[key]
    return sorted_dict


def create_new_classifier_model(classifier_name, n_jobs):
    if classifier_name == 'RandomForest':
        return RandomForestC(n_jobs)
    elif classifier_name == 'LogisticRegression':
        return LogisticRegressionC(n_jobs)
    elif classifier_name == 'XGBoost':
        return XGBoostC(n_jobs)
    elif classifier_name == 'GaussianNB':
        return GaussianNBC()
    elif classifier_name == 'Khiops':
        return PyKhiopsC()
    elif classifier_name == 'DecisionTree':
        return DecisionTreeC()
    else:
        raise ValueError('Unknown parameter for classifier.')


if __name__ == "__main__":
    args = argument_parser()

    # Setup the logging level
    setup_logging_level(args.log_lvl)

    dataset_folder = args.dataset_folder
    output_path = args.output_path

    # If no value was given for the 'output_path', we will generate it automatically
    if output_path is None:
        output_path = generate_output_path(dataset_folder, ['processed'], 'extracted_features')

        # Add the classifier name to the path so we can distinguish different feature extractions
        output_path = os.path.join(output_path, args.classifier + '_classifier')

        logging.info('Generated output path : ' + output_path)

    directory_files = [f for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]

    # Split into test and train files and sort them by fold number
    train_files_dict = {}
    test_files_dict = {}
    for file_path in directory_files:
        fold_number = file_path.split('_')[1]
        test_or_train = file_path.split('_')[2]
        if test_or_train == "TEST":
            test_files_dict[str(fold_number)] = file_path
        elif test_or_train == "TRAIN":
            train_files_dict[str(fold_number)] = file_path
        else:
            raise ValueError('Unsupported file name : ' + file_path)

    # Sort the files dictionaries by key (which is the fold number)
    train_files_dict = sort_dict(train_files_dict)
    test_files_dict = sort_dict(test_files_dict)

    # print("train_files_dict :\n", train_files_dict)
    # print("test_files_dict :\n", test_files_dict)

    for train_key, test_key, fold_index in zip(train_files_dict.keys(), test_files_dict.keys(),
                                               range(0, len(test_files_dict.keys()))):
        if train_key != test_key:
            raise ValueError('Train and test files number don\'t match')

        logging.info('========== Fold ' + str(fold_index) + ' ==========')
        train_dataset_path = train_files_dict[train_key]
        test_dataset_path = test_files_dict[test_key]

        logging.info("Reading the dataset's train and test file...")
        reading_start_time = time.time()
        imported_train_dataset = pd.read_csv(os.path.join(dataset_folder, train_dataset_path))
        imported_test_dataset = pd.read_csv(os.path.join(dataset_folder, test_dataset_path))
        logging.info("Dataset imported ({0:.2f}".format(time.time() - reading_start_time) + "sec)")

        # We keep all the columns except the goal variable ones and the one for regression
        class_columns_indexes, reg_goal_var_index = detect_class_columns(list(imported_train_dataset.columns.values))
        X_cols_to_drop = class_columns_indexes.copy()
        X_cols_to_drop.append(reg_goal_var_index)

        # We will gradually add the extracted features to these dataframes
        train_extended_dataset = imported_train_dataset.copy()
        test_extended_dataset = imported_test_dataset.copy()

        X_train = imported_train_dataset.drop(imported_train_dataset.columns[X_cols_to_drop], axis=1)
        Y_train = imported_train_dataset[imported_train_dataset.columns[class_columns_indexes]]
        Y_train_reg_goal_var = imported_train_dataset[imported_train_dataset.columns[reg_goal_var_index]]
        X_test = imported_test_dataset.drop(imported_test_dataset.columns[X_cols_to_drop], axis=1)
        Y_test = imported_test_dataset[imported_test_dataset.columns[class_columns_indexes]]
        Y_test_reg_goal_var = imported_test_dataset[imported_test_dataset.columns[reg_goal_var_index]]

        logging.debug("Dataset's first 3 rows :")
        logging.debug('X_train :\n' + str(X_train.head(3)))
        logging.debug('Y_train :\n' + str(Y_train.head(3)))
        logging.debug('X_test :\n' + str(X_test.head(3)))
        logging.debug('Y_test :\n' + str(Y_test.head(3)))

        # For each class column, train a classifier and extract its features
        for train_column, test_column, index in zip(Y_train, Y_test, range(len(np.unique(list(Y_train.columns))))):
            classifier_model = create_new_classifier_model(args.classifier, args.n_jobs)

            # We fit the classifier on the TRAIN data
            classifier_model.fit(X_train, Y_train[train_column])

            # We the extract features with this classifier on the train AND test data
            train_extracted_features, train_score = classifier_model.extract_features(X_train, Y_train[test_column])
            test_extracted_features, test_score = classifier_model.extract_features(X_test, Y_test[test_column])
            if train_score is not None and test_score is not None:
                logging.info('model ' + str(index) + ' accuracy : train = {0:.2f}'.format(train_score) + ' & test = {0:.2f}'.format(test_score))
            else:
                logging.info('model ' + str(index) + ' finished extracting features.')

            # We can now add the extracted features to the dataframes :
            for ef_train_key in train_extracted_features.keys():
                train_extended_dataset['threshold_' + str(index) + '_' + str(ef_train_key)] = train_extracted_features[ef_train_key]
            for ef_test_key in test_extracted_features.keys():
                test_extended_dataset['threshold_' + str(index) + '_' + str(ef_test_key)] = test_extracted_features[ef_test_key]

        # And finally add the (box-cox transformed) goal variable to be used by the upcoming regression
        train_extended_dataset['reg_goal_var'] = imported_train_dataset[imported_train_dataset.columns[reg_goal_var_index]]
        test_extended_dataset['reg_goal_var'] = imported_test_dataset[imported_test_dataset.columns[reg_goal_var_index]]

        logging.debug("Train extended dataset's first 3 rows :")
        logging.debug('\n' + str(train_extended_dataset.head(3)))

        logging.debug("Test extended dataset's first 3 rows :")
        logging.debug('\n' + str(train_extended_dataset.head(3)))

        # Save the extended datasets in a CSV file
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        train_output_name = os.path.join(output_path, 'Extended_' + train_dataset_path)
        train_extended_dataset.to_csv(path_or_buf=train_output_name, index=False)
        test_output_name = os.path.join(output_path, 'Extended_' + test_dataset_path)
        test_extended_dataset.to_csv(path_or_buf=test_output_name, index=False)

        logging.info("Split " + str(fold_index) + " extended dataset saved.")

        # Expressly free the data from the memory
        del train_extended_dataset, test_extended_dataset, train_extracted_features, test_extracted_features,\
            X_train, X_test, Y_test, Y_train, Y_test_reg_goal_var, Y_train_reg_goal_var,\
            imported_train_dataset, imported_test_dataset

        # Call python's garbage collector
        gc.collect()
