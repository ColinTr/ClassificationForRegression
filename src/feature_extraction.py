"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""
import os
import sys

from models.RandomForestC import RandomForestC
from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np
import argparse
import logging
import time


def argument_parser():
    """
    TODO
    """

    parser = argparse.ArgumentParser(usage='\n python3 feature_extraction.py [dataset_folder] [output_path]'
                                           '|classifier] [class_cols] [log_lvl]'
                                           '\n Example : python scripts/feature_extraction.py TODO...',
                                     description="This program allows to extract features from a dataset.")

    parser.add_argument('--dataset_folder',
                        type=str,
                        help='The folder where the k-fold datasets are stored',
                        required=True)

    parser.add_argument('--output_path',
                        type=str,
                        help='The folder where the results will be saved',
                        required=True)

    parser.add_argument('--classifier',
                        type=str,
                        help='The classifier model to use',
                        choices=["RandomForests", "LogisticRegression", "XGBoost", "GaussianNB", "Khiops"],
                        required=True)

    parser.add_argument('--class_cols',
                        type=int,
                        nargs='+',  # 1 or more values expected,
                        help='The indexes of the classes columns',
                        required=True)

    parser.add_argument('--log_lvl',
                        type=str,
                        default='warning',
                        choices=["debug", "info", "warning"],
                        help='Change the log display level')

    return parser.parse_args()


def sort_dict(dictionary):
    sorted_dict = {}
    for key in sorted(dictionary.keys()):
        sorted_dict[key] = dictionary[key]
    return sorted_dict


def create_new_classifier_model(classifier_name):
    if classifier_name == 'RandomForests':
        return RandomForestC()
    elif classifier_name == 'LogisticRegression':
        # TODO
        return None
    elif classifier_name == 'XGBoost':
        # TODO
        return None
    elif classifier_name == 'GaussianNB':
        # TODO
        return None
    elif classifier_name == 'Khiops':
        # TODO
        return None
    else:
        raise ValueError('Unknown parameter for classifier.')


def detect_class_columns(header):
    index = 0
    class_cols_indexes = []
    for column_name in header:
        if column_name.split('_')[0]=='class':
            class_cols_indexes.append(index)
        index = index + 1
    return class_cols_indexes


if __name__ == "__main__":
    args = argument_parser()

    dataset_folder = args.dataset_folder
    output_path = args.output_path

    # Setup the logging level
    if args.log_lvl == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.log_lvl == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif args.log_lvl == 'warning':
        logging.getLogger().setLevel(logging.WARNING)
    else:
        raise ValueError('Unknown parameter for log_lvl.')

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

    for train_key, test_key, fold_index in zip(train_files_dict.keys(), test_files_dict.keys(), range(0, len(test_files_dict.keys()))):
        train_dataset_path = train_files_dict[train_key]
        test_dataset_path = test_files_dict[test_key]

        logging.info("Reading the dataset's train and test file...")
        reading_start_time = time.time()
        imported_train_dataset = pd.read_csv(os.path.join(dataset_folder, train_dataset_path), delimiter=',')
        imported_test_dataset = pd.read_csv(os.path.join(dataset_folder, test_dataset_path), delimiter=',')
        logging.info("Dataset imported ({0:.2f}".format(time.time() - reading_start_time) + "sec)")

        # We keep all the columns except the goal variable ones
        class_columns_indexes = detect_class_columns(list(imported_train_dataset.columns.values))

        X_train = imported_train_dataset.drop(imported_train_dataset.columns[class_columns_indexes], axis=1)
        Y_train = imported_train_dataset[imported_train_dataset.columns[class_columns_indexes]]
        X_test = imported_test_dataset.drop(imported_test_dataset.columns[class_columns_indexes], axis=1)
        extended_X_test = X_test.copy()  # We will gradually add the extracted features to this dataframe
        Y_test = imported_test_dataset[imported_test_dataset.columns[class_columns_indexes]]

        logging.debug("Dataset's first 3 rows :")
        logging.debug('X_train :\n' + str(X_train.head(3)))
        logging.debug('Y_train :\n' + str(Y_train.head(3)))
        logging.debug('X_test :\n' + str(X_test.head(3)))
        logging.debug('Y_test :\n' + str(Y_test.head(3)))

        # For each class column, train a classifier and extract its features
        for train_column, test_column, index in zip(Y_train, Y_test, range(0, len(Y_train))):
            classifier_model = create_new_classifier_model(args.classifier)
            classifier_model.fit(X_train, Y_train[train_column])
            extracted_features, model_score = classifier_model.extract_features(X_test, Y_test[test_column])
            logging.info('Fold ' + str(fold_index) + ', model ' + str(index) + ' accuracy : ' + str(model_score))
            # np.set_printoptions(threshold=sys.maxsize)
            # print(np.array(extracted_features))

            # We can now add the extracted features to the dataframe :
            for key in extracted_features.keys():
                extended_X_test['class_' + str(index) + '_' + str(key)] = extracted_features[key]

        logging.debug("Extended dataset's first 3 rows :")
        logging.debug('\n' + str(extended_X_test.head(3)))

        # Save the extended dataset in a CSV file
        # We generate the filename while making sure that we don't add too many '/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        train_output_name = os.path.join(output_path, 'Extended_' + test_dataset_path)
        extended_X_test.to_csv(path_or_buf=train_output_name, index=False)

        logging.info("Split " + str(fold_index) + " extended dataset saved.")
