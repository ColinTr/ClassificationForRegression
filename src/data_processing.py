"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import DataProcessingUtils
import pandas as pd
import numpy as np
import argparse
import logging
import ntpath
import time


def argument_parser():
    """
    A parser to allow user to easily process any dataset using the split method and class generation he desires.
    """

    parser = argparse.ArgumentParser(usage='\n python3 data_processing.py [dataset_path] [output_path] [split_method] '
                                           '[output_classes] [delimiter] [header] [decimal] [na_values] [usecols] '
                                           '[goal_var_index] [n_bins] [k_folds] [log_lvl]'
                                           '\n Example : python scripts/data_processing.py '
                                           '--dataset_path=\"../data/raw/3D_Road_Network_Dataset/3D_spatial_network'
                                           '.csv\"',
                                     description="This program allows to process datasets to be used later.")

    parser.add_argument('--dataset_path',
                        type=str,
                        help='The dataset to process',
                        required=True)

    parser.add_argument('--output_path',
                        type=str,
                        help='The folder where the result will be written',
                        required=True)

    parser.add_argument('--split_method',
                        type=str,
                        default="equal_freq",
                        choices=["equal_width", "equal_freq", "kmeans"],
                        help='The splitting method to use')

    parser.add_argument('--output_classes',
                        type=str,
                        default="below_threshold",
                        choices=["below_threshold", "inside_bin"],
                        help='The method of class generation')

    parser.add_argument('--delimiter',
                        type=str,
                        default=",",
                        help='Delimiter to use when reading the dataset')

    parser.add_argument('--header',
                        type=str,
                        default="infer",
                        choices=["infer", "None"],
                        help='Infer the column names or use None if the first line isn\' a csv header line')

    parser.add_argument('--decimal',
                        type=str,
                        default=",",
                        help='Character to recognize as decimal point')

    parser.add_argument('--na_values',
                        type=str,
                        help='Additional string to recognize as NA/NaN')

    parser.add_argument('--usecols',
                        type=int,
                        nargs='*',  # 0 or more values expected
                        help='The indexes of the columns to keep')

    parser.add_argument('--goal_var_index',
                        type=int,
                        required=True,
                        help='The index of the column to use as the goal variable')

    parser.add_argument('--n_bins',
                        type=int,
                        default=10,
                        help='The number of bins to create')

    parser.add_argument('--k_folds',
                        type=int,
                        default=10,
                        help='The number of folds in the k-folds')

    parser.add_argument('--log_lvl',
                        type=str,
                        default='warning',
                        choices=["debug", "info", "warning"],
                        help='Change the log display level')

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    # Get the current configuration
    dataset_path = args.dataset_path
    output_path = args.output_path
    split_method = args.split_method
    output_classes = args.output_classes
    delimiter = args.delimiter
    header = args.header
    decimal = args.decimal
    na_values = args.na_values
    usecols = args.usecols
    goal_var_index = args.goal_var_index
    n_bins = args.n_bins
    k_folds = args.k_folds

    # Setup the logging level
    if args.log_lvl == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.log_lvl == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif args.log_lvl == 'warning':
        logging.getLogger().setLevel(logging.WARNING)
    else:
        raise ValueError('Unknown parameter for log_lvl.')

    logging.info("Reading the dataset's file...")
    reading_start_time = time.time()
    imported_dataset = pd.read_csv(dataset_path, delimiter=delimiter, header=header, decimal=decimal,
                                   na_values=na_values, usecols=usecols)
    logging.info("Dataset imported ({0:.2f}".format(time.time() - reading_start_time) + "sec)")

    # We keep all the columns except the goal variable one
    X = imported_dataset.drop(imported_dataset.columns[goal_var_index], axis=1)
    Y = imported_dataset[imported_dataset.columns[goal_var_index]]

    logging.debug("Dataset's first 3 rows :")
    logging.debug("X :")
    logging.debug(X.head(3))
    logging.debug("Y :")
    logging.debug(Y.head(3))

    # TODO : Categorical data encoding => Maybe do that in another script (or even in a notebook)

    k_fold_indexes = DataProcessingUtils.kfold_train_test_split(len(Y), k_folds)

    # We iterate for k_folds folds to create the datasets
    for k_fold_index in range(0, k_folds):
        logging.debug("========== FOLD N°" + str(k_fold_index) + " ==========")

        # Get the training and testing indexes based on the k_fold_indexes and the k_fold_index
        train_indexes = []
        test_indexes = []
        for group in range(k_folds):
            if group == k_fold_index:
                test_indexes.append(k_fold_indexes[group])
            else:
                train_indexes.append(k_fold_indexes[group])
        train_indexes = np.concatenate(train_indexes).ravel()  # Ravel the list of lists of indexes
        test_indexes = np.concatenate(test_indexes).ravel()

        X_train, Y_train = X.iloc[train_indexes].copy(), np.array(Y.iloc[train_indexes])
        X_test, Y_test = X.iloc[test_indexes].copy(), np.array(Y.iloc[test_indexes])

        # Fits the box-cox on Y_train and applies it on Y_train AND Y_test
        Y_train, Y_test = DataProcessingUtils.box_cox(Y_train, Y_test)

        # Fits the normalization on Y_train and applies it on Y_train AND Y_test
        X_train, X_test = DataProcessingUtils.normalize(X_train, X_test)

        # =================== discretization ===================
        # Thresholds definition (= Fit on the training data)
        thresholds_list = None
        if split_method == "equal_width":
            thresholds_list = DataProcessingUtils.equal_width_split(Y_train, n_bins)
        elif split_method == "equal_freq":
            thresholds_list = DataProcessingUtils.equal_freq_split(Y_train, n_bins)
        elif split_method == "kmeans":
            # (optional) TODO
            raise ValueError('This split method has not been implemented yet.')
        else:
            raise ValueError('Unknown parameter for split_method.')
        logging.debug("Thresholds :" + str(thresholds_list))

        # Discretization
        train_discretized_classes, test_discretized_classes = None, None
        if output_classes == "below_threshold":
            train_discretized_classes = DataProcessingUtils.below_threshold_class_gen(Y_train, thresholds_list)
            test_discretized_classes = DataProcessingUtils.below_threshold_class_gen(Y_test, thresholds_list)
        elif output_classes == "inside_bin":
            train_discretized_classes = DataProcessingUtils.inside_bin_class_gen(Y_train, thresholds_list)
            test_discretized_classes = DataProcessingUtils.inside_bin_class_gen(Y_test, thresholds_list)
        else:
            raise ValueError('Unknown parameter for output_classes.')
        logging.debug("Generated classes (for train dataset) :\n"
                      + str(pd.DataFrame(train_discretized_classes).head(5)))
        # ======================================================

        # We then add the generated classes to the dataframe
        # If there is only one column of class add it directly
        if type(train_discretized_classes[0]) == int:
            X_train['class'] = train_discretized_classes
            X_test['class'] = test_discretized_classes
        # But if there are multiple class columns, add each of them to the dataframe
        else:
            for class_index in range(len(train_discretized_classes[1])):
                X_train['class_' + str(class_index)] = train_discretized_classes[:, class_index]
                X_test['class_' + str(class_index)] = test_discretized_classes[:, class_index]

        logging.debug("Final dataframe (train) :\n" + str(X_train.head(3)))

        # Save the result in a CSV file
        # We generate the filename while making sure that we don't add too many '/'
        file_prefix = output_path + ('/' if output_path[-1] == '' else '/') + 'fold_' + str(k_fold_index)
        train_output_name = file_prefix + '_TRAIN_' + ntpath.basename(dataset_path)
        X_train.to_csv(path_or_buf=train_output_name, index=False)

        test_output_name = file_prefix + '_TEST_' + ntpath.basename(dataset_path)
        X_test.to_csv(path_or_buf=test_output_name, index=False)

        logging.info("Split " + str(k_fold_index) + " datasets saved in files")
