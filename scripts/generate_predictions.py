"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import platform
import argparse
import logging
import time
import glob
import json
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
from sklearn.tree import DecisionTreeRegressor
from os.path import isfile, join
from xgboost import XGBRegressor
from os import listdir

try:
    from pykhiops.sklearn import KhiopsRegressor
except ImportError:
    logging.warning('Failed to import PyKhiops, KhiopsRegressor will be unavailable')


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
                        choices=["RandomForest", "LinearRegression", "XGBoost", "Khiops", "DecisionTree"],
                        required=True)

    parser.add_argument('--n_estimators',
                        type=str,
                        help='The number of trees in the forest of RandomForest or the number of gradient boosted trees'
                             ' for XGBoost',
                        default=None)

    parser.add_argument('--max_depth',
                        type=str,
                        help='The maximum depth of the trees in RandomForest, XGBoost or DecisionTree',
                        default=None)

    parser.add_argument('--max_features',
                        type=str,
                        help='The number of features to consider when looking for the best split in RandomForest or '
                             'DecisionTree',
                        default=None)

    parser.add_argument('--learning_rate',
                        type=str,
                        help='Boosting learning rate of XGBoost',
                        default=None)

    parser.add_argument('--use_hyperparam_file',
                        type=str,
                        choices=['True', 'False'],
                        help='Use the hyperparameters in the hyperparameters.json file that is '
                             'in the same folder of the dataset',
                        default='False')

    parser.add_argument('--n_jobs',
                        type=int,
                        help='The number of cores to use',
                        default=-1)

    parser.add_argument('--grid_search',
                        type=str,
                        choices=['True', 'False'],
                        help='Automatically optimize the hyperparameters for '
                             'the given dataset using a grid search',
                        default='False')

    parser.add_argument('--log_lvl',
                        type=str,
                        default='info',
                        choices=["debug", "info", "warning"],
                        help='Change the log display level')

    return parser.parse_args()


def get_column_with_word(full_dataframe, word):
    thresholds_cols_indexes = []
    for column_name, column_index in zip(list(full_dataframe.columns.values), range(0, len(list(full_dataframe.columns.values)))):
        if word in column_name.split('_'):
            thresholds_cols_indexes.append(column_index)
    classes_df = full_dataframe[full_dataframe.columns[thresholds_cols_indexes]]
    return classes_df


def xgboost_grid_search(X, Y):
    """
    Execute a grid search for XGBoost regressor on the given dataset to find
    the optimal max_depth and learning_rate
    :param X: DataFrame or np.array : The dataset
    :param Y: DataFrame or np.array : The target values
    :return: best_max_depth, best_learning_rate
    """
    param_grid = {'n_jobs': [4],
                  'n_estimators': [100],
                  'max_depth': [4, 8, 16, 32],
                  'learning_rate': [0.01, 0.1, 0.3]}

    grid = GridSearchCV(estimator=XGBRegressor(),
                        param_grid=param_grid,
                        scoring='neg_mean_squared_error',
                        n_jobs=4)

    grid.fit(np.ascontiguousarray(X), np.ascontiguousarray(Y))

    logging.info('Grid search\'s optimal parameters are : '
                 'max_depth=' + str(grid.best_params_['max_depth']) +
                 ' & learning_rate=' + str(grid.best_params_['learning_rate']))

    return grid.best_params_['max_depth'], grid.best_params_['learning_rate']


def random_forest_grid_search(X, Y):
    """
    Execute a grid search for Random Forest regressor on the given dataset to
    find the optimal max_depth and max_features
    :param X: DataFrame or np.array : The dataset
    :param Y: DataFrame or np.array : The target values
    :return: best_max_depth, best_max_features
    """
    param_grid = {'n_jobs': [4],
                  'n_estimators': [100],
                  'max_depth': [4, 8, 16, 32],
                  'max_features': []}

    values_to_explore = list(map(int, np.linspace(2, X.shape[1], num=4)))
    values_to_explore.append(int(np.sqrt(X.shape[1])))
    values_to_explore = np.unique(values_to_explore)
    param_grid['max_features'] = values_to_explore

    grid = GridSearchCV(estimator=RandomForestRegressor(),
                        param_grid=param_grid,
                        scoring='neg_mean_squared_error',
                        n_jobs=4)

    grid.fit(X, Y)

    logging.info('Grid search\'s optimal parameters are : '
                 'max_depth=' + str(grid.best_params_['max_depth']) +
                 ' & max_features=' + str(grid.best_params_['max_features']))

    return grid.best_params_['max_depth'], grid.best_params_['max_features']


def decision_tree_grid_search(X, Y):
    """
    Execute a grid search for Decision Tree regressor on the given dataset to
    find the optimal max_depth and max_features
    :param X: DataFrame or np.array : The dataset
    :param Y: DataFrame or np.array : The target values
    :return: best_max_depth, best_max_features
    """
    param_grid = {'max_depth': [4, 8, 16, 32],
                  'max_features': []}

    values_to_explore = list(map(int, np.linspace(2, X.shape[1], num=4)))
    values_to_explore.append(int(np.sqrt(X.shape[1])))
    values_to_explore = np.unique(values_to_explore)
    param_grid['max_features'] = values_to_explore

    grid = GridSearchCV(estimator=DecisionTreeRegressor(),
                        param_grid=param_grid,
                        scoring='neg_mean_squared_error',
                        n_jobs=4)

    grid.fit(X, Y)

    logging.info('Grid search\'s optimal parameters are : '
                 'max_depth=' + str(grid.best_params_['max_depth']) +
                 ' & max_features=' + str(grid.best_params_['max_features']))

    return grid.best_params_['max_depth'], grid.best_params_['max_features']


if __name__ == "__main__":
    args = argument_parser()

    # Setup the logging level
    setup_logging_level(args.log_lvl)

    dataset_folder = args.dataset_folder
    output_path = args.output_path
    use_hyperparam_file = args.use_hyperparam_file

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

    # If no value was given for the parameter 'goal_var_index', try to find if a .index file is in the dataset's folder
    hyperparameters = None
    if use_hyperparam_file == 'True':
        dataset_name = dataset_folder.split(os.path.sep)[dataset_folder.split(os.path.sep).index('data') + 2]
        hyperparameters_files = glob.glob(os.path.join('..', 'data', 'cleaned', dataset_name, 'hyperparameters.json'))
        if len(hyperparameters_files) > 1:
            raise ValueError('More than one hyperparameters file was found in the dataset\'s folder.')
        elif len(hyperparameters_files) == 0:
            raise ValueError('No index hyperparameters was found in the dataset\'s folder, please create one or'
                             'define explicitly each hyperparameter.')
        else:
            with open(hyperparameters_files[0]) as f:
                data = json.load(f)
            if args.regressor not in data.keys():
                raise ValueError('Hyperparameters file does not contain values for ' + args.regressor)
            else:
                hyperparameters = data[args.regressor]
            logging.info('Using file\'s hyperparameters : ' + str(hyperparameters))

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

        train_dataframe_path = os.path.join(dataset_folder, train_filename)
        logging.debug('Reading training file : ' + train_dataframe_path + '...')
        reading_start_time = time.time()
        col_names = pd.read_csv(train_dataframe_path, nrows=0).columns
        types_dict = {}
        for col_name in col_names:
            if 'class' in col_name.split('_'):
                types_dict[col_name] = np.int16
            else:
                types_dict[col_name] = np.float16 if args.regressor != "XGBoost" else np.float32
        train_dataframe = pd.read_csv(train_dataframe_path, dtype=types_dict)
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

        test_dataframe_path = os.path.join(dataset_folder, test_filename)
        logging.debug('Reading testing file : ' + test_dataframe_path + '...')
        reading_start_time = time.time()
        col_names = pd.read_csv(test_dataframe_path, nrows=0).columns
        types_dict = {}
        for col_name in col_names:
            if 'class' in col_name.split('_'):
                types_dict[col_name] = np.int16
            else:
                types_dict[col_name] = np.float16 if args.regressor != "XGBoost" else np.float32
        test_dataframe = pd.read_csv(test_dataframe_path, dtype=types_dict)
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
            n_estimators = 100 if (args.n_estimators is None or args.n_estimators == 'None') else args.n_estimators  # Default value is 100
            n_estimators = int(n_estimators)

            if args.grid_search == 'True' or args.grid_search is True:
                logging.info('Starting grid_search for RandomForest...')
                max_depth, max_features = random_forest_grid_search(X_train, Y_train)
            else:
                max_depth = None  # Default value is None
                if use_hyperparam_file == 'True' and 'max_depth' in hyperparameters.keys():
                    max_depth = hyperparameters['max_depth']
                if args.max_depth is not None and args.max_depth != 'None':
                    max_depth = args.max_depth
                if max_depth is not None and max_depth != 'None':
                    max_depth = int(max_depth)

                max_features = 'auto'  # Default value is 'auto'
                if use_hyperparam_file == 'True' and 'max_features' in hyperparameters.keys():
                    max_features = hyperparameters['max_features']
                if args.max_features is not None and args.max_features != 'None':
                    max_features = args.max_features
                if max_features != 'auto' and max_features != 'sqrt' and max_features != 'log2':
                    max_features = int(max_features)

                logging.info('Using the following parameters for RandomForestRegressor : '
                             'n_estimators=' + str(n_estimators) + ' / max_depth=' + str(max_depth) + ' / max_features=' + str(max_features))

            model = RandomForestRegressor(n_jobs=args.n_jobs, n_estimators=n_estimators,
                                          max_depth=max_depth, max_features=max_features)

            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

        elif args.regressor == "LinearRegression":
            model = LinearRegression(n_jobs=args.n_jobs)
            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

        elif args.regressor == "XGBoost":
            n_estimators = 100 if (args.n_estimators is None or args.n_estimators == 'None') else args.n_estimators  # Default value is 100
            n_estimators = int(n_estimators)

            if args.grid_search == 'True' or args.grid_search is True:
                logging.info('Starting grid_search for XGBoost...')
                max_depth, learning_rate = xgboost_grid_search(X_train, Y_train)
            else:
                max_depth = 6  # Default value is 6
                if use_hyperparam_file == 'True' and 'max_depth' in hyperparameters.keys():
                    max_depth = hyperparameters['max_depth']
                if args.max_depth is not None and args.max_depth != 'None':
                    max_depth = args.max_depth
                if max_depth is not None and max_depth != 'None':
                    max_depth = int(max_depth)

                learning_rate = 0.3  # Default value is 0.3
                if use_hyperparam_file == 'True' and 'learning_rate' in hyperparameters.keys():
                    learning_rate = hyperparameters['learning_rate']
                if args.learning_rate is not None and args.learning_rate != 'None':
                    learning_rate = args.learning_rate
                learning_rate = float(learning_rate)

                logging.info('Using the following parameters for XGBRegressor : '
                             'n_estimators=' + str(n_estimators) + ' / max_depth=' + str(max_depth) + ' / learning_rate=' + str(learning_rate))

            model = XGBRegressor(n_jobs=args.n_jobs, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(np.ascontiguousarray(X_train))
            Y_test_pred = model.predict(np.ascontiguousarray(X_test))

        elif args.regressor == "Khiops":
            model = KhiopsRegressor()

            model.fit(X=X_train, y=Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

        elif args.regressor == "DecisionTree":
            if args.grid_search == 'True' or args.grid_search is True:
                logging.info('Starting grid_search for DecisionTree...')
                max_depth, max_features = decision_tree_grid_search(X_train, Y_train)
            else:
                max_depth = None  # Default value is None
                if use_hyperparam_file == 'True' and 'max_depth' in hyperparameters.keys():
                    max_depth = hyperparameters['max_depth']
                if args.max_depth is not None and args.max_depth != 'None':
                    max_depth = args.max_depth
                if max_depth is not None and max_depth != 'None':
                    max_depth = int(max_depth)

                max_features = None  # Default value is None
                if use_hyperparam_file == 'True' and 'max_features' in hyperparameters.keys():
                    max_features = hyperparameters['max_features']
                if args.max_features is not None and args.max_features != 'None':
                    max_features = args.max_features
                if max_features is not None and max_features != 'auto' and max_features != 'sqrt' and max_features != 'log2':
                    max_features = int(max_features)

                logging.info('Using the following parameters for DecisionTreeRegressor : '
                             'max_depth=' + str(max_depth) + ' / max_features=' + str(max_features))

            model = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features)

            model.fit(X_train, Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

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

        # If the predictions were made on an extended dataset, we add the classifiers predictions as well
        #    for future metrics computation
        if extended:
            # Real class numbers, needed for AUC ROC for instance
            train_prediction_dataset = pd.concat([train_prediction_dataset, get_column_with_word(train_dataframe, 'class')], axis=1)
            test_prediction_dataset = pd.concat([test_prediction_dataset, get_column_with_word(test_dataframe, 'class')], axis=1)

            # The predicted probabilities for each class of every classifier associated to the thresholds
            train_prediction_dataset = pd.concat([train_prediction_dataset, get_column_with_word(train_dataframe, 'threshold')], axis=1)
            test_prediction_dataset = pd.concat([test_prediction_dataset, get_column_with_word(test_dataframe, 'threshold')], axis=1)

            # And finally the predicted probability of the real classes
            train_prediction_dataset = pd.concat([train_prediction_dataset, train_df_predicted_probas], axis=1)
            test_prediction_dataset = pd.concat([test_prediction_dataset, test_df_predicted_probas], axis=1)

        # Save the extended datasets in a CSV file
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        train_prediction_dataset.to_csv(path_or_buf=os.path.join(output_path, train_filename), index=False)
        test_prediction_dataset.to_csv(path_or_buf=os.path.join(output_path, test_filename), index=False)

        logging.info("Split " + str(fold_num) + " predictions predictions saved.")

        # Expressly free the variables from the memory
        del train_dataframe, test_dataframe, X_train, X_test, Y_train, Y_test, Y_train_pred, Y_test_pred, model

        # Call python's garbage collector
        gc.collect()

    logging.info('All predictions saved in folder ' + output_path + ' !')
