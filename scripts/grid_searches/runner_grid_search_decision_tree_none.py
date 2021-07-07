"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""
import logging

from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import pandas as pd
import numpy as np
import glob
import sys
import os
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.logging_util import setup_file_logging
from src.utils.logging_util import setup_logging_level


def box_cox(Y):
    Y = Y.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(1, 2))
    scaler.fit(Y)
    Y = scaler.transform(Y)

    # Fit BoxCox on the training dataset
    transform = PowerTransformer(method='box-cox')  # Only works with strictly positive values !
    transform.fit(Y)
    return transform.transform(Y)


def normalize(X):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    return pd.DataFrame(scaler.transform(X))


if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """
    setup_file_logging()
    setup_logging_level('info')

    output_classes = 'below_threshold'
    split_method = 'equal_freq'
    log_lvl = 'warning'
    model = 'RandomForest'
    dt_regr_grid = {'max_depth': [None],
                    'max_features': []}

    datasets_directories = [x[0] for x in os.walk(os.path.join('..', '..', 'data', 'cleaned'))][1:]
    datasets_paths = [glob.glob(dataset_directory + os.path.sep + '*.csv')[0] for dataset_directory in datasets_directories]
    datasets_paths = sorted(datasets_paths)  # Sort alphabetically

    indexes_paths = [glob.glob(dataset_directory + os.path.sep + '*.index')[0] for dataset_directory in datasets_directories]
    indexes_paths = sorted(indexes_paths)  # Sort alphabetically

    for dataset_path, index_path in zip(datasets_paths, indexes_paths):
        target_var_index = None
        with open(index_path) as f:
            target_var_index = int(f.readline())

        logging.info('--- ' + dataset_path.split(os.path.sep)[3] + ' (target index = ' + str(target_var_index) + ')')
        full_data = pd.read_csv(dataset_path)
        X = full_data.drop(full_data.columns[target_var_index], axis=1)
        Y = np.ascontiguousarray(full_data[full_data.columns[target_var_index]])

        X = np.ascontiguousarray(normalize(X))
        Y = np.ascontiguousarray(box_cox(Y))
        Y = Y.ravel()

        values_to_explore = list(map(int, np.linspace(2, X.shape[1], num=4)))
        values_to_explore.append(int(np.sqrt(X.shape[1])))
        values_to_explore = np.unique(values_to_explore)
        dt_regr_grid['max_features'] = values_to_explore

        logging.info('    --- DecisionTreeRegressor...')
        grid = GridSearchCV(estimator=DecisionTreeRegressor(),
                            param_grid=dt_regr_grid,
                            scoring='neg_mean_squared_error',
                            n_jobs=4)
        grid.fit(X, Y)

        logging.info('       ' + str(grid.best_params_) + 'Actual max_depth :' + str(grid.best_estimator_.get_depth()))
        logging.info('        best_score_ :' + str(grid.best_score_))

        del grid, X, Y, full_data
        gc.collect()
