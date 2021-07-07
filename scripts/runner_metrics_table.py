"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""
import time
from os.path import isfile
from os import listdir
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics')) if f.is_dir()]
    datasets_names = [dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    for dataset_name in datasets_names:
        bins_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics', dataset_name)) if f.is_dir()]
        bins_level_directories = [bins_level_directory.split(os.path.sep)[-1] for bins_level_directory in bins_level_directories]

        for bins_level_directory in bins_level_directories:
            classifier_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics', dataset_name, bins_level_directory)) if f.is_dir()]
            classifier_level_directories = [classifier_level_directory.split(os.path.sep)[-1] for classifier_level_directory in classifier_level_directories]

            for classifier_level_directory in classifier_level_directories:
                regressor_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics', dataset_name, bins_level_directory, classifier_level_directory)) if f.is_dir()]
                regressor_level_directories = [regressor_level_directory.split(os.path.sep)[-1] for regressor_level_directory in regressor_level_directories]

                for regressor_level_directory in regressor_level_directories:
                    tmp_path = os.path.join('..', 'data', 'metrics', dataset_name, bins_level_directory, classifier_level_directory, regressor_level_directory)
                    metrics_file_name = [f for f in listdir(tmp_path) if isfile(os.path.join(tmp_path, f))][0]

                    metrics_dataframe = pd.read_csv(os.path.join(tmp_path, metrics_file_name))

                    thresholds_gen_method = None
                    if 'freq' in bins_level_directory.split('_'):
                        thresholds_gen_method = 'equal_freq'
                    elif 'width' in bins_level_directory.split('_'):
                        thresholds_gen_method = 'equal_width'
                    output_classes_method = None
                    if 'below' in bins_level_directory.split('_'):
                        output_classes_method = 'below_threshold'
                    elif 'inside' in bins_level_directory.split('_'):
                        output_classes_method = 'inside_bin'

                    results_dataframe = pd.DataFrame({'dataset_name': [dataset_name + '_' + bins_level_directory.split('_')[0] + '_bins'],
                                                      'mean_train_mean_absolute_error': ['{0:.4f}'.format(np.mean(metrics_dataframe['train_mean_absolute_error']))],
                                                      'mean_test_mean_absolute_error': ['{0:.4f}'.format(np.mean(metrics_dataframe['test_mean_absolute_error']))],
                                                      'mean_train_mean_squared_error': ['{0:.4f}'.format(np.mean(metrics_dataframe['train_mean_squared_error']))],
                                                      'mean_test_mean_squared_error': ['{0:.4f}'.format(np.mean(metrics_dataframe['test_mean_squared_error']))],
                                                      'mean_train_root_mean_squared_error': ['{0:.4f}'.format(np.mean(metrics_dataframe['train_root_mean_squared_error']))],
                                                      'mean_test_root_mean_squared_error': ['{0:.4f}'.format(np.mean(metrics_dataframe['test_root_mean_squared_error']))],
                                                      'mean_train_r_squared': ['{0:.4f}'.format(np.mean(metrics_dataframe['train_r_squared']))],
                                                      'mean_test_r_squared': ['{0:.4f}'.format(np.mean(metrics_dataframe['test_r_squared']))],
                                                      'mean_train_adjusted_r_squared': ['{0:.4f}'.format(np.mean(metrics_dataframe['train_adjusted_r_squared']))],
                                                      'mean_test_adjusted_r_squared': ['{0:.4f}'.format(np.mean(metrics_dataframe['test_adjusted_r_squared']))],
                                                      'mean_train_mean_log_loss': ['{0:.4f}'.format(np.mean(metrics_dataframe['train_mean_log_loss']))],
                                                      'mean_test_mean_log_loss': ['{0:.4f}'.format(np.mean(metrics_dataframe['test_mean_log_loss']))],
                                                      'mean_train_mean_roc_auc_score': ['{0:.4f}'.format(np.mean(metrics_dataframe['train_mean_roc_auc_score']))],
                                                      'mean_test_mean_roc_auc_score': ['{0:.4f}'.format(np.mean(metrics_dataframe['test_mean_roc_auc_score']))]})

                    results_dataframe_folder_path = os.path.join('..', 'data', 'results_tables', thresholds_gen_method + '_' + output_classes_method + '_' + classifier_level_directory + '_' + regressor_level_directory)
                    results_dataframe_file_path = os.path.join(results_dataframe_folder_path, 'results_table.csv')

                    if os.path.exists(results_dataframe_folder_path):
                        if os.path.exists(results_dataframe_file_path):
                            previous_results_dataframe = pd.read_csv(results_dataframe_file_path)
                            results_dataframe = previous_results_dataframe.append(results_dataframe)
                    else:
                        os.makedirs(results_dataframe_folder_path)

                    results_dataframe.to_csv(path_or_buf=results_dataframe_file_path, index=False)
