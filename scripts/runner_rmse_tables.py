"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import argparse
import os

import numpy as np
import pandas as pd


def argument_parser():
    """
    A parser to allow user to easily extract features of any folder of datasets with the classifier of his choice.
    """
    parser = argparse.ArgumentParser(usage='\n python feature_extraction.py [dataset_folder] [output_path]'
                                           '|classifier] [class_cols] [log_lvl]',
                                     description="This program allows to extract features from a dataset.")

    parser.add_argument('--regressor',
                        type=str,
                        help='The regression model to use',
                        choices=["RandomForest", "LinearRegression", "XGBoost", "Khiops", "DecisionTree"],
                        required=True)

    parser.add_argument('--classifier',
                        type=str,
                        help='The classifier model to use',
                        choices=["RandomForest", "LogisticRegression", "XGBoost", "GaussianNB", "Khiops", "DecisionTree"],
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

    return parser.parse_args()


if __name__ == "__main__":
    """
    Takes all of the metrics of a regressor and classifier for all the datasets and organises them in a csv file
    """
    args = argument_parser()

    split_method = None
    if 'width' in args.split_method.split('_'):
        split_method = 'width'
    elif 'freq' in args.split_method.split('_'):
        split_method = 'freq'
    elif 'kmeans' in args.split_method.split('_'):
        split_method = 'kmeans'

    output_classes = None
    if 'below' in args.output_classes.split('_'):
        output_classes = 'below'
    elif 'inside' in args.output_classes.split('_'):
        output_classes = 'inside'

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics')) if f.is_dir()]
    datasets_names = [dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    results_dict_list = []

    for dataset_name in datasets_names:
        results_dict = {'dataset_name': dataset_name, 'Base Train RMSE': None, '2 T Train RMSE': None,
                        '4 T Train RMSE': None, '8 T Train RMSE': None, '16 T Train RMSE': None,
                        '32 T Train RMSE': None, 'Base Test RMSE': None, '2 T Test RMSE': None,
                        '4 T Test RMSE': None, '8 T Test RMSE': None, '16 T Test RMSE': None,
                        '32 T Test RMSE': None}

        bins_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics', dataset_name)) if f.is_dir()]
        bins_level_directories = [bins_level_directory.split(os.path.sep)[-1] for bins_level_directory in bins_level_directories]

        # Filter only the folder that are of the right output_classes and split_method
        bins_level_directories = [a for a in bins_level_directories if split_method in a.split('_') and output_classes in a.split('_')]

        for bins_level_directory in bins_level_directories:
            classifier_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics', dataset_name, bins_level_directory)) if f.is_dir()]
            classifier_level_directories = [classifier_level_directory.split(os.path.sep)[-1] for classifier_level_directory in classifier_level_directories]

            # Filter only the folder that are of the right classifier
            classifier_level_directories = [a for a in classifier_level_directories if args.classifier in a.split('_') or a == 'Standard']

            for classifier_level_directory in classifier_level_directories:
                regressor_level_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'metrics', dataset_name, bins_level_directory, classifier_level_directory)) if f.is_dir()]
                regressor_level_directories = [regressor_level_directory.split(os.path.sep)[-1] for regressor_level_directory in regressor_level_directories]

                # Filter only the folder that are of the right regressor
                regressor_level_directories = [a for a in regressor_level_directories if args.regressor in a.split('_')]

                for regressor_level_directory in regressor_level_directories:
                    tmp_path = os.path.join('..', 'data', 'metrics', dataset_name, bins_level_directory, classifier_level_directory, regressor_level_directory)
                    metrics_file_name = [f for f in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, f))][0]

                    metrics_dataframe = pd.read_csv(os.path.join(tmp_path, metrics_file_name))


                    n_bins = bins_level_directory.split('_')[0]

                    if classifier_level_directory == 'Standard':
                        results_dict['Base Train RMSE'] = '{0:.4f}'.format(np.mean(metrics_dataframe['train_root_mean_squared_error']))
                        results_dict['Base Test RMSE'] = '{0:.4f}'.format(np.mean(metrics_dataframe['test_root_mean_squared_error']))
                    else:
                        results_dict[str(n_bins) + ' T Train RMSE'] = '{0:.4f}'.format(np.mean(metrics_dataframe['train_root_mean_squared_error']))
                        results_dict[str(n_bins) + ' T Test RMSE'] = '{0:.4f}'.format(np.mean(metrics_dataframe['test_root_mean_squared_error']))

        results_dict_list.append(results_dict)

    path = os.path.join('..', 'data', 'results_tables', args.split_method + '_' + args.output_classes + '_' + args.classifier + '_classifier_' + args.regressor + '_regressor')

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, 'RMSE_table.csv')

    pd.DataFrame(results_dict_list).to_csv(path_or_buf=path, index=False)
