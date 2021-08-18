"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import pandas as pd
import numpy as np
import argparse
import time
import os


def argument_parser():
    """
    A parser to allow user to easily generate the results of any number of classifiers using our method.
    """

    parser = argparse.ArgumentParser(usage='\n python runner.py [dataset_name] [goal_index] [classifiers]+',
                                     description="This program allows to run all the scripts necessary to generate the"
                                                 " final figures on any number of classifiers for a given dataset.")

    parser.add_argument('--output_classes',
                        type=str,
                        default="below_threshold",
                        choices=["below_threshold", "inside_bin"],
                        help='The method of class generation')

    parser.add_argument('--split_method',
                        type=str,
                        default="equal_freq",
                        choices=["equal_width", "equal_freq", "kmeans"],
                        help='The splitting method to use')

    parser.add_argument('--classifiers',
                        type=str,
                        nargs='+',  # 1 or more values expected
                        help='The classifier models to use',
                        choices=["RandomForest", "LogisticRegression", "XGBoost", "GaussianNB", "Khiops", "DecisionTree"],
                        required=True)

    parser.add_argument('--regressors',
                        type=str,
                        nargs='+',  # 1 or more values expected
                        help='The regressor models to use',
                        choices=["RandomForest", "LinearRegression", "XGBoost", "GaussianNB", "Khiops", "DecisionTree"],
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

    parser.add_argument('--preprocess',
                        type=str,
                        help='Do the pre-processing step or not',
                        choices=["True", "False"],
                        default='False')

    parser.add_argument('--extract',
                        type=str,
                        help='Do the feature_extraction step or not',
                        choices=["True", "False"],
                        default='True')

    parser.add_argument('--grid_search',
                        type=str,
                        choices=['True', 'False'],
                        help='Automatically optimize the hyperparameters for '
                             'the given dataset using a grid search',
                        default='True')

    parser.add_argument('--n_jobs',
                        type=int,
                        help='The number of cores to use',
                        default=16)

    parser.add_argument('--log_lvl',
                        type=str,
                        default='warning',
                        choices=["debug", "info", "warning"],
                        help='Change the log display level')

    return parser.parse_args()


if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """

    args = argument_parser()

    bins_to_explore = [32]

    cmd_list = []

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'cleaned')) if f.is_dir()]
    datasets_names = [dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    results_dict_list = []

    for dataset_name in datasets_names:
        results_dict = {'dataset_name': dataset_name,
                        'Base Train RMSE': None, 'New Train RMSE': None,
                        'Base Test RMSE': None, 'New Test RMSE': None}

        if args.preprocess == 'True':
            # Process the dataset
            for bins in bins_to_explore:
                cmd = "python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --n_bins=\"{}\" --output_classes=\"{}\" --split_method=\"{}\" --log_lvl=\"{}\""\
                    .format(dataset_name, bins, args.output_classes, args.split_method, args.log_lvl)
                print("Launching " + str(cmd))
                os.system(cmd)

        # Extract the features
        for classifier in args.classifiers:
            if args.extract == 'True':
                for bins in bins_to_explore:
                    cmd = "python feature_extraction.py --dataset_folder=\"../data/processed/{}/{}_bins_{}_{}/\" --classifier=\"{}\" --log_lvl=\"{}\" --n_jobs={}"\
                        .format(dataset_name, bins, args.split_method, args.output_classes, classifier, args.log_lvl, args.n_jobs)
                    print("Launching " + str(cmd))
                    os.system(cmd)

        print("RESULTS FOR NORMAL EXTENDED ", dataset_name)
        # Generate the predictions
        for classifier in args.classifiers:
            for regressor in args.regressors:
                for bins in bins_to_explore:
                    cmd = "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/{}_bins_{}_{}/{}/\""\
                          " --regressor=\"{}\" --n_estimators=\"{}\" --max_depth=\"{}\" --max_features=\"{}\" --learning_rate=\"{}\" "\
                          "--log_lvl=\"{}\" --grid_search {} --n_jobs={}"\
                              .format(dataset_name, bins, args.split_method, args.output_classes, classifier + '_classifier', regressor,
                                      args.n_estimators, args.max_depth, args.max_features, args.learning_rate,
                                      'warning', args.grid_search, args.n_jobs)
                    print("Launching " + str(cmd))
                    start_time = time.time()
                    os.system(cmd)
                    print("Elapsed time : {0:.2f}".format(time.time() - start_time))

        # Compute the metrics
        for classifier in args.classifiers:
            for regressor in args.regressors:
                for bins in bins_to_explore:
                    cmd = "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/{}_bins_{}_{}/{}/{}/\" --log_lvl=\"{}\""\
                        .format(dataset_name, bins, args.split_method, args.output_classes, classifier + '_classifier', regressor + '_regressor', 'info')
                    print("Launching " + str(cmd))
                    os.system(cmd)

        metrics_dataframe = pd.read_csv(os.path.join('..', 'data', 'metrics', dataset_name, str(bins_to_explore[0]) + '_bins' + '_' + args.split_method + '_' + args.output_classes, str(args.classifiers[0]) + '_classifier', str(args.regressors[0]) + '_regressor', 'metrics_extracted_features.csv'))
        results_dict['Base Train RMSE'] = '{0:.4f}'.format(np.mean(metrics_dataframe['train_root_mean_squared_error']))
        results_dict['Base Test RMSE'] = '{0:.4f}'.format(np.mean(metrics_dataframe['test_root_mean_squared_error']))

        print("RESULTS FOR FIXED EXTENDED ", dataset_name)
        # Generate the predictions
        for classifier in args.classifiers:
            for regressor in args.regressors:
                for bins in bins_to_explore:
                    cmd = "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/{}_bins_{}_{}/{}/\""\
                          " --regressor=\"{}\" --n_estimators=\"{}\" --max_depth=\"{}\" --max_features=\"{}\" --learning_rate=\"{}\" "\
                          "--log_lvl=\"{}\" --grid_search {} --n_jobs={} --REMOVE_DUPLICATES True"\
                        .format(dataset_name, bins, args.split_method, args.output_classes, classifier + '_classifier', regressor,
                                args.n_estimators, args.max_depth, args.max_features, args.learning_rate,
                                'warning', args.grid_search, args.n_jobs)
                    print("Launching " + str(cmd))
                    start_time = time.time()
                    os.system(cmd)
                    print("Elapsed time : {0:.2f}".format(time.time() - start_time))

        # Compute the metrics
        for classifier in args.classifiers:
            for regressor in args.regressors:
                for bins in bins_to_explore:
                    cmd = "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/{}_bins_{}_{}/{}/{}/\" --log_lvl=\"{}\""\
                        .format(dataset_name, bins, args.split_method, args.output_classes, classifier + '_classifier', regressor + '_regressor', 'info')
                    print("Launching " + str(cmd))
                    os.system(cmd)

        metrics_dataframe = pd.read_csv(os.path.join('..', 'data', 'metrics', dataset_name, str(bins_to_explore[0]) + '_bins' + '_' + args.split_method + '_' + args.output_classes, str(args.classifiers[0]) + '_classifier', str(args.regressors[0]) + '_regressor', 'metrics_extracted_features.csv'))
        results_dict['New Train RMSE'] = '{0:.4f}'.format(np.mean(metrics_dataframe['train_root_mean_squared_error']))
        results_dict['New Test RMSE'] = '{0:.4f}'.format(np.mean(metrics_dataframe['test_root_mean_squared_error']))

        results_dict_list.append(results_dict)

        pd.DataFrame(results_dict_list).to_csv(path_or_buf=os.path.join('..', 'TMP_DIFF_DICT' + str(args.regressors[0]) + '.csv'), index=False)
