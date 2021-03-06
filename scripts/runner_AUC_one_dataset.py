"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import argparse
import os
import time


def argument_parser():
    """
    A parser to allow user to easily generate the results of any number of classifiers using our method.
    """

    parser = argparse.ArgumentParser(usage='\n python runner.py [dataset_name] [goal_index] [classifiers]+',
                                     description="This program allows to run all the scripts necessary to generate the"
                                                 " ROC AUC metric for a number of thresholds.")

    parser.add_argument('--dataset_name',
                        type=str,
                        help='The dataset to use',
                        required=True)

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

    bins_to_explore = [2, 4, 8, 16, 32]
    n_jobs = 16

    cmd_list = []

    if args.preprocess == 'True':
        # Process the dataset
        for bins in bins_to_explore:
            cmd_list.append("python data_processing.py --dataset_path=\"" + os.path.join('..', 'data', 'cleaned', args.dataset_name, 'data.csv') + "\" --n_bins=\"{}\" --output_classes=\"{}\" --split_method=\"{}\" --log_lvl=\"{}\""
                            .format(bins, args.output_classes, args.split_method, args.log_lvl))

    # Extract the features
    for classifier in args.classifiers:
        for bins in bins_to_explore:
            cmd_list.append("python feature_extraction.py --dataset_folder=\"" + os.path.join('..', 'data', 'processed', args.dataset_name, str(bins) + '_bins_' + str(args.split_method) + '_' + str(args.output_classes)) + "\" --classifier=\"{}\" --log_lvl=\"{}\" --n_jobs={}"
                            .format(classifier, args.log_lvl, n_jobs))

    # Generate the predictions
    for classifier in args.classifiers:
        for regressor in args.regressors:
            for bins in bins_to_explore:
                cmd_list.append("python generate_predictions.py --dataset_folder=\"" + os.path.join('..', 'data', 'extracted_features', args.dataset_name, str(bins) + '_bins_' + str(args.split_method) + '_' + str(args.output_classes), classifier + '_classifier') + "\""
                                " --regressor=\"{}\" --n_estimators=\"{}\" --max_depth=\"{}\" --max_features=\"{}\" --learning_rate=\"{}\" --log_lvl=\"{}\" --n_jobs={}"
                                .format(regressor, args.n_estimators, args.max_depth, args.max_features, args.learning_rate, args.log_lvl, n_jobs))

    # Compute the metrics
    for classifier in args.classifiers:
        for regressor in args.regressors:
            for bins in bins_to_explore:
                cmd_list.append("python compute_metrics.py --predictions_folder=\"" + os.path.join('..', 'data', 'predictions', args.dataset_name, str(bins) + '_bins_' + str(args.split_method) + '_' + str(args.output_classes), classifier + '_classifier', regressor + '_regressor') + "\" --log_lvl=\"{}\""
                                .format(args.log_lvl))

    # Create the graph
    cmd_list.append("python visualisation.py --parent_folder=\"" + os.path.join('..', 'data', 'metrics', args.dataset_name) + "\" --metric=\"RMSE\"")

    for c in cmd_list:
        print("Launching :\n" + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
