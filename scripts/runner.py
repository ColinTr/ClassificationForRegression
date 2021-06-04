"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import argparse
import os


def argument_parser():
    """
    A parser to allow user to easily generate the results of any number of classifiers using our method.
    """

    parser = argparse.ArgumentParser(usage='\n python runner.py [dataset_name] [goal_index] [classifiers]+',
                                     description="This program allows to run all the scripts necessary to generate the"
                                                 " final figures on any number of classifiers for a given dataset.")

    parser.add_argument('--dataset_name',
                        type=str,
                        help='The dataset to use',
                        required=True)

    parser.add_argument('--goal_index',
                        type=int,
                        help='The index of the goal variable',
                        required=True)

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
                        default=False)

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

    bins_to_explore = [5, 10, 15, 20, 30, 40, 50]

    cmd_list = []

    if args.preprocess == 'True':
        # Process the dataset
        for bins in bins_to_explore:
            cmd_list.append("python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index=\"{}\" --n_bins=\"{}\" --log_lvl=\"{}\""
                            .format(args.dataset_name, args.goal_index, bins, args.log_lvl))

    # Extract the features
    for classifier in args.classifiers:
        for bins in bins_to_explore:
            cmd_list.append("python feature_extraction.py --dataset_folder=\"../data/processed/{}/{}_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=\"{}\""
                            .format(args.dataset_name, bins, classifier, args.log_lvl))

    # Generate the predictions
    for classifier in args.classifiers:
        for regressor in args.regressors:
            for bins in bins_to_explore:
                cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/{}_bins_equal_freq_below_threshold/{}/\""
                                " --regressor=\"{}\" --n_estimators=\"{}\" --max_depth=\"{}\" --max_features=\"{}\" --learning_rate=\"{}\" --log_lvl=\"{}\""
                                .format(args.dataset_name, bins, classifier + '_classifier', regressor, args.n_estimators, args.max_depth, args.max_features, args.learning_rate, args.log_lvl))

    # Compute the metrics
    for classifier in args.classifiers:
        for regressor in args.regressors:
            for bins in bins_to_explore:
                cmd_list.append("python compute_metrics.py --predictions_folder=\"../data/predictions/{}/{}_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=\"{}\""
                                .format(args.dataset_name, bins, classifier + '_classifier', regressor + '_regressor', args.log_lvl))

    # Compute the baseline
    for regressor in args.regressors:
        cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/processed/{}/5_bins_equal_freq_below_threshold/\" "
                        "--regressor=\"{}\" --n_estimators=\"{}\" --max_depth=\"{}\" --max_features=\"{}\" --learning_rate=\"{}\" --log_lvl=\"{}\""
                        .format(args.dataset_name, regressor, args.n_estimators, args.max_depth, args.max_features, args.learning_rate, args.log_lvl))
        cmd_list.append("python compute_metrics.py --predictions_folder=\"../data/predictions/{}/5_bins_equal_freq_below_threshold/Standard/{}\" --log_lvl=\"{}\""
                        .format(args.dataset_name, regressor + '_regressor', args.log_lvl))

    # Create the graphs
    cmd_list.append("python visualisation.py --parent_folder=\"../data/metrics/{}\" --metric=\"r_squared\"".format(args.dataset_name))
    cmd_list.append("python visualisation.py --parent_folder=\"../data/metrics/{}\" --metric=\"RMSE\"".format(args.dataset_name))

    for c in cmd_list:
        print("Launching :\n" + str(c))
        os.system(c)
