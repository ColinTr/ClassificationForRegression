"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import os

if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """

    dataset_name = 'Appliances_energy_prediction_Dataset'
    goal_index = 0
    method_1 = 'RandomForest'
    method_2 = 'XGBoost'

    cmd_list = [
        # ========== Process the dataset ==========
        "python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index={} --n_bins=5 --log_lvl=warning".format(dataset_name, goal_index),
        "python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index={} --n_bins=10 --log_lvl=warning".format(dataset_name, goal_index),
        "python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index={} --n_bins=15 --log_lvl=warning".format(dataset_name, goal_index),
        "python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index={} --n_bins=20 --log_lvl=warning".format(dataset_name, goal_index),
        "python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index={} --n_bins=30 --log_lvl=warning".format(dataset_name, goal_index),
        "python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index={} --n_bins=40 --log_lvl=warning".format(dataset_name, goal_index),
        "python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index={} --n_bins=50 --log_lvl=warning".format(dataset_name, goal_index),
        # =========================================

        # ========== Extract the features ==========
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/5_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_1),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/10_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_1),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/15_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_1),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/20_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_1),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/30_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_1),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/40_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_1),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/50_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_1),

        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/5_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_2),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/10_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_2),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/15_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_2),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/20_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_2),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/30_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_2),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/40_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_2),
        "python feature_extraction.py --dataset_folder=\"../data/processed/{}/50_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(dataset_name, method_2),
        # ==========================================

        # ========== Generate the predictions ==========
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/5_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/10_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/15_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/20_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/30_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/40_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/50_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1),

        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/5_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/10_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/15_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/20_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/30_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/40_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/50_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2),

        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/5_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/10_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/15_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/20_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/30_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/40_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/50_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2),

        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/5_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/10_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/15_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/20_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/30_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/40_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1),
        "python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/50_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1),
        # ==============================================

        # ========== Compute the metrics ==========
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/5_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/10_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/15_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/20_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/30_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/40_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/50_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_1 + '_regressor'),

        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/5_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/10_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/15_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/20_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/30_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/40_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/50_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_1 + '_regressor'),

        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/5_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/10_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/15_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/20_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/30_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/40_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/50_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_1 + '_classifier', method_2 + '_regressor'),

        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/5_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/10_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/15_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/20_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/30_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/40_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/50_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(dataset_name, method_2 + '_classifier', method_2 + '_regressor'),
        # =========================================

        # ========== Compute the baseline ==========
        "python generate_predictions.py --dataset_folder=\"../data/processed/{}/5_bins_equal_freq_below_threshold/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_1),
        "python generate_predictions.py --dataset_folder=\"../data/processed/{}/5_bins_equal_freq_below_threshold/\" --regressor=\"{}\" --log_lvl=warning".format(dataset_name, method_2),

        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/5_bins_equal_freq_below_threshold/Standard/{}\" --log_lvl=warning".format(dataset_name, method_1 + '_regressor'),
        "python compute_metrics.py --predictions_folder=\"../data/predictions/{}/5_bins_equal_freq_below_threshold/Standard/{}\" --log_lvl=warning".format(dataset_name, method_2 + '_regressor'),
        # ==========================================

        # ========== Create the graphs ==========
        "python visualisation.py --parent_folder=\"../data/metrics/{}\"".format(dataset_name)
        # =======================================
    ]

    for c in cmd_list:
        print("Launching :\n" + str(c))
        os.system(c)
