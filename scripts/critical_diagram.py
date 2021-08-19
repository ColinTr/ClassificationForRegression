"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import Orange
import os


if __name__ == "__main__":
    n_bins = 32
    n_bins_2 = 16

    prefix_path = os.path.join('D:', 'results_tables', 'tmp_rmse_for_critical_diagrams')
    prefix_path_2 = os.path.join('..', 'data', 'results_tables')

    # The paths to the RMSE table where the results are stored. At least two tables must be given to be compared
    rmse_tables = [
        (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_DecisionTree_regressor.csv'), str(n_bins) + ' T Test RMSE', 'DT+'),
        (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_DecisionTree_regressor.csv'), 'Base Test RMSE', 'DT'),

        # (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_LinearRegression_regressor.csv'), str(n_bins) + ' T Test RMSE', 'LR+'),
        # (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_LinearRegression_regressor.csv'), 'Base Test RMSE', 'LR'),

        # (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_RandomForest_regressor.csv'), str(n_bins) + ' T Test RMSE', 'RF+'),
        # (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_RandomForest_regressor.csv'), 'Base Test RMSE', 'RF'),

        # (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_XGBoost_regressor.csv'), str(n_bins) + ' T Test RMSE', 'XGB+'),
        # (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_XGBoost_regressor.csv'), 'Base Test RMSE', 'XGB'),

        # (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_Khiops_regressor.csv'), str(n_bins) + ' T Test RMSE', 'Khiops+'),
        # (os.path.join(prefix_path, 'RMSE_table_RandomForest_classifier_Khiops_regressor.csv'), 'Base Test RMSE', 'Khiops'),
    ]

    # The method that is compared with other methods If omitted, show pairwise comparison of methods
    cdmethod = None

    groups = []
    # Get for each table the column corresponding to the parameters
    for path, column, _ in rmse_tables:
        tmp_df = pd.read_csv(path)
        groups.append(list(tmp_df[column]))

    # Rank the data for each dataset
    transposed_groups = list(zip(*groups))
    ranked_data = [list(stats.rankdata(row)) for row in transposed_groups]

    average_ranks = [np.mean(vect) for vect in list(zip(*ranked_data))]

    cd = Orange.evaluation.compute_CD(average_ranks, len(rmse_tables))
    Orange.evaluation.graph_ranks(average_ranks, [vect[2] for vect in rmse_tables], cd=cd, width=6, textspace=1.5, cdmethod=cdmethod)
    plt.savefig('critical_diagram_' + str(n_bins) + 'bins.png')
