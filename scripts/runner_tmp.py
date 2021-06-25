"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import os
import time

if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """

    cmd_list = [
        "python data_processing.py --dataset_path=\"../data/cleaned/3D_Road_Network_Dataset/data.csv\" --goal_var_index=3",
        "python data_processing.py --dataset_path=\"../data/cleaned/Appliances_energy_prediction_Dataset/data.csv\" --goal_var_index=0",
        "python data_processing.py --dataset_path=\"../data/cleaned/Beijing_PM2.5_Data_Dataset/data.csv\" --goal_var_index=4",
        "python data_processing.py --dataset_path=\"../data/cleaned/Bike_Sharing_Dataset/data.csv\" --goal_var_index=8",
        "python data_processing.py --dataset_path=\"../data/cleaned/BlogFeedback_Dataset/data.csv\" --goal_var_index=18",
        "python data_processing.py --dataset_path=\"../data/cleaned/Buzz_in_social_media_Dataset/data.csv\" --goal_var_index=70",
        "python data_processing.py --dataset_path=\"../data/cleaned/Combined_Cycle_Power_Plant_Dataset/data.csv\" --goal_var_index=4",
        "python data_processing.py --dataset_path=\"../data/cleaned/Condition_Based_Maintenance_of_Naval_Propulsion_Plants_Dataset/data.csv\" --goal_var_index=15",
        "python data_processing.py --dataset_path=\"../data/cleaned/Cuff-Less_Blood_Pressure_Estimation_Dataset/data.csv\" --goal_var_index=1",
        "python data_processing.py --dataset_path=\"../data/cleaned/Facebook_Comment_Volume_Dataset/data.csv\" --goal_var_index=53",
        "python data_processing.py --dataset_path=\"../data/cleaned/Greenhouse_Gas_Observing_Network_Dataset/data.csv\" --goal_var_index=15",
        "python data_processing.py --dataset_path=\"../data/cleaned/Individual_household_electric_power_consumption_Dataset/data.csv\" --goal_var_index=12",
        "python data_processing.py --dataset_path=\"../data/cleaned/KEGG_Metabolic_Reaction_Network_(Undirected)_Dataset/data.csv\" --goal_var_index=27",
        "python data_processing.py --dataset_path=\"../data/cleaned/KEGG_Metabolic_Relation_Network_(Directed)_Dataset/data.csv\" --goal_var_index=22",
        "python data_processing.py --dataset_path=\"../data/cleaned/Online_News_Popularity_Dataset/data.csv\" --goal_var_index=59",
        "python data_processing.py --dataset_path=\"../data/cleaned/Online_Video_Characteristics_and_Transcoding_Time_Dataset/data.csv\" --goal_var_index=18",
        "python data_processing.py --dataset_path=\"../data/cleaned/PM2.5_Data_of_Five_Chinese_Cities_Dataset/data.csv\" --goal_var_index=4",
        "python data_processing.py --dataset_path=\"../data/cleaned/Physicochemical_Properties_of_Protein_Tertiary_Structure_Dataset/data.csv\" --goal_var_index=0",
        "python data_processing.py --dataset_path=\"../data/cleaned/Relative_location_of_CT_slices_on_axial_axis_Dataset/data.csv\" --goal_var_index=384",
        "python data_processing.py --dataset_path=\"../data/cleaned/UJIIndoorLoc_Dataset/data.csv\" --goal_var_index=520",
        "python data_processing.py --dataset_path=\"../data/cleaned/Parkinsons_Telemonitoring_Dataset/data.csv\" --goal_var_index=0",
        "python data_processing.py --dataset_path=\"../data/cleaned/SML2010_Dataset/data.csv\" --goal_var_index=0",
        "python data_processing.py --dataset_path=\"../data/cleaned/Communities_and_Crime_Unnormalized_Dataset/data.csv\" --goal_var_index=124",
        "python data_processing.py --dataset_path=\"../data/cleaned/Communities_and_Crime_Dataset/data.csv\" --goal_var_index=122",
        "python data_processing.py --dataset_path=\"../data/cleaned/YearPredictionMSD_Dataset/data.csv\" --goal_var_index=0",
        "python data_processing.py --dataset_path=\"../data/cleaned/Airfoil_Self-Noise_Dataset/data.csv\" --goal_var_index=5",
        "python data_processing.py --dataset_path=\"../data/cleaned/Air_Quality_Dataset/data.csv\" --goal_var_index=1",
        "python data_processing.py --dataset_path=\"../data/cleaned/Geographical_Original_of_Music_Dataset/data.csv\" --goal_var_index=116",
        "python data_processing.py --dataset_path=\"../data/cleaned/Parkinson_Speech_Dataset_with_Multiple_Types_of_Sound_Recordings_Dataset/data.csv\" --goal_var_index=26",
        "python data_processing.py --dataset_path=\"../data/cleaned/Concrete_Compressive_Strength_Dataset/data.csv\" --goal_var_index=8",
        "python data_processing.py --dataset_path=\"../data/cleaned/Seoul_Bike_Sharing_Demand_Dataset/data.csv\" --goal_var_index=0",
        "python data_processing.py --dataset_path=\"../data/cleaned/SGEMM_GPU_kernel_performance_Dataset/data.csv\" --goal_var_index=14",
        "python data_processing.py --dataset_path=\"../data/cleaned/Uber_location_price_data/data.csv\" --goal_var_index=0",
        "python data_processing.py --dataset_path=\"../data/cleaned/Electrical_Grid_Stability_Simulated_Dataset/data.csv\" --goal_var_index=12",
        "python data_processing.py --dataset_path=\"../data/cleaned/Production_quality/data.csv\" --goal_var_index=17",
        "python data_processing.py --dataset_path=\"../data/cleaned/Bias_correction_ucl/data.csv\" --goal_var_index=22"
    ]

    for c in cmd_list:
        print("Launching :\n" + str(c))
        start_time = time.time()
        os.system(c)
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
