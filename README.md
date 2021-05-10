# ClassificationForRegression
Combinaison of ensemblist methods for regression

python 3.7

### Launch commands examples :

Pre-process :
> python data_processing.py --dataset_path="../data/raw/Combined_Cycle_Power_Plant_Dataset/Folds5x2_pp.csv" --output_path="../data/processed/Combined_Cycle_Power_Plant_Dataset" --delimiter=',' --decimal='.'  --goal_var_index=4 --split_method="equal_freq" --output_classes="below_threshold" --log_lvl="info"

Extract features :
> python feature_extraction.py --dataset_folder="../data/processed/Combined_Cycle_Power_Plant_Dataset/" --output_path="../data/extracted_features/Combined_Cycle_Power_Plant_Dataset/" --classifier="RandomForests" --class_cols=8 --log_lvl="info"

Compute metrics :
> python compute_test_metrics.py --dataset_folder="../data/processed/Combined_Cycle_Power_Plant_Dataset/" --regressor="RandomForests" --log_lvl="info"