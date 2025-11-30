Ian Stedham

This project was coded utilizing Python 3.11
The paper I was referecing did not provide any code, so all code in this project is produce by me.

XGBoost Model:
To run this project first run the xgboot_model.py file. This will produce 4 files: 
- cv_results_xgb.csv, this is a csv file containing the results of Cross Validation without utilizing SMOTE
- cv_results_xgb_smote.csv, this is a csv file containing the results of Cross Validation utilizing SMOTE on training data
- xgb_model.json, this is a json file containing the XGBoost model trained on the 80/20 train/test split without utlizing SMOTE
- xgb_model_smote.json, this is a json file containing the XGBoost model trained on the 80/20 train/test split SMOTE on training data

Next run the xgboost_analysis.py file. This will automatically load the Cross Validation results as well as the XGBoost models and run analysis
based on these results. Various plots and graphs will appear based on the models trained on the 80/20 split, both one that utlizes SMOTE 
and one that does not, then box plots for the 1-sample t-test referencing the cross validation results will appear. Along with this
various standard statistics will be printed to the terminal.

NEAT Neural-Network Model:
To run this project first run the neat_model.py file. This will produce 4 files:
- final_genome.pkl
- best_validation_genome.pkl
- scaler.pkl
- stats.pkl
The final_genome and best_validation_genome are neural network files from the network created at the end of evolution and the best one achieved
from the validation training set respectively. The scaler and stats file are used for analysis

Next run the neat_analysis.py. This will load the final_genome neural network and run it through several tests as well as produce all of the
plots shown in the paper.

Notes:
- For all seeds and random states, 17 was used