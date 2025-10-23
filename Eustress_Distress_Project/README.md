Ian Stedham

This project was coded utilizing Python 3.11
The paper I was referecing did not provide any code, so all code in this project is produce by me.

To run this project first run the model.py file. This will produce 4 files: 
- cv_results_xgb.csv, this is a csv file containing the results of Cross Validation without utilizing SMOTE
- cv_results_xgb_smote.csv, this is a csv file containing the results of Cross Validation utilizing SMOTE on training data
- xgb_model.json, this is a json file containing the XGBoost model trained on the 80/20 train/test split without utlizing SMOTE
- xgb_model_smote.json, this is a json file containing the XGBoost model trained on the 80/20 train/test split SMOTE on training data

Next run the analysis.py file. This will automatically load the Cross Validation results as well as the XGBoost models and run analysis
based on these results. Various plots and graphs will appear based on the models trained on the 80/20 split, both one that utlizes SMOTE 
and one that does not, then box plots for the 1-sample t-test referencing the cross validation results will appear. Along with this
various standard statistics will be printed to the terminal.

Notes:
- For all seeds and random states, 17 was used