# BURNOUT--A-MotoGP-Datathon

MotoGP Lap Time Prediction
Overview
This project aims to predict lap times (Lap_Time_Seconds) for MotoGP races using machine learning. The dataset includes features like circuit length, rider statistics, track conditions, and tire compounds. A LightGBM model is trained to predict lap times, achieving a validation RMSE of approximately 0.35–0.37. The project includes exploratory data analysis (EDA), feature engineering, model training, and evaluation.

Objective: Predict Lap_Time_Seconds for MotoGP races.
Model: LightGBM (LGBMRegressor) with early stopping.
Evaluation Metric: RMSE.
Validation RMSE: ~0.35–0.37.

Dataset
The dataset consists of MotoGP race data with the following key features:

Numerical Features: Circuit_Length_km, Laps, Grid_Position, Avg_Speed_kmh, Humidity_%, Track_Temperature_Celsius, etc.
Categorical Features: category_x (MotoGP, Moto2, Moto3), Tire_Compound_Front, Tire_Compound_Rear, rider, track, etc.
Target: Lap_Time_Seconds.

Files:

train.csv: Training data with features and target.
test.csv: Test data for predictions.
val.csv: Validation data.
sample_submission.csv: Template for submission (Unique ID, Lap_Time_Seconds).

Note: The dataset is not included in this repository due to size and privacy constraints. You can replace it with your own MotoGP dataset or source it from a similar competition.
Project Structure

motogp_prediction.ipynb: Main Jupyter Notebook with the full pipeline (EDA, feature engineering, modeling, evaluation).
README.md: Project documentation (this file).
solution.csv: Output file with predictions (generated after running the notebook).

Approach
1. Data Preprocessing

Encoded Track_Condition using LabelEncoder for feature engineering.
Imputed missing numerical values with median.
Set categorical columns to category dtype for LightGBM.
Removed outliers in Lap_Time_Seconds using the IQR method.

2. Exploratory Data Analysis (EDA)

Visualized the distribution of Lap_Time_Seconds.
Analyzed relationships:
Lap_Time_Seconds by race category (category_x).
Avg_Speed_kmh vs. Lap_Time_Seconds, colored by Track_Condition.
Correlation heatmap for numerical features.
Lap_Time_Seconds by Grid_Position and Tire_Compound_Front.


Explored rider performance metrics (wins, podiums, years_active).

3. Feature Engineering

Created new features:
Speed_Degradation: Avg_Speed_kmh * Tire_Degradation_Factor_per_Lap.
Corners_per_Km: Corners_per_Lap / Circuit_Length_km.
Experience_Score: years_active * (wins + podiums + 0.5 * with_points).
Track_Complexity: Corners_per_Lap * Circuit_Length_km * (1 + Humidity_% / 100).
Temp_Interaction: Track_Temperature_Celsius * Track_Condition_Encoded.
Tire_Match: Binary feature (1 if Tire_Compound_Front == Tire_Compound_Rear).
Pit_Stop_Effect: Pit_Stop_Duration_Seconds / Laps.
Grid_Position_Speed: Grid_Position / Avg_Speed_kmh.


Dropped irrelevant columns (e.g., weather) and retained key categorical features (e.g., track, rider).

4. Model Training

Model: LightGBM (LGBMRegressor).
Parameters:
n_estimators=500, learning_rate=0.025, max_depth=12, num_leaves=100.
subsample=0.85, colsample_bytree=0.8, reg_alpha=1.5, reg_lambda=2.0.
random_state=42, verbosity=-1 (silent mode).


Categorical Features: Handled via categorical_feature parameter.
Early Stopping: Used lightgbm.early_stopping(stopping_rounds=100) to prevent overfitting.
Split train.csv into training and validation sets (80-20 split).

5. Evaluation

Validation RMSE: ~0.35–0.37.
Visualized predicted vs. actual Lap_Time_Seconds and feature importance.

6. Prediction

Generated predictions on test.csv.
Rounded Lap_Time_Seconds to 3 decimal places.
Saved results to solution.csv with columns Unique ID and Lap_Time_Seconds.

Key Insights

MotoGP races have shorter lap times than Moto2/Moto3 due to faster bikes.
Wet tracks increase lap times, emphasizing tire choice.
Front-row grid positions correlate with faster lap times.
Experienced riders (more wins/podiums) perform better.
Matching tire compounds may improve stability.
Features like Speed_Degradation and Pit_Stop_Effect enhance prediction accuracy.

Requirements

Python 3.8+
Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm



Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm

How to Run

Clone the Repository:git clone <repository-url>
cd <repository-name>


Prepare Data:
Place train.csv, test.csv, val.csv, and sample_submission.csv in the project directory.
Alternatively, modify the notebook to load your own dataset.


Run the Notebook:
Open motogp_prediction.ipynb in Jupyter Notebook:jupyter notebook motogp_prediction.ipynb


Execute all cells to preprocess data, perform EDA, train the model, and generate solution.csv.


View Results:
Check solution.csv for predictions.
Review EDA plots and feature importance in the notebook.



Future Improvements

Perform hyperparameter tuning using GridSearchCV or Optuna.
Add more domain-specific features (e.g., track elevation, rider fatigue).
Experiment with ensemble models (e.g., XGBoost, CatBoost).
Handle missing data with advanced imputation techniques.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or contributions, feel free to open an issue or pull request on GitHub.
