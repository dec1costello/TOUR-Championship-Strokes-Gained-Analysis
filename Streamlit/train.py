import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import HashingEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module='category_encoders.base_contrast_encoder')
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

import optuna
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import bz2
import joblib
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show
from bokeh.io import export_png
from bokeh.layouts import column
from bokeh.palettes import viridis, cividis
import pickle


# Create viridis palette
viridis_palette = viridis(256)


################################
########## DATA PREP ##########
################################

# Load in the data
result_df = pd.read_csv(r"Streamlit/Temp Data/light_weight_model.csv")
#connect to supabase with github secret keys

# Pre-processing
numeric_features = ["distance_to_pin",]
categorical_features = ['lie']

# Pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ("scaler", StandardScaler()) # Standardize features by removing the mean and scaling to unit variance
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("HashingEncoder", HashingEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ("num_transform", numeric_transformer, numeric_features),
    ("cat_transform", categorical_transformer, categorical_features)
])

feature_cols = categorical_features + numeric_features
X = result_df.loc[:, feature_cols]

target_cols = ['strokes_to_hole_out']
y = result_df.loc[:, target_cols]

# Extract the columns for stratification
stratify_cols = ['lie']
stratify_data = result_df[stratify_cols]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=42,stratify=stratify_data)

# Apply the preprocessor to the training and validation data
X_train_transformed = preprocessor.fit_transform(X_train)
X_valid_transformed = preprocessor.transform(X_valid)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters for the GradientBoostingRegressor
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0)
    }
    
    # Create a GradientBoostingRegressor model with the trial's parameters
    model = GradientBoostingRegressor(**param, random_state=42)
    
    # Fit the model on the training data
    model.fit(X_train_transformed, y_train)
    
    # Predict on the validation set
    y_pred = model.predict(X_valid_transformed)
    
    # Calculate mean squared error on the validation set
    mse = mean_squared_error(y_valid, y_pred)
    
    return mse

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Output the best parameters found
print("Best hyperparameters: ", study.best_params)

# Train the GradientBoostingRegressor using the best hyperparameters
best_model = GradientBoostingRegressor(**study.best_params, random_state=42)
best_model.fit(X_train_transformed, y_train)
with open('light_weight_expected_strokes.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Evaluate the model on the validation set
y_pred = best_model.predict(X_valid_transformed)
mse = mean_squared_error(y_valid, y_pred)
print(f"Mean Squared Error on validation set: {mse:.4f}")

Mean_Squared_Error = f"Mean Squared Error on validation set: {mse:.4f}"

# Write scores to a file
with open(r"Streamlit/Training Report/results.txt", 'w') as outfile:
        outfile.write(Mean_Squared_Error)

##########################################
############ PLOT RESIDUALS  #############
##########################################
y_valid = np.array(y_valid).flatten()  # Flatten in case y_valid is multidimensional
# Calculate residuals
residuals = y_valid - y_pred

# Set the parameters of the normal distribution
mean, std = 0, 1  # You can adjust these parameters based on your desired shape

# Generate synthetic data from the normal distribution
resid_scores = residuals

# Fit the normal distribution to the synthetic data
params = norm.fit(resid_scores)

# Now you can use the parameters for further analysis or generating random samples

# Calculate histogram
hist, edges = np.histogram(resid_scores, density=True, bins=88)

# Generate the x values for the fitted normal distribution
x = np.linspace(min(resid_scores),
                max(resid_scores), 
                100)

# Calculate the probability density function for the fitted normal distribution
pdf = norm.pdf(x, *params)

# Create a Bokeh figure
p = figure(height=450, 
           width=600,
           title=f'Residual Normality \n\nMean: {params[0]:.2f}, Std Dev: {params[1]:.2f}', 
           x_axis_label='Residual Scores', 
           y_axis_label='Density')

# Plot the histogram
p.quad(top=hist, 
       bottom=0, 
       left=edges[:-1], 
       right=edges[1:], 
       line_color='white', 
       fill_color=viridis_palette[180], 
       alpha=0.9)

# Plot the fitted normal distribution
p.line(x, 
       pdf, 
       line_color=viridis_palette[10], 
       line_width=4, 
       legend_label='Fitted Normal Distribution')

p.x_range.start = -1.25
p.x_range.end = 1.25
p.y_range.start = 0

export_png(p, filename=r"Streamlit/Training Report/residuals.png")
