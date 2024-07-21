import bz2
import time
import mlflow
import joblib
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna.visualization as ov
from bokeh.io import export_svgs
from bokeh.layouts import row, gridplot
from bokeh.plotting import figure, show
from bokeh.palettes import viridis, cividis
from bokeh.models import ColumnDataSource, Range1d
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_val_score
from textwrap import wrap
from matplotlib.cm import get_cmap
from sklearn.impute import SimpleImputer
from optuna.exceptions import TrialPruned
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse, unquote
import os

import ipywidgets

from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
                                   OneHotEncoder, OrdinalEncoder, PolynomialFeatures, 
                                   QuantileTransformer,  PowerTransformer)
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder

from optuna.integration import MLflowCallback
from sklearn.experimental import enable_hist_gradient_boosting
from category_encoders import TargetEncoder, BinaryEncoder, HashingEncoder, HelmertEncoder

from tqdm import tqdm
from mlflow.tracking import MlflowClient

from bokeh.palettes import viridis, cividis
viridis_palette = viridis(256)





def get_scaler(trial):
    """

    Suggest and return a scaler based on the trial's categorical suggestion.

    This function uses an optimization trial to suggest a scaler type from a predefined list.
    Depending on the suggested scaler type, it returns an appropriate scaler instance.

    Parameters:
        trial (optuna.trial.Trial): The optimization trial object used to suggest the scaler type
                                    and relevant hyperparameters.

    Returns:
        Scaler: An instance of a scaler corresponding to the suggested scaler type.
                This could be an instance of StandardScaler, MinMaxScaler, MaxAbsScaler,
                RobustScaler, QuantileTransformer, or PowerTransformer.

    Raises:
        ValueError: If an unknown scaler type is suggested by the trial.
    
    """
    scaler_type = trial.suggest_categorical('scaler', ['standard', 'minmax', 'maxabs', 'robust', 'quantile', 'power'])
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    elif scaler_type == 'maxabs':
        return MaxAbsScaler()
    elif scaler_type == 'robust':
        return RobustScaler()
    elif scaler_type == 'quantile':
        return QuantileTransformer(output_distribution='normal', n_quantiles=trial.suggest_int('n_quantiles', 10, 1000))
    elif scaler_type == 'power':
        return PowerTransformer()
    else:
        raise ValueError("Unknown scaler type")


def get_encoder(trial):
    """

    Suggest and return an encoder based on the trial's categorical suggestion.

    This function uses an optimization trial to suggest an encoder type from a predefined list.
    Depending on the suggested encoder type, it returns an appropriate encoder instance.

    Parameters:
        trial (optuna.trial.Trial): The optimization trial object used to suggest the encoder type.

    Returns:
        Encoder: An instance of an encoder corresponding to the suggested encoder type.
                 This could be an instance of OneHotEncoder, OrdinalEncoder, TargetEncoder,
                 BinaryEncoder, HashingEncoder, or HelmertEncoder.

    Raises:
        ValueError: If an unknown encoder type is suggested by the trial.    
    """
    encoder_type = trial.suggest_categorical('encoder', ['onehot', 'ordinal', 'target', 'binary', 'hashing', 'helmert'])
    if encoder_type == 'onehot':
        return OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    elif encoder_type == 'ordinal':
        return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    elif encoder_type == 'target':
        return TargetEncoder(handle_missing='value', handle_unknown='value')
    elif encoder_type == 'binary':
        return BinaryEncoder(handle_unknown='value', handle_missing='value')
    elif encoder_type == 'hashing':
        return HashingEncoder()  # HashingEncoder inherently handles unknowns by design
    elif encoder_type == 'helmert':
        return HelmertEncoder(handle_unknown='value', handle_missing='value')
    else:
        raise ValueError("Unknown encoder type")

def get_imputer(strategy, fill_value=None):
    """

    Return an imputer based on the specified strategy and fill value.

    This function creates and returns an instance of `SimpleImputer` from scikit-learn,
    configured with the given strategy and optionally a fill value if the strategy is 'constant'.

    Parameters:
        strategy (str): The imputation strategy. Can be one of 'mean', 'median', 'most_frequent', or 'constant'.
        fill_value (optional): The value to use for imputing missing values when the strategy is 'constant'.
                               If not provided, the default fill value for 'constant' strategy is used.

    Returns:
        SimpleImputer: An instance of `SimpleImputer` configured with the specified strategy and fill value.    
    """
    if strategy == 'constant' and fill_value is not None:
        return SimpleImputer(strategy=strategy, fill_value=fill_value)
    else:
        return SimpleImputer(strategy=strategy)



# Define the objective function with MLflow logging
def objective(trial, model_class, model_param_suggestion, is_stacking=False, base_models=None, categorical_cols=None, numerical_cols=None, X=None, y=None,strat_col='putting_distance_to_pin_bins'):
    """
    strat_col = 'putting_distance_to_pin_bins'
    Define the objective function for hyperparameter optimization with MLflow logging.

    This function performs hyperparameter optimization using Optuna. It suggests various preprocessing
    steps and model parameters, builds a machine learning pipeline, and evaluates the model using nested
    cross-validation. Results and parameters are logged to MLflow.

    Parameters:
        trial (optuna.trial.Trial): The optimization trial object.
        model_class (class): The model class to be instantiated with suggested parameters.
        model_param_suggestion (function): A function that suggests model parameters for the given trial.
        is_stacking (bool, optional): Whether to use a stacking regressor. Defaults to False.
        base_models (list of tuple, optional): List of base models for stacking, required if is_stacking is True.
        categorical_cols (list of str, optional): List of column names corresponding to categorical features.
        numerical_cols (list of str, optional): List of column names corresponding to numerical features.
        X (DataFrame, optional): The feature matrix.
        y (Series, optional): The target variable.

    Returns:
        float: The mean outer score from nested cross-validation.

    Raises:
        optuna.TrialPruned: If the trial is pruned by Optuna.
        ValueError: If an unknown encoder type or scaler type is suggested.
        Exception: For any other exceptions, logs the error message to MLflow and returns None.

    """
    try:
        with mlflow.start_run(nested=True) as run:
            #as run and line bloew new
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            trial.set_user_attr("mlflow_name",run.info.run_name)
            
            # Suggest scaler type
            scaler = get_scaler(trial)
            # Suggest encoder type
            encoder = get_encoder(trial)

            numerical_imputer_strategy = trial.suggest_categorical(
                'numerical_imputer_strategy', ['mean', 'median', 'most_frequent', 'constant']
            )
            categorical_imputer_strategy = trial.suggest_categorical(
                'categorical_imputer_strategy', ['most_frequent', 'constant']
            )

            numerical_imputer = get_imputer(numerical_imputer_strategy, fill_value=-1)
            categorical_imputer = get_imputer(categorical_imputer_strategy, fill_value='missing')

            # Define the ColumnTransformer with the chosen scaler and imputer for numerical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[
                        ('imputer', numerical_imputer),
                        ('scaler', scaler)
                    ]), numerical_cols),
                    ('cat', Pipeline(steps=[
                        ('imputer', categorical_imputer),
                        ('encoder', encoder)
                    ]), categorical_cols)
                ],
                remainder='drop'  # Drop columns not specified in transformers
            )

            # Feature selection (optional)
            feature_selector_type = trial.suggest_categorical('feature_selector', ['none', 'kbest', 'model'])
            if feature_selector_type == 'kbest':
                from sklearn.feature_selection import SelectKBest, f_regression
                feature_selector = SelectKBest(score_func=f_regression, k=trial.suggest_int('k', 5, 20))
                preprocessor.transformers.append(('feature_selector', feature_selector, []))  # Add empty list for columns
            elif feature_selector_type == 'model':
                from sklearn.feature_selection import SelectFromModel
                feature_selector = SelectFromModel(estimator=GradientBoostingRegressor(n_estimators=50))
                preprocessor.transformers.append(('feature_selector', feature_selector, []))  # Add empty list for columns

            # Polynomial features (optional)
            poly_degree = trial.suggest_int('poly_degree', 1, 3)
            if poly_degree > 1:
                preprocessor.transformers.append(('poly', PolynomialFeatures(degree=poly_degree, include_bias=False), []))  # Add empty list for columns

            # Suggest hyperparameters for the model
            params = model_param_suggestion(trial)
            
            if is_stacking:
                # Create the stacking model
                final_estimator_params = model_param_suggestion(trial)
                final_estimator = model_class(**final_estimator_params, random_state=42)
                model = StackingRegressor(estimators=base_models, final_estimator=final_estimator)
            else:
                # Create the pipeline
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', model_class(**params, random_state=42))
                ])

            # Use StratifiedKFold with 'putting_distance_to_pin_bins' for nested cross-validation
            outer_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            outer_scores = []

            for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_kf.split(X, X[strat_col])):
                X_outer_train, X_outer_val = X.iloc[outer_train_idx], X.iloc[outer_val_idx]
                y_outer_train, y_outer_val = y.iloc[outer_train_idx], y.iloc[outer_val_idx]
                #strat_col = 'putting_distance_to_pin_bins'

                # Use StratifiedKFold with 'putting_distance_to_pin_bins' for inner cross-validation
                inner_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                val_scores = []

                for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(X_outer_train, X_outer_train[strat_col])):
                    X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
                    y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_val_idx]

                    model.fit(X_inner_train, y_inner_train)
                    preds = model.predict(X_inner_val)
                    val_score = mean_absolute_error(y_inner_val, preds)
                    val_scores.append(val_score)

                    # Report the intermediate value
                    trial.report(val_score, inner_fold)

                    # Check for pruning
                    if trial.should_prune():
                        mlflow.log_metric('pruned', 1)
                        raise optuna.TrialPruned()

                mean_inner_val_score = np.mean(val_scores)
                outer_scores.append(mean_inner_val_score)

                # Log the outer fold score
                mlflow.log_metric(f'outer_fold_{outer_fold}_score', mean_inner_val_score)

            mean_outer_score = np.mean(outer_scores)

            # Log parameters and metrics to MLflow
            params_to_log = trial.params
            for param_name, param_value in params_to_log.items():
                try:
                    mlflow.log_param(param_name, param_value)
                except mlflow.exceptions.MlflowException:
                    pass  # Ignore if the parameter is already logged

            mlflow.log_metric('mean_outer_score', mean_outer_score)

            # Define the artifact path and log it as a trial attribute
            artifact_path = "model.joblib.bz2"
            trial.set_user_attr("artifact_path", artifact_path)
            
            # Log trained model
            with bz2.BZ2File(artifact_path, 'wb', compresslevel=9) as f:
                joblib.dump(model, f)
            mlflow.log_artifact(artifact_path)

            # Return mean_outer_score
            return mean_outer_score

    except optuna.TrialPruned:
        mlflow.log_metric('pruned', 1)
        raise
    except Exception as e:
        # Log the error if needed
        mlflow.log_metric('failed', 1)
        mlflow.log_param('error_message', str(e))
        print(f"Trial failed with exception: {e}")
        return None


def get_best_model(experiment_name):
    """
    
    Register the best model to the MLflow Model Registry and transition it to production.

    This function retrieves the best model's artifact from MLflow, registers it as a new model version
    in the MLflow Model Registry, and transitions the model version to the "Production" stage.

    Parameters:
        experiment_name (optuna): The best trial containing user attributes with MLflow run details.

    Returns:
        tuple: A tuple containing the best model object and the model version information.

    Raises:
        mlflow.exceptions.RestException: If there is an issue with the MLflow REST API.    
    """
    # log and register best model to prod
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Get the experiment ID
    experiment_id = experiment.experiment_id

    # Search for the best run based on a specific metric
    best_run = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metric.mean_outer_score ASC"],
        max_results=10
    ).iloc[0]

    url = best_run.artifact_uri
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract and decode the path from the URL
    path = unquote(parsed_url.path)

    # Convert the path to a proper file path
    file_path = os.path.normpath(path)

    # Append the desired file name
    final_path = os.path.join(file_path[1:], 'model.joblib.bz2')

    with bz2.BZ2File(final_path, 'rb') as f:
        best_model = joblib.load(f)

    return best_model