from download_day import download_day
from load_saved_fd4 import load_saved_fd4
from scrape_aircraft_list import scrape_aircraft_list
from get_perf_model_typecodes import get_perf_model_typecodes 
from match_icao_model import match_icao_model
from process_airport_list import process_airport_list
from generate_flightpath import generate_flightpath
from plot_flightpaths import plot_flightpaths
from get_engine_data import get_engine_data
from perf_model_powerplant_parser import perf_model_powerplant_parser
from match_engine_to_emissions_db import match_engine_to_emissions_db
from process_month_emissions import process_month_emissions
from get_era5_wind import get_era5_wind
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import tqdm
from time import sleep
from geopy import distance
import requests
from icet import icet
from bffm2 import bffm2
import multiprocessing
from functools import partial

# User Inputs:
start_time_str       = '2023-01-01T00:00:00Z'
stop_time_str        = '2023-12-31T23:59:59Z'
query_limit          = 15e4
send_notification    = True
make_plot            = True
output_dir           = "/scratch/omg28/Data/"

# Convert start and stop times to datetime objects
start_time_simple = pd.to_datetime(start_time_str).strftime("%Y-%m-%d")
stop_time_simple = pd.to_datetime(stop_time_str).strftime("%Y-%m-%d")
analysis_year = pd.to_datetime(start_time_str).year

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import pickle
import os

max_distances = pd.read_csv(f'{output_dir}/aircraftdb/{start_time_simple}_to_{stop_time_simple}_typecode_max_distances.csv')
performance_and_emissions_model = pd.read_pickle('performance_and_emissions_model.pkl')

num_altitudes = 6; num_distances = 12
typecodes = performance_and_emissions_model['typecode'].unique()
model_save_dir = 'saved_models_nox_flux'
os.makedirs(model_save_dir, exist_ok=True)

def get_X_y(typecode):
    max_dist_row = max_distances[max_distances['typecode'] == typecode]
    if max_dist_row.empty: return None, None
    max_distance_km = max_dist_row['max_distance_km'].values[0]
    max_cruise_altitude_ft = performance_and_emissions_model.loc[performance_and_emissions_model['typecode'] == typecode, 'cruise_Ceiling'].max() * 1e2
    if pd.isnull(max_cruise_altitude_ft) or pd.isnull(max_distance_km): return None, None

    cruise_altitudes_ft = np.linspace(18000, max_cruise_altitude_ft, num_altitudes)
    flight_distances_km = np.logspace(np.log10(200), np.log10(max_distance_km), num_distances)
    X, y = [], []
    for alt_ft in cruise_altitudes_ft:
        for d_km in flight_distances_km:
            try:
                fp = generate_flightpath(typecode, d_km, performance_and_emissions_model, cruise_altitude_ft=alt_ft)
                total_nox = 0; total_time = 0
                for nox_key, t_key in zip(['NOx_climb_0_5','NOx_climb_5_10','NOx_climb_10_15','NOx_climb_15_24','NOx_climb_ceil'],['t_climb_0_5','t_climb_5_10','t_climb_10_15','t_climb_15_24','t_climb_ceil']):
                    total_nox += fp.get('climb', {}).get(nox_key, 0)
                    total_time += fp.get('climb', {}).get(t_key, 0)
                for nox_key, t_key in zip(['NOx_descent_ceil','NOx_descent_24_15','NOx_descent_15_10','NOx_descent_10_5','NOx_descent_5_0'],['t_descent_ceil','t_descent_24_15','t_descent_15_10','t_descent_10_5','t_descent_5_0']):
                    total_nox += fp.get('descent', {}).get(nox_key, 0)
                    total_time += fp.get('descent', {}).get(t_key, 0)
                total_nox += fp.get('cruise', {}).get('NOx_cruise', 0)
                total_time += fp.get('cruise', {}).get('t_cruise', 0)
                if total_time > 0:
                    mean_nox_flux = total_nox / total_time
                    X.append([d_km, alt_ft])
                    y.append(mean_nox_flux)
            except Exception: continue
    X = np.array(X); y = np.array(y)
    if len(y) < 8: return None, None
    return X, y

# XGBoost with 5-fold CV and Grid Search
xgb_results = {}
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.05, 0.1, 0.2]
}

for typecode in tqdm.tqdm(typecodes):
    X, y = get_X_y(typecode)
    if X is None: continue
    
    model = XGBRegressor(random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(model, xgb_param_grid, cv=kf, scoring='r2', n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    y_pred = cross_val_predict(best_model, X, y, cv=kf)
    r2s = cross_val_score(best_model, X, y, cv=kf, scoring='r2')
    rmses = []
    for train_idx, test_idx in kf.split(X):
        best_model.fit(X[train_idx], y[train_idx])
        y_test_pred = best_model.predict(X[test_idx])
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], y_test_pred)))
    xgb_results[typecode] = {
        'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
        'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
        'n_samples': len(y), 'best_params': grid.best_params_
    }
    # Save model
    os.makedirs(model_save_dir, exist_ok=True)
    fout = os.path.join(model_save_dir, f'xgb_{typecode}.ubj')
    best_model.save_model(str(fout))
    print(f"[XGBoost CV] {typecode}: R2={np.mean(r2s):.3f}±{np.std(r2s):.3f}, RMSE={np.mean(rmses):.3f}±{np.std(rmses):.3f}, N={len(y)}, Best={grid.best_params_}")
    