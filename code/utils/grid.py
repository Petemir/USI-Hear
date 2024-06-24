import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product
from utils.constants import *

PATH_GRID = PATH_EXPERIMENTS / "grid.csv"

def create_grid(path_grid=PATH_GRID):
    models = ["conv_net_1", "conv_net_2", "conv_net_3", "conv_net_4", "conv_net_5", "StresNet"]
    participants = ["P%02d"%n for n in range(1, 9)] + ["P10", "P11"]
    
    window_sizes = [d for d in range (1, 11)] + [15]
    sampling_rates = [5] + [d for d in range(10,51,10)]
    sampling_rates_subset = [10, 40, 50]
    window_overlaps = [round(n, 1) for n in np.linspace(0.1, 0.90, num=9)]
    window_overlaps_subset = [0.1, 0.5, 0.75, 0.9]

    empty = [0]
    features = list(product([0, 1], repeat=4)) # Presence/Absence of Acc, Gyr, Mag and Der
    features = [f for f in features if (f[0] or f[1])] # If neither Acc nor Gyr, nonsense to analyze Mag or Der (i.e., already considered when having both Acc and Gyr)
    features_subset = [(1, 1, 1, 1)]
    
    grid_search = {} 
    grid_search["full"] = list(product(models, participants, window_sizes, sampling_rates, window_overlaps, features))
    grid_search["hyperparameters"] = list(product(models, participants, window_sizes, sampling_rates, window_overlaps, features_subset))
    grid_search["window_size"] = list(product(models, participants, window_sizes, sampling_rates_subset, window_overlaps_subset, features_subset))
    # "None" to be filled with the best parameter from the previous step
    grid_search["sampling_rate"] = list(product(models, participants, empty, sampling_rates, window_overlaps_subset, features_subset))
    grid_search["window_overlap"] = list(product(models, participants, empty, empty, window_overlaps, features_subset))
    grid_search["features"] = list(product(models, participants, empty, empty, empty, features))

    # Explode features
    for grid in grid_search.keys():
        grid_search[grid] = [list(x[:5] + x[5]) for x in grid_search[grid]]

    final_grid = pd.concat([pd.DataFrame(map(lambda x: [grid] + x, grid_search[grid])) for grid in grid_search.keys()])
    final_grid.columns = ["grid_type", "model", "participant", "window_size", "sampling_rate", "window_overlap", "acc", "gyr", "mag", "der"]

    final_grid.sort_values(["grid_type", "sampling_rate", "window_size", "window_overlap", "acc", "gyr", "mag", "der", "model", "participant"], inplace=True)
    final_grid["grid_index"] = final_grid.groupby("grid_type").cumcount()

    final_grid = pd.concat([final_grid.iloc[:, 0], final_grid.iloc[:, -1], final_grid.iloc[:, 1:-1]], axis=1)
    final_grid.reset_index(drop=True, inplace=True)

    final_grid[["Accuracy", "Precision", "Recall", "F1", "AccuracyB", "TestLabels", "TestPredictions"]] = [None]*7

    final_grid.to_csv(path_grid)

def get_grid(path_grid=PATH_GRID):
    if (not path_grid.exists()):
        create_grid(path_grid)
        
    return pd.read_csv(path_grid, index_col=0)
    