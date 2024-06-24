import pandas as pd
import numpy as np
from utils.constants import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# TODO ACC, GYR
# accelerometer=True, gyroscope=True,
def preprocess_dataframe(df, magnitude=True, derivative=True, grouping=False, scaling=""):
    current_features = []
    current_features += FEATURES_ORIGINAL

    df_res = df.copy(deep=True)
    
    if(magnitude):
        print("Calculating accelerometer magnitude...")
        df_res['Am'] = calc_magnitude(df_res[['Ax', 'Ay', 'Az']])
        print("Calculating gyroscope magnitude...")
        df_res['Gm'] = calc_magnitude(df_res[['Gx', 'Gy', 'Gz']])
        current_features += FEATURES_MAGNITUDE
        
    if(derivative):
        print("Calculating derivatives...")
        df_diff = calc_features_derivative(df_res, current_features)
        df_res = pd.concat([df_res, df_diff], axis=1)
        
        current_features += list(df_diff.columns.values)

    if (scaling):
        print("Scaling data...")
        if (scaling.startswith('Overall')):
            df_res.loc[:, current_features] = scale_data(df_res[current_features], scaling.strip('Overall'))
        else:
            df_res.loc[:, current_features] = df_res.groupby(["Participant"])[current_features].apply(lambda x: scale_data(x, scaling)).values
        
    print("Finished preprocessing")
    return df_res, current_features

def calc_features_derivative(df, features):
    new_features = [(f, "d"+f) for f in features]

    df_diff = df.groupby(["Participant", "Activity"])[features].diff().bfill().rename(dict(new_features), axis=1)

    return df_diff

def calc_magnitude(df):
    return np.sqrt(np.square(df).sum(axis=1))

def scale_data(df, scaler_name):
    df_scaled = pd.DataFrame(columns=df.columns, index=df.index)
    
    if scaler_name == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_name == 'RobustScaler':
        scaler = RobustScaler()
    else:
        return df
    
    df_scaled.loc[:, df.columns] = scaler.fit_transform(df)

    return df_scaled

# Moving average for a 1d segemnt
def my_moving_average(data, windows_size = 3): #samples
    return pd.DataFrame(data).rolling(windows_size).mean().values[windows_size:, 0]

# Moving average for the overall dataset
def moving_average(data):
    ma_signal = []
    for signal_num in range(data.shape[2]):
        ma_signal.append(np.apply_along_axis(my_moving_average, 1, data[:, :, signal_num]))

    return np.stack(ma_signal, axis=2)