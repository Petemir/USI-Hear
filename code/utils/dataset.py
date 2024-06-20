import os
import pandas as pd
from pathlib import Path
from utils.constants import *

def get_dataset(relative_timestamps=False):
    path_participants = [f for f in PATH_DATA_RAW.glob('*') if f.is_dir()]
    
    global FEATURES
    global N_FEATURES
    
    df_list = []
   
    for path_participant in path_participants:
        path_activity_files = list(path_participant.glob("*csv"))
        assert(len(path_activity_files) == 7), "Missing activity at %s" % (path_participant)
        
        for path_activity_file in path_activity_files:
            df_tmp = pd.read_csv(path_activity_file, index_col=0, header=0)
            df_tmp.drop(columns="OFF", errors="ignore", inplace=True)
            # TODO -> Check the several minutes gaps in data
            # TODO -> interpolate and fix missing data
            # df.Timestamp = pd.to_datetime(df.Timestamp*1000000)
            df_tmp["Participant"] = path_participant.name

            if(relative_timestamps):
                df_tmp.loc[:, "Timestamp"] = df_tmp.Timestamp.values - df_tmp.Timestamp.values.min()
            df_list.append(df_tmp)

    df = pd.concat(df_list, axis = 0, sort = True, ignore_index = True)
    df = df[["Participant", "Activity", "Label", "Timestamp", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"]]

    df.Activity = df.Activity.replace("Speak and Walk", "Walking\nw/Speaking")

    df["Label"] = df["Label"].astype("int32")
    df['Ax'] = pd.to_numeric(df['Ax'])
    df['Ay'] = pd.to_numeric(df['Ay'])
    df['Az'] = pd.to_numeric(df['Az'])
    df['Gx'] = pd.to_numeric(df['Gx'])
    df['Gy'] = pd.to_numeric(df['Gy'])
    df['Gz'] = pd.to_numeric(df['Gz'])

    df.sort_values(["Participant", "Activity", "Timestamp"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    return df

# TODO - make sure this global variable thingy works...
# TODO - REMOVE?
# Group labels into different classes: 
# - InteractingNotInteractingTalking
# - InteractingNotInteracting
# - InteractingNotInteracting2
# - TalkingNotTalking
def group_labels(df, grouping=None):
    global N_OUTPUTS
    global ACTIVATION_FUNCTION
    global LOSS_FUNCTION
    global LABELS
    if (grouping == 'InteractingNotInteractingTalking'):
        df.loc[df.Activity == 'Speaking', 'Activity'] = 'Talking'
        df.loc[df.Label == 2, 'Label'] = 1
        df.loc[df.Activity == 'Speak and Walk', 'Activity'] = 'Talking'
        df.loc[df.Label == -1, 'Label'] = 1
        df.loc[df.Activity == 'Nodding', 'Activity'] = 'Interacting'
        df.loc[df.Label == 3, 'Label'] = 2
        df.loc[df.Activity == 'Head Shake', 'Activity'] = 'Interacting'
        df.loc[df.Label == 1, 'Label'] = 2

        df.loc[df.Activity == 'Staying', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 4, 'Label'] = 0
        df.loc[df.Activity == 'Eating', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 5, 'Label'] = 0
        df.loc[df.Activity == 'Walking', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 6, 'Label'] = 0
        
        N_OUTPUTS = 3
        ACTIVATION_FUNCTION = 'softmax'
        LOSS_FUNCTION = 'categorical_crossentropy'
        LABELS = ['Not Interacting', 'Talking', 'Interacting']
    elif (grouping == 'InteractingNotInteracting'):
        df.loc[df.Activity == 'Speaking', 'Activity'] = 'Interacting'
        df.loc[df.Label == 2, 'Label'] = 1
        df.loc[df.Activity == 'Speak and Walk', 'Activity'] = 'Interacting'
        df.loc[df.Label == -1, 'Label'] = 1
        df.loc[df.Activity == 'Nodding', 'Activity'] = 'Interacting'
        df.loc[df.Label == 3, 'Label'] = 1
        df.loc[df.Activity == 'Head Shake', 'Activity'] = 'Interacting'
        df.loc[df.Label == 1, 'Label'] = 1

        df.loc[df.Activity == 'Staying', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 4, 'Label'] = 0
        df.loc[df.Activity == 'Eating', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 5, 'Label'] = 0
        df.loc[df.Activity == 'Walking', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 6, 'Label'] = 0
        
        N_OUTPUTS = 2
        ACTIVATION_FUNCTION = 'sigmoid'
        LOSS_FUNCTION = 'binary_crossentropy'
        LABELS = ['Not Interacting', 'Interacting']
    elif (grouping == 'InteractingNotInteracting2'):
        df.loc[df.Activity == 'Speaking', 'Activity'] = 'Interacting'
        df.loc[df.Label == 2, 'Label'] = 1
        df.loc[df.Activity == 'Speak and Walk', 'Activity'] = 'Interacting'
        df.loc[df.Label == -1, 'Label'] = 1
        df.loc[df.Activity == 'Nodding', 'Activity'] = 'Interacting'
        df.loc[df.Label == 3, 'Label'] = 1

        df.loc[df.Activity == 'Head Shake', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 1, 'Label'] = 0
        df.loc[df.Activity == 'Staying', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 4, 'Label'] = 0
        df.loc[df.Activity == 'Eating', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 5, 'Label'] = 0
        df.loc[df.Activity == 'Walking', 'Activity'] = 'Not Interacting'
        df.loc[df.Label == 6, 'Label'] = 0
        
        N_OUTPUTS = 2
        ACTIVATION_FUNCTION = 'sigmoid'
        LOSS_FUNCTION = 'binary_crossentropy'
        LABELS = ['Not Interacting', 'Interacting']
    elif (grouping == 'TalkingNotTalking'):
        df.loc[df.Activity == 'Speaking', 'Activity'] = 'Talking'
        df.loc[df.Label == 2, 'Label'] = 1
        df.loc[df.Activity == 'Speak and Walk', 'Activity'] = 'Talking'
        df.loc[df.Label == -1, 'Label'] = 1
        
        df.loc[df.Activity == 'Nodding', 'Activity'] = 'Not Talking'
        df.loc[df.Label == 3, 'Label'] = 0
        df.loc[df.Activity == 'Head Shake', 'Activity'] = 'Not Talking'
        df.loc[df.Label == 1, 'Label'] = 0
        df.loc[df.Activity == 'Staying', 'Activity'] = 'Not Talking'
        df.loc[df.Label == 4, 'Label'] = 0
        df.loc[df.Activity == 'Eating', 'Activity'] = 'Not Talking'
        df.loc[df.Label == 5, 'Label'] = 0
        df.loc[df.Activity == 'Walking', 'Activity'] = 'Not Talking'
        df.loc[df.Label == 6, 'Label'] = 0
        
        N_OUTPUTS = 2
        ACTIVATION_FUNCTION = 'sigmoid'
        LOSS_FUNCTION = 'binary_crossentropy'
        LABELS = ['Not Talking', 'Talking']
    else:
        ACTIVATION_FUNCTION = 'softmax'
        LOSS_FUNCTION = 'categorical_crossentropy'
        N_OUTPUTS = 7

    return df

# Converts data into segments of size STEP
def segment_data(df, column_name='Activity'):
    segments = []
    labels = []
    users = []
    for user in df.Username.unique():
        df_by_user = df[df.Username == user]
        
        for column_value in df[column_name].unique():
            df_by_column = df_by_user[df_by_user[column_name] == column_value]

            for i in range(0, len(df_by_column) - N_TIME_STEPS, STEP):
                values = []
                for feature in FEATURES:
                    values.append(df_by_column[feature].values[i: i + N_TIME_STEPS])

                label = stats.mode(df_by_column[column_name][i: i + N_TIME_STEPS])[0][0]
                segments.append(values)
                labels.append(label)
                users.append(user)
                
    return segments, labels, users

def scale_data(df, scaler_name):
    df_scaled = pd.DataFrame(columns=FEATURES, index=df.index)
    
    if scaler_name == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_name == 'RobustScaler':
        scaler = RobustScaler()
    else:
        return df[FEATURES]
    
    df_scaled.loc[:, FEATURES] = scaler.fit_transform(df.loc[:, FEATURES])

    return df_scaled

def get_x_y(df):
    segments, labels_indices, users = segment_data(df, column_name='Label')
    reshaped_segment = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    labels = np.eye(7)[labels_indices]

    return reshaped_segment, labels, users

#moving average foe a 1-d segemnt
def my_moving_average(data, windows_size = 3): #samples
    return pd.DataFrame(data).rolling(windows_size).mean().values[windows_size:,0]

#moving average for the voerall dataset
def moving_average(data):
    ma_signal = []
    for signal_num in range(data.shape[2]):
        ma_signal.append(np.apply_along_axis(my_moving_average, 1, data[:,:,signal_num]))
    return np.stack(ma_signal, axis=2)