import os
import pandas as pd
from pathlib import Path
from utils.constants import *

def get_dataset(relative_timestamps=False, method="interpolation", sampling_rate=None):
    if (not sampling_rate):
        path_participants = [f for f in PATH_DATA_RAW.glob('*') if f.is_dir()]
    else:
        path_participants = [f for f in (PATH_DATA_RESAMPLED / method / str(sampling_rate)).glob("*") if f.is_dir()]

    # TODO - IMPORTANT?
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

    # # Remove unwanted signals
    # if(accelerometer and gyroscope):
    #     FEATURES = FEATURES_ORIGINAL
    #     N_FEATURES = len(FEATURES)
    # elif(accelerometer and not gyroscope):
    #     FEATURES = FEATURES_ACCELEROMETER
    #     N_FEATURES = len(FEATURES)
    #     df.drop(FEATURES_GYROSCOPE, axis=1, inplace=True)
    # elif(not accelerometer and gyroscope):
    #     FEATURES = FEATURES_GYROSCOPE
    #     N_FEATURES = len(FEATURES)
    #     df.drop(FEATURES_ACCELEROMETER, axis=1, inplace=True)
    # else:
    #     if not (only_der or only_mag):
    #         FEATURES = [] 
    #         N_FEATURES = len(FEATURES)
    #         return
    #     else:
    #         FEATURES = FEATURES_ORIGINAL
    #         N_FEATURES = len(FEATURES)
    
    # # Calculate magnitudes of signals
    # if(magnitude and accelerometer) or only_mag:
    #     df['Am'] = get_magnitude(df[['Ax', 'Ay', 'Az']])
    #     FEATURES = FEATURES + FEATURES_MAGNITUDE_ACCELEROMETER
    # if(magnitude and gyroscope) or only_mag:
    #     df['Gm'] = get_magnitude(df[['Gx', 'Gy', 'Gz']])
    #     FEATURES = FEATURES + FEATURES_MAGNITUDE_GYROSCOPE
    # N_FEATURES = len(FEATURES)

    # # Calculate derivatives
    # if(derivative or only_der):
    #     features_to_add = []
    #     for feature in FEATURES:
    #         new_feature = 'd'+feature
    #         features_to_add.append(new_feature)

    #         for participant in df.Username.unique():
    #             df_participant = df[df.Username == participant]
    #             for activity in df_participant.Activity.unique():
    #                 df_feature = df.loc[(df.Username == participant) & (df.Activity == activity), feature]

    #                 df.loc[(df.Username == participant) & (df.Activity == activity), new_feature] = df_feature.diff().bfill()

    #     FEATURES = FEATURES + features_to_add
    #     N_FEATURES = len(FEATURES)
    
    # #keep only der or mag
    # if only_der:
    #     for feature in FEATURES:
    #         if not feature.startswith('d'):
    #             df.drop(feature, axis=1, inplace=True)
    #             N_FEATURES -= 1
    #             FEATURES.remove(feature)

    # elif only_mag:
    #     for feature in FEATURES:
    #         if feature != "Am" and feature != "Gm":
    #             df.drop(feature, axis=1, inplace=True)
    #             N_FEATURES -= 1
    #             FEATURES.remove(feature)
    

    # if(scaler):
    #     if (scaler.startswith('Overall')):
    #         df.loc[:, FEATURES] = scale_data(df, scaler.strip('Overall'))
    #     else:
    #         for participant in df.Username.unique():
    #             df_user = df[df.Username == participant]

    #             df_user_scaled = scale_data(df_user, scaler)
    #             df.loc[df_user.index, FEATURES] = df_user_scaled

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
        ACTIVATION_FUNCTION = 'softmax'
        LOSS_FUNCTION = 'categorical_crossentropy'
        LABELS = ['Not Interacting', 'Talking', 'Interacting']
        N_OUTPUTS = 3
        
        df.loc[df.Activity == 'Speaking', ['Activity', 'Label']] = ['Talking', 1]
        df.loc[df.Activity == 'Walking\nw/Speaking', ['Activity', 'Label']] = ['Talking', 1]

        df.loc[df.Activity == 'Nodding', ['Activity', 'Label']] = ['Interacting', 2]
        df.loc[df.Activity == 'Head Shake', ['Activity', 'Label']] = ['Interacting', 2]
        
        df.loc[df.Activity == 'Staying', ['Activity', 'Label']] = ['Not Interacting', 0]
        df.loc[df.Activity == 'Eating', ['Activity', 'Label']] = ['Not Interacting', 0]
        df.loc[df.Activity == 'Walking', ['Activity', 'Label']] = ['Not Interacting', 0]
                
    elif (grouping == 'InteractingNotInteracting'):
        ACTIVATION_FUNCTION = 'sigmoid'
        LOSS_FUNCTION = 'binary_crossentropy'
        LABELS = ['Not Interacting', 'Interacting']
        N_OUTPUTS = 2

        df.loc[df.Activity == 'Speaking', ['Activity', 'Label']] = ['Interacting', 1]
        df.loc[df.Activity == 'Walking\nw/Speaking', ['Activity', 'Label']] = ['Interacting', 1]
        df.loc[df.Activity == 'Nodding', ['Activity', 'Label']] = ['Interacting', 1]
        df.loc[df.Activity == 'Head Shake', ['Activity', 'Label']] = ['Interacting', 1]
        
        df.loc[df.Activity == 'Staying', ['Activity', 'Label']] = ['Not Interacting', 0]
        df.loc[df.Activity == 'Eating', ['Activity', 'Label']] = ['Not Interacting', 0]
        df.loc[df.Activity == 'Walking', ['Activity', 'Label']] = ['Not Interacting', 0]
        
    elif (grouping == 'InteractingNotInteracting2'):
        ACTIVATION_FUNCTION = 'sigmoid'
        LOSS_FUNCTION = 'binary_crossentropy'
        LABELS = ['Not Interacting', 'Interacting']
        N_OUTPUTS = 2

        df.loc[df.Activity == 'Speaking', ['Activity', 'Label']] = ['Interacting', 1]
        df.loc[df.Activity == 'Walking\nw/Speaking', ['Activity', 'Label']] = ['Interacting', 1]
        df.loc[df.Activity == 'Nodding', ['Activity', 'Label']] = ['Interacting', 1]

        df.loc[df.Activity == 'Head Shake', ['Activity', 'Label']] = ['Not Interacting', 0]
        df.loc[df.Activity == 'Staying', ['Activity', 'Label']] = ['Not Interacting', 0]
        df.loc[df.Activity == 'Eating', ['Activity', 'Label']] = ['Not Interacting', 0]
        df.loc[df.Activity == 'Walking', ['Activity', 'Label']] = ['Not Interacting', 0]
                
    elif (grouping == 'TalkingNotTalking'):
        ACTIVATION_FUNCTION = 'sigmoid'
        LOSS_FUNCTION = 'binary_crossentropy'
        LABELS = ['Not Talking', 'Talking']
        N_OUTPUTS = 2

        df.loc[df.Activity == 'Speaking', ['Activity', 'Label']] = ['Talking', 1]
        df.loc[df.Activity == 'Walking\nw/Speaking', ['Activity', 'Label']] = ['Talking', 1]

        df.loc[df.Activity == 'Nodding', ['Activity', 'Label']] = ['Not Talking', 0]
        df.loc[df.Activity == 'Head Shake', ['Activity', 'Label']] = ['Not Talking', 0]    
        df.loc[df.Activity == 'Staying', ['Activity', 'Label']] = ['Not Talking', 0]
        df.loc[df.Activity == 'Eating', ['Activity', 'Label']] = ['Not Talking', 0]
        df.loc[df.Activity == 'Walking', ['Activity', 'Label']] = ['Not Talking', 0]

    elif (grouping == 'NoWalking'):
        ACTIVATION_FUNCTION = 'softmax'
        LOSS_FUNCTION = 'categorical_Crossentropy'
        LABELS = ['Staying', 'Head Shake', 'Speaking', 'Nodding', 'Eating']
        N_OUTPUTS = 5

        df.drop(df[df.Activity == 'Walking'].index, inplace=True)
        df.drop(df[df.Activity == 'Walking\nw/Speaking'].index, inplace=True)
        df.loc[df.Label == 6, 'Label'] = 0 # Label "Staying" as 0
        
    elif (grouping == 'InteractingNotInteractingNoWalking'):
        ACTIVATION_FUNCTION = 'sigmoid'
        LOSS_FUNCTION = 'binary_crossentropy'
        LABELS = ['Not Interacting', 'Interacting']
        N_OUTPUTS = 2

        df.loc[df.Activity == 'Speaking', ['Activity', 'Label']] = ['Interacting', 1]
        df.loc[df.Activity == 'Nodding', ['Activity', 'Label']] = ['Interacting', 1]

        df.loc[df.Activity == 'Head Shake', ['Activity', 'Label']] = ['Not Interacting', 0]    
        df.loc[df.Activity == 'Staying', ['Activity', 'Label']] = ['Not Interacting', 0]
        df.loc[df.Activity == 'Eating', ['Activity', 'Label']] = ['Not Interacting', 0]

        df.drop(df[df.Activity == 'Walking'].index, inplace=True)
        df.drop(df[df.Activity == 'Walking\nw/Speaking'].index, inplace=True)
    elif (grouping == 'InteractingNotInteractingTalkingNoWalking'):
        ACTIVATION_FUNCTION = 'softmax'
        LOSS_FUNCTION = 'categorical_Crossentropy'
        LABELS = ['Not Interacting', 'Talking', 'Interacting']
        N_OUTPUTS = 3

        df.loc[df.Activity == 'Speaking', ['Activity', 'Label']] = ['Talking', 1]

        df.loc[df.Activity == 'Nodding', ['Activity', 'Label']] = ['Interacting', 2]
        df.loc[df.Activity == 'Head Shake', ['Activity', 'Label']] = ['Interacting', 2]

        df.loc[df.Activity == 'Staying', ['Activity', 'Label']] = ['Not Interacting', 0]
        df.loc[df.Activity == 'Eating', ['Activity', 'Label']] = ['Not Interacting', 0]

        df.drop(df[df.Activity == 'Walking'].index, inplace=True)
        df.drop(df[df.Activity == 'Walking\nw/Speaking'].index, inplace=True)
    elif (grouping == 'BinaryHeadShake'):
        ACTIVATION_FUNCTION = 'sigmoid'
        LOSS_FUNCTION = 'binary_crossentropy'
        LABELS = ['Other', 'Head Shake']
        N_OUTPUTS = 2

        df.loc[df.Activity == 'Head Shake', ['Activity', 'Label']] = ['Head Shake', 1]

        df.loc[df.Activity == 'Speaking', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Walking\nw/Speaking', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Nodding', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Staying', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Eating', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Walking', ['Activity', 'Label']] = ['Other', 0]

    elif (grouping == 'BinaryNodding'):
        ACTIVATION_FUNCTION = 'sigmoid'
        LOSS_FUNCTION = 'binary_crossentropy'
        LABELS = ['Other', 'Nodding']
        N_OUTPUTS = 2

        df.loc[df.Activity == 'Nodding', ['Activity', 'Label']] = ['Nodding', 1]
        
        df.loc[df.Activity == 'Speaking', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Walking\nw/Speaking', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Head Shake', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Staying', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Eating', ['Activity', 'Label']] = ['Other', 0]
        df.loc[df.Activity == 'Walking', ['Activity', 'Label']] = ['Other', 0]

    else:
        ACTIVATION_FUNCTION = 'softmax'
        LOSS_FUNCTION = 'categorical_crossentropy'
        LABELS = LABELS_ORIGINAL
        N_OUTPUTS = 7
    
    return df

# Converts data into segments of size STEP
def segment_data_ml(df, column_name='Activity'):
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

# Converts data into segments of size window_size with window_step overlap
def segment_data_dl(df, window_size, window_step, column_name='Activity'):
    segments = []
    labels = []
    if (window_step % 1):
        leap_step = int(1 / (window_step % 1))
    else:
        leap_step = False
    window_step = floor(window_step)
    
    for user in df.Username.unique():
        df_by_user = df[df.Username == user]
        
        for column_value in df[column_name].unique():
            df_by_column = df_by_user[df_by_user[column_name] == column_value]

            i = 0
            j = 0
            # for i in range(0, len(df_by_column) - window_size, window_step):
            stop = len(df_by_column) - window_size# - 1
            while (i <= stop):
              values = []
              if j and leap_step and (j % leap_step == 0) and (i+1 <= stop):
                i += 1

              for feature in FEATURES:
                values.append(df_by_column[feature].values[i: i + window_size])

              label = stats.mode(df_by_column[column_name][i: i + window_size])[0][0]
              segments.append(values)
              labels.append(label)

              if (i >= stop):
                break

              j += 1
              i += window_step

              if (i > stop):
                i = stop

    return segments, labels
    
def get_x_y_dl(df, window_size, window_step):
    segments, labels_indices = segment_data(df, window_size, window_step, column_name='Label')
    reshaped_segment = np.asarray(segments, dtype= np.float32).reshape(-1, window_size, N_FEATURES)
    labels = np.eye(N_OUTPUTS)[labels_indices]

    return reshaped_segment, labels

def get_x_y_ml(df):
    segments, labels_indices, users = segment_data(df, column_name='Label')
    reshaped_segment = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    labels = np.eye(7)[labels_indices]

    return reshaped_segment, labels, users