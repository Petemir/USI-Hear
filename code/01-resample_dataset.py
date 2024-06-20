from utils.constants import *

import pandas as pd
from scipy.interpolate import interp1d
from math import floor
import numpy as np

PATH_DATA_RESAMPLED = PATH_DATASET / "raw_resampled"
PATH_DATA_RESAMPLED.mkdir(parents=True, exist_ok=True)

for path_participant in [p_dir for p_dir in PATH_DATA_RAW.glob("*") if p_dir.is_dir()]:
    participant = path_participant.name
    print(participant)
    for file_act in path_participant.glob('*csv'):
        activity = file_act.stem
        print(activity)
        csv_path = PATH_DATA_RAW / path_participant / file_act
        
        df = pd.read_csv(csv_path, index_col=0).drop(columns="OFF")
        pairing = df[["Label", "Activity"]].apply(pd.unique)
        assert(len(pairing), 1), "More than one activity/label pair recorded in this file"
        df_label, df_activity = pairing.values[0]

        assert(activity == df_activity), "DataFrame activity (%s) does not match filename activity (%s)" %(df_activity, activity)

        sample_timestamp_min = df.Timestamp.min()
        sample_timestamp_max = df.Timestamp.max()
        recording_duration = sample_timestamp_max - sample_timestamp_min
        recording_samples_no = len(df)

        # Calculate an approximate recording frequency        
        freq_df = recording_samples_no / recording_duration

        # Sanity check: verify differences between the minimum distance between samples, the maximum distance between samples, and the calculated recording frequency 
        sample_distance = df.Timestamp.diff()
        sample_distance_max = sample_distance.max()
        sample_distance_min = sample_distance.min()

        assert(abs((1/sample_distance_max)-freq_df) < 0.01), "%s: Difference between maximum sample distance and frequency is bigger than 0.1" % path_activity
        assert(abs((1/sample_distance_min)-freq_df) < 0.01), "%s: Difference between minimum sample distance and frequency is bigger than 0.1" % path_activity
        assert(abs(sample_distance_max-sample_distance_min) < 0.000001), "%s: Difference between minimum sample distance and frequency is bigger than 1e-6" % path_activity

        # Previous differences are not big, so I can reindex to an evenly-distributed index.
        period_df_mean = round(np.mean([1/freq_df, sample_distance_max, sample_distance_min]), 3)
        df_new_index = pd.date_range(start=pd.to_datetime(sample_timestamp_min, unit='s'), end=pd.to_datetime(sample_timestamp_max, unit='s'), freq=str(period_df_mean)+"s")

        assert(abs(len(df) - len(df_new_index)) <= 1), "New dataframe loses too much data"
        df_new_length = min(len(df), len(df_new_index))

        df = df[:df_new_length]
        df.set_index(df_new_index[:df_new_length], inplace=True, drop=True)

        for freq_new in [5, 10, 20, 30, 40, 50]:
            print(freq_new, "Hz")
            assert(freq_new <= (1/period_df_mean)), "Impossible to downsample, chosen downsampling frequency is higher than the dataframe's current frequency"
                        
            period_new = round((1/freq_new), 3)
            
            common_time_vector = np.arange(sample_timestamp_min, sample_timestamp_max, period_new)
            interpolated_data = []
            for sensor in sensor_names:
                interp_func = interp1d(df.Timestamp, df.loc[:, sensor], kind='cubic', fill_value="extrapolate")
                interpolated_data.append(interp_func(common_time_vector))

            df_resampled = pd.DataFrame(data=np.stack(interpolated_data, axis=-1), index=common_time_vector, columns=sensor_names)
            df_resampled.reset_index(names="Timestamp", inplace=True)
            df_resampled[["Participant", "Activity", "Label"]] = [participant, df_activity, df_label]

            path_output = PATH_DATA_RESAMPLED / "interpolation" / str(freq_new) / participant / file_act.name
            path_output.parent.mkdir(parents=True, exist_ok=True)
            df_resampled.to_csv(path_output, index=None)

            for method in ["polynomial", "spline"]:
                failed = True
                backwards = False
                rounded_decimals = 3

                # For some specific recording sessions, when freq_new is e.g. 30Hz and period_new is e.g. 0.033, interpolation fails, but it works with e.g. 0.0333 or 0.03333. Therefore, change the number of precision decimals until it works.
                # If also these fail, until reaching 5 rounded decimals, then add a small bias, e.g. for 0.03333 add 0.000001 (i.e., a 1 after the last decimal)
                while(failed):
                    if (not backwards):
                        period_new = round((1/freq_new), rounded_decimals)
                    else:
                        period_new = round(round(1/freq_new, rounded_decimals) + pow(10, -(rounded_decimals+1)), rounded_decimals+1)

                    path_output = PATH_DATA_RESAMPLED / method / str(freq_new) / participant / file_act.name                        
                    try:
                        df_resampled = df[sensor_names].resample(str(period_new)+'s').interpolate(method=method, order=3).dropna()
                        df_resampled.reset_index(names="Timestamp", inplace=True)
                        df_resampled[["Participant", "Activity", "Label"]] = [participant, df_activity, df_label]

                        path_output.parent.mkdir(parents=True, exist_ok=True)
                        df_resampled.to_csv(path_output, index=None)
                        failed = False
                    except:
                        failed = True
                        pass

                    if(df_resampled.empty):
                        failed = True

                    if (rounded_decimals >= 5):
                        backwards = True
                    elif (rounded_decimals <= 2):
                        backwards = False
                        failed = True
                        print(path_output, "failed")
                        break

                    if (not backwards):
                        rounded_decimals += 1
                    else:
                        rounded_decimals -= 1