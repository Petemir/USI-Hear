from pathlib import Path

RANDOM_SEED=42

VERBOSITY=3

ACTIVITIES = ['Eating', 'Head Shake', 'Nodding', 'Walking\nw/Speaking', 'Speaking', 'Staying', 'Walking']
SENSORS = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

# TODO - adjust accordingly
# PATH_PROJECT = Path('..').resolve()
PATH_PROJECT = Path("/home/matias/Projects/USI-HEAR/final_version/")

PATH_DATASET = PATH_PROJECT / "dataset"
PATH_DATA_RAW = PATH_DATASET / "raw"
PATH_DATA_RESAMPLED = PATH_DATASET / "raw_resampled"
PATH_DATA_FEATURES = PATH_DATASET / "features"
PATH_PLOTS = PATH_PROJECT / "plots"
PATH_EXPERIMENTS = PATH_PROJECT / "experiments"

PATHS = {
    "project": PATH_PROJECT,
    "dataset": PATH_DATASET,
    "data_raw": PATH_DATA_RAW,
    "data_resampled": PATH_DATA_RESAMPLED,
    "data_features": PATH_DATA_FEATURES,
    "plots": PATH_PLOTS,
    "experiments": PATH_EXPERIMENTS
}

for path in PATHS:
    PATHS[path].mkdir(parents=True, exist_ok=True)

FEATURES_ACCELEROMETER = ['Ax', 'Ay', 'Az']
FEATURES_GYROSCOPE = ['Gx', 'Gy', 'Gz']
FEATURES_ORIGINAL = FEATURES_ACCELEROMETER + FEATURES_GYROSCOPE

FEATURES_MAGNITUDE_ACCELEROMETER = ['Am']
FEATURES_MAGNITUDE_GYROSCOPE = ['Gm']
FEATURES_MAGNITUDE = FEATURES_MAGNITUDE_ACCELEROMETER + FEATURES_MAGNITUDE_GYROSCOPE

def def_v_print(verbosity=0):
  """
    Return the verbose printing function

    Args:
      * verbosity: verbosity level (1 for ERROR, 2 for WARN, 3 for INFO)
  """
  _v_print = lambda *args, **kwargs: None

  if (verbosity):
    def _v_print(*print_args, **print_kwargs): 
      print_level = print_kwargs.get('level') if print_kwargs.get('level') else 1

      if (print_level > (3 - verbosity)):
        print(*print_args)

  return _v_print

v_print = def_v_print(VERBOSITY)