from pathlib import Path

activities = ['Eating', 'Head Shake', 'Nodding', 'Walking\nw/Speaking', 'Speaking', 'Staying', 'Walking']
sensor_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

# TODO - adjust accordingly
PATH_PROJECT = Path('..').resolve()
PATH_DATASET = PATH_PROJECT / "dataset"
PATH_DATA_RAW = PATH_DATASET / "raw"
PATH_DATA_FEATURES = PATH_DATASET / "features"
PATH_PLOTS = PATH_PROJECT / "plots"

PATH_DATASET.mkdir(parents=True, exist_ok=True)
PATH_DATA_RAW.mkdir(parents=True, exist_ok=True)
PATH_DATA_FEATURES.mkdir(parents=True, exist_ok=True)
PATH_PLOTS.mkdir(parents=True, exist_ok=True)
