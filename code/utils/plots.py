import matplotlib.pyplot as plt
from pathlib import Path

def plot_gyroscope(df, timestamps, title=None, output_path=None):
    title = title + " - Gyroscope" if title else "Gyroscope"

    plot_3axis(df, timestamps, columns=["Gx", "Gy", "Gz"], label="rotational speed (deg/sec)", title=title, output_path=output_path)

def plot_accelerometer(df, timestamps, title=None, output_path=None):
    title = title + " - Accelerometer" if title else "Accelerometer"

    plot_3axis(df, timestamps, columns=["Ax", "Ay", "Az"], label="acceleration (g)", title=title, output_path=output_path)

def plot_3axis(df, timestamps, columns, label="", title=None, output_path=None):
    assert(len(columns) == 3)

    fig, ax = plt.subplots(2,2, figsize=(22,11))
    fig.suptitle(title, fontsize=16)
    colors = ["blue", "red", "green"]

    ax[1,1].set_title('All')
    ax[1,1].set_xlabel('time (ns)')
    ax[1,1].set_ylabel(label)

    for idx,col in enumerate(columns):
        i = round(idx/2)
        j = idx%2

        ax[i,j].plot((timestamps.values - timestamps.values.min()), df[col].values, color=colors[idx], alpha=0.9)
        ax[i,j].set_title(col)
        ax[i,j].set_xlabel('time (ns)')
        ax[i,j].set_ylabel(label)

        ax[1,1].plot((timestamps.values - timestamps.values.min()), df[col].values, color=colors[idx], alpha=0.9)

    if(output_path):
        fig.savefig(output_path, bbox_inches="tight", dpi=600)

        fig.savefig(output_path.parent / ("transparent_" + output_path.name), bbox_inches="tight", dpi=600, transparent=True)
    fig.show()