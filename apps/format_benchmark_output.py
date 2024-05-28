import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def create_graph(data, kernel):
    values = np.array(list(data.values())).reshape(3, 3)

    x_labels = [1280, 2560, 5120]
    y_labels = [960, 1920, 3840]

    sns.set(font_scale=2.0)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_colormap", ["red", "lightgreen", "green"], N=256
    )
    ax = sns.heatmap(
        values,
        annot=True,
        fmt=".2f",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        vmin=0.8,
        vmax=1.2,
        cbar=False,
        clip_on=False,
        linewidths=2,
        linecolor='black',
        annot_kws={
            'color': 'black',
            'fontfamily': 'serif',
            'fontweight': 'regular'
        }
    )

    ax.axhline(y=0, color='k',linewidth=2)
    ax.axhline(y=3, color='k',linewidth=2)
    ax.axvline(x=0, color='k',linewidth=2)
    ax.axvline(x=3, color='k',linewidth=2)

    plt.title("Exo speedup over Halide")
    plt.xlabel("Out Width (pixels)")
    plt.ylabel("Out Height (pixels)")

    plt.subplots_adjust(bottom=0.20, left=.15)
    plt.savefig(f"{kernel}_speedup_heatmap.pdf", format="pdf")


def main():
    data = {}
    kernel = sys.argv[1]

    while line := sys.stdin.readline():
        print(repr(line))
        vals = line.strip().split(" ")
        W = vals[0]
        H = vals[1]

        vals = sys.stdin.readline().strip().split(" ")
        halide_t = float(vals[1])
        exo_t = float(vals[2])
        data[(W, H)] = halide_t / exo_t

        line = sys.stdin.readline()

    create_graph(data, kernel)


if __name__ == "__main__":
    main()
