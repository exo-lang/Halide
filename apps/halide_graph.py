import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.font_manager


plt.rcParams['figure.figsize'] = 4*(3.33/6), 3.5*(3.33/6)

rc_fonts = {
    "font.family": "serif",
    'font.serif': 'Linux Libertine O',
    "pdf.fonttype" : 42,
    "ps.fonttype" : 42
}
plt.rcParams.update(rc_fonts)

def create_graph(data, kernel):
    values = np.array(list(data.values())).reshape(3, 3)

    x_labels = [1280, 2560, 5120]
    y_labels = [960, 1920, 3840]

    sns.set(font='Linux Libertine O')
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
        linewidths=1,
        linecolor='black',
        annot_kws={
            'color': 'black',
            'fontweight': 'bold'
        }
    )

    ax.axhline(y=0, color='k',linewidth=2)
    ax.axhline(y=3, color='k',linewidth=2)
    ax.axvline(x=0, color='k',linewidth=2)
    ax.axvline(x=3, color='k',linewidth=2)

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    #sns.set(font_scale=1.4)

    plt.title("Runtime of Halide / AIRxo")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")

    plt.subplots_adjust(bottom=0.26, left=.25)
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
