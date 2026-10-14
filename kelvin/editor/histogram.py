"""Draw the histogram: the brightness graph editors look at.

How to read it: the left side is black, the right side is white. A tall bar
means lots of pixels have that brightness.
  - Everything bunched on the left  -> photo is too dark
  - A spike stuck against the right -> bright parts are "blown out" (pure white, no detail)
  - Nothing near either edge        -> photo looks flat / washed out
"""

import matplotlib

matplotlib.use("Agg")  # draw to an image without opening a window
import matplotlib.pyplot as plt
import numpy as np

from kelvin.ui import tokens as t

BINS = 64


def histogram_figure(edited: np.ndarray, original: np.ndarray | None = None):
    """`edited` and `original` are (H, W, 3) arrays with values from 0 to 1.
    The original is drawn as a dashed grey outline so you can see what changed.

    It's drawn on the dark panel colour, not white. That stops the graph being
    the brightest thing on screen, which would throw off your eye for the photo."""
    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
    fig.patch.set_facecolor(t.SURFACE)
    ax.set_facecolor(t.SURFACE)
    edges = np.linspace(0, 1, BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    def counts(values):
        c, _ = np.histogram(values, bins=edges)
        return c / max(c.max(), 1)  # scale so the tallest bar is 1

    for channel, colour in enumerate([t.HIST_RED, t.HIST_GREEN, t.HIST_BLUE]):
        ax.fill_between(centres, counts(edited[..., channel]), color=colour, alpha=0.2, linewidth=0)
    if original is not None:
        ax.plot(centres, counts(_luma(original)), color=t.MUTED, linestyle="--", linewidth=1.2, label="Before")
    ax.plot(centres, counts(_luma(edited)), color=t.TEXT, linewidth=1.6, label="After")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1], ["Black", "Mid", "White"])
    ax.tick_params(colors=t.MUTED, length=0, labelsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    legend = ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    for text in legend.get_texts():
        text.set_color(t.TEXT_SOFT)
    fig.tight_layout()
    plt.close(fig)  # Gradio keeps its own reference; this stops memory piling up
    return fig


def _luma(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
