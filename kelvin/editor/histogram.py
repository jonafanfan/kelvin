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

BINS = 64


def histogram_figure(edited: np.ndarray, original: np.ndarray | None = None):
    """`edited` and `original` are (H, W, 3) arrays with values from 0 to 1.
    The original is drawn as a faint grey outline so you can see what changed."""
    fig, ax = plt.subplots(figsize=(5, 2.4), dpi=100)
    edges = np.linspace(0, 1, BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    def counts(values):
        c, _ = np.histogram(values, bins=edges)
        return c / max(c.max(), 1)  # scale so the tallest bar is 1

    if original is not None:
        ax.plot(centres, counts(_luma(original)), color="grey", linestyle="--", linewidth=1, label="before")

    for channel, colour in enumerate(["red", "green", "blue"]):
        ax.fill_between(centres, counts(edited[..., channel]), color=colour, alpha=0.15)
    ax.plot(centres, counts(_luma(edited)), color="black", linewidth=1.5, label="after")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1], ["black", "mid", "white"])
    ax.legend(loc="upper center", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    plt.close(fig)  # Gradio keeps its own reference; this stops memory piling up
    return fig


def _luma(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
