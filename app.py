"""Kelvin's editor app (step 1: manual sliders, no AI yet).

Run it with:   python app.py
Then open the link it prints (usually http://127.0.0.1:7860).

Why Gradio? It turns a Python function into a web page with sliders and image
boxes in a few lines, so nobody has to learn web development to try ideas.
"""

import gradio as gr
import numpy as np
import torch
from PIL import Image

from kelvin.editor import SLIDERS, apply_edits
from kelvin.editor.histogram import histogram_figure

# Big phone photos have 12+ million pixels. Shrinking the preview keeps the
# sliders responsive. (Saving at full size can come later.)
PREVIEW_MAX_SIDE = 1024


def load_preview(image: Image.Image | None):
    """Shrink the uploaded photo once, and store it as numbers from 0 to 1."""
    if image is None:
        return None
    image = image.convert("RGB")
    image.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    return np.asarray(image, dtype=np.float32) / 255.0


def render(preview: np.ndarray | None, *slider_values):
    """Apply the slider values to the photo and redraw the histogram."""
    if preview is None:
        return None, None
    params = {s.name: v for s, v in zip(SLIDERS, slider_values)}
    # numpy is (height, width, 3); the editor wants (3, height, width)
    tensor = torch.from_numpy(preview).permute(2, 0, 1)
    with torch.no_grad():  # no AI training happening, so skip the gradient bookkeeping
        edited = apply_edits(tensor, params).permute(1, 2, 0).numpy()
    return edited, histogram_figure(edited, original=preview)


with gr.Blocks(title="Kelvin") as demo:
    gr.Markdown("# Kelvin\nUpload a photo and move the sliders. Watch how the histogram changes.")
    preview_state = gr.State(None)  # the shrunk original, kept between slider moves

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.Image(type="pil", label="Original")
            sliders = [
                gr.Slider(-1, 1, value=0, step=0.01, label=s.label, info=s.help)
                for s in SLIDERS
            ]
            reset = gr.Button("Reset all sliders")
        with gr.Column(scale=2):
            output = gr.Image(label="Edited", interactive=False)
            histogram = gr.Plot(label="Histogram (left = black, right = white)")

    upload.change(load_preview, inputs=upload, outputs=preview_state).then(
        render, inputs=[preview_state, *sliders], outputs=[output, histogram]
    )
    for slider in sliders:
        slider.change(render, inputs=[preview_state, *sliders], outputs=[output, histogram])
    reset.click(lambda: [0] * len(sliders), outputs=sliders)


if __name__ == "__main__":
    demo.launch()
