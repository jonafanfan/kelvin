"""Kelvin's editor app (step 1: manual sliders, no AI yet).

Run it with:   python app.py
Then open the link it prints (usually http://127.0.0.1:7860).

Why Gradio? It turns a Python function into a web page with sliders and image
boxes in a few lines, so nobody has to learn web development to try ideas.
The look (colours, fonts, slider style) lives in kelvin/ui/, so this file is
only about what the app does.
"""

from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from kelvin.editor import SLIDERS, apply_edits
from kelvin.editor.histogram import histogram_figure
from kelvin.ui.theme import CSS, HEAD, build_theme, header_html

# Big phone photos have 12+ million pixels. Shrinking the preview keeps the
# sliders responsive. (Saving at full size can come later.)
PREVIEW_MAX_SIDE = 1024


def load_preview(path: str | None):
    """Shrink the uploaded photo once, and store it as numbers from 0 to 1."""
    if path is None:
        return None, header_html()
    image = Image.open(path).convert("RGB")
    image.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    return np.asarray(image, dtype=np.float32) / 255.0, header_html(Path(path).name)


def render(preview: np.ndarray | None, *slider_values):
    """Apply the slider values to the photo and redraw the histogram."""
    if preview is None:
        return None, None
    params = {s.name: v for s, v in zip(SLIDERS, slider_values)}
    # numpy is (height, width, 3); the editor wants (3, height, width)
    tensor = torch.from_numpy(preview).permute(2, 0, 1)
    with torch.no_grad():  # no AI training happening, so skip the gradient bookkeeping
        edited = apply_edits(tensor, params).permute(1, 2, 0).numpy()
    before_after = (_to_8bit(preview), _to_8bit(edited))
    return before_after, histogram_figure(edited, original=preview)


def _to_8bit(img: np.ndarray) -> np.ndarray:
    """Screens show whole numbers from 0 to 255, not 0.0 to 1.0."""
    return (img * 255 + 0.5).astype(np.uint8)


with gr.Blocks(title="Kelvin") as demo:
    preview_state = gr.State(None)  # the shrunk original, kept between slider moves

    with gr.Row(elem_id="k-header", equal_height=True):
        header = gr.HTML(header_html())
        with gr.Row(elem_id="k-header-buttons"):
            reset = gr.Button("Reset all", variant="secondary", scale=0)
            upload = gr.UploadButton("Open photo", file_types=["image"], variant="secondary", scale=0)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320, elem_id="k-sliders"):
            sliders = [
                gr.Slider(-1, 1, value=0, step=0.01, label=s.label, info=s.help, elem_id=f"k-{s.name}")
                for s in SLIDERS
            ]
        with gr.Column(scale=2, min_width=480):
            photo = gr.ImageSlider(
                label="Before / after (drag the line to compare)", elem_id="k-photo", interactive=False
            )
            histogram = gr.Plot(label="Histogram", elem_id="k-histogram")

    upload.upload(load_preview, inputs=upload, outputs=[preview_state, header]).then(
        render, inputs=[preview_state, *sliders], outputs=[photo, histogram]
    )
    for slider in sliders:
        slider.change(render, inputs=[preview_state, *sliders], outputs=[photo, histogram])
    reset.click(lambda: [0] * len(sliders), outputs=sliders)


if __name__ == "__main__":
    # Gradio 6 takes the look (theme, extra CSS, fonts) at launch, not in gr.Blocks.
    demo.launch(theme=build_theme(), css=CSS, head=HEAD)
