"""Kelvin's editor app: manual sliders plus an Auto button.

Auto uses the trained AI (step 4) if its weights are in checkpoints/. They
aren't in git, so on a fresh copy run  python -m kelvin.model.train  first.
Without them, Auto falls back to the hand-written rules from step 2.

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

from kelvin.auto import ai_suggest, suggest
from kelvin.editor import SLIDERS, apply_edits
from kelvin.editor.histogram import histogram_figure
from kelvin.model import load_trained
from kelvin.model.train import CHECKPOINT
from kelvin.ui.theme import CSS, HEAD, build_theme, header_html, reasons_html

# Big phone photos have 12+ million pixels. Shrinking the preview keeps the
# sliders responsive. (Saving at full size can come later.)
PREVIEW_MAX_SIDE = 1024

AI = load_trained(CHECKPOINT) if CHECKPOINT.exists() else None
AI_TITLE = "What the AI changed"
RULES_NOTE = f"No trained AI found ({CHECKPOINT}), so these come from the hand-written rules."


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
    with torch.no_grad():  # no AI training happening, so skip the gradient bookkeeping
        edited = apply_edits(_to_tensor(preview), params).permute(1, 2, 0).numpy()
    before_after = (_to_8bit(preview), _to_8bit(edited))
    return before_after, histogram_figure(edited, original=preview)


def auto(preview: np.ndarray | None):
    """Let the AI (or, without one, the rules) set every slider, then redraw and explain."""
    if preview is None:
        return [gr.update()] * len(SLIDERS) + [None, None, reasons_html()]
    with torch.no_grad():
        s = ai_suggest(_to_tensor(preview), AI) if AI else suggest(_to_tensor(preview))
    values = [s.params[slider.name] for slider in SLIDERS]
    rows = [(slider.label, s.params[slider.name], s.reasons[slider.name]) for slider in SLIDERS if slider.name in s.reasons]
    panel = reasons_html(rows, AI_TITLE) if AI else reasons_html(rows, note=RULES_NOTE)
    return [*values, *render(preview, *values), panel]


def reset(preview: np.ndarray | None):
    values = [0] * len(SLIDERS)
    return [*values, *render(preview, *values), reasons_html()]


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """numpy is (height, width, 3); the editor wants (3, height, width)."""
    return torch.from_numpy(img).permute(2, 0, 1)


def _to_8bit(img: np.ndarray) -> np.ndarray:
    """Screens show whole numbers from 0 to 255, not 0.0 to 1.0."""
    return (img * 255 + 0.5).astype(np.uint8)


with gr.Blocks(title="Kelvin") as demo:
    preview_state = gr.State(None)  # the shrunk original, kept between slider moves

    with gr.Row(elem_id="k-header", equal_height=True):
        header = gr.HTML(header_html())
        with gr.Row(elem_id="k-header-buttons"):
            auto_button = gr.Button("Auto", variant="primary", scale=0, elem_id="k-auto")
            reset_button = gr.Button("Reset all", variant="secondary", scale=0)
            upload = gr.UploadButton("Open photo", file_types=["image"], variant="secondary", scale=0)

    # On a phone these two columns stack. The CSS in kelvin/ui/theme.py then
    # moves the photo to the top and pins it there while you scroll the sliders.
    with gr.Row(equal_height=False, elem_id="k-main"):
        with gr.Column(scale=1, min_width=320, elem_id="k-sliders"):
            sliders = [
                gr.Slider(-1, 1, value=0, step=0.01, label=s.label, info=s.help, elem_id=f"k-{s.name}")
                for s in SLIDERS
            ]
        with gr.Column(scale=2, min_width=480, elem_id="k-view"):
            photo = gr.ImageSlider(
                label="Before / after (drag the line to compare)", elem_id="k-photo", interactive=False
            )
            histogram = gr.Plot(label="Histogram", elem_id="k-histogram")
            reasons = gr.HTML(reasons_html(), elem_id="k-reasons")

    upload.upload(load_preview, inputs=upload, outputs=[preview_state, header]).then(
        render, inputs=[preview_state, *sliders], outputs=[photo, histogram]
    ).then(reasons_html, outputs=reasons)
    # .input (not .change) so this only fires when a person drags a slider.
    # Auto and Reset set all 10 sliders at once and redraw a single time
    # themselves; with .change, they'd trigger 10 redraws.
    for slider in sliders:
        slider.input(render, inputs=[preview_state, *sliders], outputs=[photo, histogram])
    auto_button.click(auto, inputs=preview_state, outputs=[*sliders, photo, histogram, reasons])
    reset_button.click(reset, inputs=preview_state, outputs=[*sliders, photo, histogram, reasons])


if __name__ == "__main__":
    # Gradio 6 takes the look (theme, extra CSS, fonts) at launch, not in gr.Blocks.
    demo.launch(theme=build_theme(), css=CSS, head=HEAD)
