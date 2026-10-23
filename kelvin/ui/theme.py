"""Turns the Darkroom design tokens into a Gradio theme.

Gradio styles things in two ways, so there are two parts here:
  1. THEME: Gradio's built-in settings (background, fonts, button colours).
  2. CSS: the extra details Gradio has no setting for, like sliders that fill
     outward from the middle, or the Warmth slider's orange-to-blue track.
"""

import html

import gradio as gr

from . import tokens as t

# Gradio has a light and a dark mode. Kelvin is always dark, so every setting
# gets the same value in both modes.
_SETTINGS = {
    "body_background_fill": t.INK,
    "body_text_color": t.TEXT,
    "body_text_color_subdued": t.MUTED,
    "background_fill_primary": t.SURFACE,
    "background_fill_secondary": t.INK,
    "block_background_fill": t.SURFACE,
    "block_border_color": t.LINE,
    "block_label_background_fill": t.SURFACE,
    "block_label_text_color": t.MUTED,
    "block_title_text_color": t.TEXT,
    "block_info_text_color": t.MUTED,
    "border_color_primary": t.LINE,
    "input_background_fill": t.RAISED,
    "input_border_color": t.LINE,
    "panel_background_fill": t.SURFACE,
    "slider_color": t.TUNGSTEN,
    "button_secondary_background_fill": "transparent",
    "button_secondary_background_fill_hover": t.RAISED,
    "button_secondary_border_color": t.BUTTON_BORDER,
    "button_secondary_text_color": t.TEXT,
    "button_primary_background_fill": t.DAYLIGHT,
    "button_primary_text_color": "#08112A",
    "color_accent": t.TUNGSTEN,
    "color_accent_soft": t.RAISED,
}


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        font=[gr.themes.GoogleFont(t.FONT_UI), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont(t.FONT_MONO), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_lg,
    )
    both_modes = {}
    for name, value in _SETTINGS.items():
        both_modes[name] = value
        if hasattr(theme, name + "_dark"):  # a few settings have no dark version
            both_modes[name + "_dark"] = value
    return theme.set(**both_modes)


# Loads the display font for the wordmark (the theme only loads the UI fonts).
HEAD = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Bricolage+Grotesque:opsz,wght@12..96,700&display=swap">'
)

# Gradio keeps a percentage like "62%" in --range_progress for each slider.
# Our sliders go from -1 to +1 with 0 in the middle, so the coloured part
# should run from the middle (50%) to the thumb, not from the left edge.
# min()/max() pick whichever side of the middle the thumb is on.
_FROM = "min(50%, var(--range_progress))"
_TO = "max(50%, var(--range_progress))"

CSS = f"""
body, .gradio-container {{ background: {t.INK} !important; }}
.gradio-container {{ max-width: 1440px !important; }}
footer {{ display: none !important; }}

/* Header: logo, wordmark, file name, buttons */
#k-header {{ align-items: center; border-bottom: 1px solid {t.DIVIDER}; padding-bottom: 14px; }}
.k-brand {{ display: flex; align-items: center; gap: 12px; }}
.k-wordmark {{ font-family: '{t.FONT_DISPLAY}', 'Helvetica Neue', sans-serif; font-weight: 700;
               font-size: 26px; letter-spacing: -0.04em; color: {t.TEXT}; }}
.k-filename {{ font-family: '{t.FONT_MONO}', ui-monospace, monospace; font-size: 13px; color: {t.MUTED}; margin-left: 8px; }}
#k-header-buttons {{ justify-content: flex-end; gap: 10px; }}
#k-header-buttons button {{ min-height: 44px; padding: 0 18px; border-radius: 10px; font-weight: 500;
                            border: 1px solid {t.BUTTON_BORDER} !important; background: transparent; }}
#k-header-buttons button:hover {{ background: {t.RAISED}; }}

/* Slider panel */
#k-sliders {{ background: {t.SURFACE}; border: 1px solid {t.LINE}; border-radius: 14px; padding: 8px 22px 16px; }}
#k-sliders .form, #k-sliders .block {{ background: transparent !important; box-shadow: none !important; border: 0 !important; }}
#k-sliders .block {{ border-bottom: 1px solid #232327 !important; border-radius: 0 !important; }}
#k-sliders input[type=number] {{ font-family: '{t.FONT_MONO}', ui-monospace, monospace; color: {t.TUNGSTEN_TEXT};
                                  background: transparent; border: 0; box-shadow: none; text-align: right;
                                  -moz-appearance: textfield; }}
#k-sliders input[type=number]::-webkit-inner-spin-button,
#k-sliders input[type=number]::-webkit-outer-spin-button {{ -webkit-appearance: none; margin: 0; }}

/* Slider track: grey line, a centre tick, and an orange fill from the middle to the thumb */
#k-sliders input[type=range]::-webkit-slider-runnable-track {{
  height: 4px !important;
  background:
    linear-gradient({t.TICK}, {t.TICK}) 50% 50% / 2px 10px no-repeat,
    linear-gradient(to right, {t.LINE} {_FROM}, {t.TUNGSTEN} {_FROM}, {t.TUNGSTEN} {_TO}, {t.LINE} {_TO}) !important;
}}
#k-sliders input[type=range]::-moz-range-track {{
  height: 4px;
  background: linear-gradient(to right, {t.LINE} {_FROM}, {t.TUNGSTEN} {_FROM}, {t.TUNGSTEN} {_TO}, {t.LINE} {_TO});
}}
#k-sliders input[type=range]::-moz-range-progress {{ background: transparent; }}

/* Thumb: light circle with a dark ring, so it shows on any track colour */
#k-sliders input[type=range]::-webkit-slider-thumb {{
  width: 18px; height: 18px; margin-top: -7px; background: {t.TEXT};
  border: 3px solid {t.SURFACE}; box-shadow: 0 0 0 1px {t.TICK};
}}
#k-sliders input[type=range]::-moz-range-thumb {{
  width: 12px; height: 12px; background: {t.TEXT};
  border: 3px solid {t.SURFACE}; box-shadow: 0 0 0 1px {t.TICK};
}}

/* Warmth: the kelvin scale replaces the fill, because the track itself shows the colour */
#k-temp input[type=range]::-webkit-slider-runnable-track {{ height: 6px !important; background: {t.KELVIN_SCALE} !important; }}
#k-temp input[type=range]::-moz-range-track {{ height: 6px; background: {t.KELVIN_SCALE}; }}

/* Photo and histogram cards */
#k-photo, #k-histogram {{ border-radius: 14px !important; border: 1px solid {t.LINE} !important; }}
#k-photo {{ background: {t.RAISED} !important; }}

/* Auto: Daylight blue, because it's the AI's button */
#k-auto {{ background: {t.DAYLIGHT} !important; border-color: {t.DAYLIGHT} !important; color: #08112A !important; }}
#k-auto:hover {{ background: {t.DAYLIGHT_TEXT} !important; }}

/* "Why Auto did that": one row per slider it moved */
.k-reasons {{ background: {t.SURFACE}; border: 1px solid {t.LINE}; border-left: 3px solid {t.DAYLIGHT};
              border-radius: 14px; padding: 14px 18px; }}
.k-reasons h3 {{ margin: 0 0 8px; font-size: 13px; font-weight: 500; color: {t.MUTED}; }}
.k-reasons p {{ margin: 0; color: {t.MUTED}; font-size: 14px; }}
.k-reason {{ display: grid; grid-template-columns: 7.5em 3.5em 1fr; gap: 10px; padding: 6px 0;
             border-top: 1px solid #232327; font-size: 14px; color: {t.TEXT_SOFT}; }}
.k-reason b {{ color: {t.TEXT}; font-weight: 500; }}
.k-reason code {{ font-family: '{t.FONT_MONO}', ui-monospace, monospace; color: {t.DAYLIGHT_TEXT};
                  background: none; text-align: right; }}
@media (max-width: 640px) {{ .k-reason {{ grid-template-columns: 1fr auto; }} .k-reason span {{ grid-column: 1 / -1; }} }}
"""

# The Kelvin mark from the identity board: a "K" made of a bar and a chevron,
# with a tungsten-orange dot where they meet, on a rounded dark tile.
LOGO_SVG = f"""
<svg width="36" height="36" viewBox="0 0 100 100" aria-hidden="true">
  <rect width="100" height="100" rx="23" fill="#202023"></rect>
  <g transform="translate(18 18) scale(0.64)">
    <rect x="18" y="18" width="13" height="64" rx="6.5" fill="{t.TEXT}"></rect>
    <path d="M80 18L52 50L80 82" fill="none" stroke="{t.TEXT}" stroke-width="13" stroke-linecap="round" stroke-linejoin="round"></path>
    <circle cx="52" cy="50" r="10" fill="{t.TUNGSTEN}"></circle>
  </g>
</svg>
"""


def header_html(filename: str | None = None) -> str:
    name = f'<span class="k-filename">{filename}</span>' if filename else ""
    return f'<div class="k-brand">{LOGO_SVG}<span class="k-wordmark">kelvin</span>{name}</div>'


def reasons_html(rows: list[tuple[str, float, str]] | None = None) -> str:
    """The "why" panel. `rows` is (slider label, value, reason) for each slider
    Auto moved; None means Auto hasn't run yet, [] means it found nothing to fix."""
    if rows is None:
        body = "<p>Press <b>Auto</b> and Kelvin will set the sliders, and explain each change here.</p>"
    elif not rows:
        body = "<p>Nothing needed fixing: this photo already looks balanced.</p>"
    else:
        body = "".join(
            f'<div class="k-reason"><b>{label}</b><code>{value:+.2f}</code><span>{html.escape(reason)}</span></div>'
            for label, value, reason in rows
        )
    return f'<div class="k-reasons"><h3>Why Auto did that</h3>{body}</div>'
