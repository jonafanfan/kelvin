"""Kelvin's design system, "Darkroom", as named colours.

These come from the team's design canvas. Every colour the app uses is named
here, so changing the look means changing one file.

The two rules of the system:
  1. Neutral greys only, so the interface never tints how you judge colour.
  2. Two accents: warm (Tungsten, orange) = a change YOU made,
                  cool (Daylight, blue)   = a change the AI made.
"""

# Neutrals, darkest to lightest
INK = "#0E0E0F"  # page background
SURFACE = "#18181A"  # panels and cards
RAISED = "#222225"  # things sitting on a panel (photo frame)
LINE = "#2E2E33"  # borders, empty slider track
DIVIDER = "#26262A"  # header underline
TICK = "#4A4A52"  # centre mark on sliders
BUTTON_BORDER = "#3A3A40"
MUTED = "#9A9AA0"  # small labels, captions
TEXT_SOFT = "#B8B8BE"  # secondary text
TEXT = "#EDEDED"  # main text

# Accents
TUNGSTEN = "#FF7A2E"  # 3200K: you
TUNGSTEN_TEXT = "#FF9A57"  # the same, lighter, for text on dark backgrounds
DAYLIGHT = "#5B8CFF"  # 6500K: the AI
DAYLIGHT_TEXT = "#8DB0FF"

# The kelvin scale, from cool to warm. Only ever used as the Warmth slider's track.
KELVIN_SCALE = "linear-gradient(90deg, #4D7CFF 0%, #8DB4FF 16%, #CFE0FF 32%, #F3EEE6 48%, #FFC489 64%, #FF8A3D 82%, #FF5A12 100%)"

# Histogram channels
HIST_RED = "#E8636B"
HIST_GREEN = "#4FBF7F"
HIST_BLUE = "#6C8CFF"

# Fonts
FONT_DISPLAY = "Bricolage Grotesque"  # wordmark and headings
FONT_UI = "Geist"  # everything else
FONT_MONO = "Geist Mono"  # numbers: slider values, kelvins
