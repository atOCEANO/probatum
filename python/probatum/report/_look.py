"""How the reports look: every colour, font and size a figure or a page uses, the one
file a redesign edits (ADR 0041)."""

# matplotlib's own font, so every machine draws the same glyphs
FONT = "DejaVu Sans"
SIZE = 10.0
# a figure is drawn at 72 dots to the inch, so a point is a pixel of the page
DPI    = 72
WIDTH  = 720
HEIGHT = 260
SHORT  = 200

SURFACE   = "#ffffff"
INK       = "#0b0b0b"
SECONDARY = "#52514e"
MUTED     = "#66655f"
GRID      = "#ebeae4"
AXIS      = "#c3c2b7"
# the strategy and buying and holding, the two series every report sets side by side
STRATEGY = "#2a78d6"
HOLD     = "#eb6834"
# the heatmap's diverging ramp: a loss, nothing, a gain
LOSS   = "#e34948"
MIDDLE = "#f0efec"
GAIN   = "#2a78d6"
# the instruments' series in a fixed order, the rest folded into "others"
SERIES = ("#2a78d6", "#eb6834", "#1baa7a", "#9a59d0", "#d4a20f", "#e34948", "#3fb6c9",
          "#8a7f73")

LINE = 2.0
THIN = 1.0
WASH = 0.10

RC = {
    "figure.facecolor":      SURFACE,
    "axes.facecolor":        SURFACE,
    "savefig.facecolor":     SURFACE,
    "font.family":           FONT,
    "font.size":             SIZE,
    "text.color":            SECONDARY,
    "axes.edgecolor":        AXIS,
    "axes.linewidth":        THIN,
    "axes.labelcolor":       MUTED,
    "axes.titlesize":        SIZE,
    "axes.titlecolor":       INK,
    "axes.titlelocation":    "left",
    "axes.titlepad":         10.0,
    "xtick.color":           MUTED,
    "ytick.color":           MUTED,
    "xtick.major.size":      0.0,
    "ytick.major.size":      0.0,
    "xtick.minor.size":      0.0,
    "xtick.major.pad":       6.0,
    "ytick.major.pad":       6.0,
    "axes.grid":             True,
    "axes.grid.axis":        "y",
    "grid.color":            GRID,
    "grid.linewidth":        THIN,
    "axes.axisbelow":        True,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.spines.left":      False,
    "legend.frameon":        False,
    "legend.fontsize":       SIZE,
    "lines.solid_capstyle":  "round",
    "lines.solid_joinstyle": "round",
    # text drawn as paths, so the pinned font is the one every viewer sees, and ids
    # salted by a constant, so a figure's SVG is the same on every run
    "svg.fonttype":          "path",
    "svg.hashsalt":          "probatum",
}

PAGE = """
:root { color-scheme: light; }
body { margin: 0; background: #ffffff; color: #0b0b0b;
       font: 15px/1.5 "DejaVu Sans", Verdana, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 40px 0 8px; }
p { margin: 4px 0; color: #52514e; }
.warn { color: #0b0b0b; background: #fbeee6; border-left: 3px solid #eb6834;
        padding: 6px 10px; }
figure { margin: 12px 0; }
figure svg { display: block; width: 100%; height: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 8px 0; }
th, td { text-align: left; padding: 4px 10px 4px 0; border-bottom: 1px solid #ebeae4;
         vertical-align: top; }
th { color: #66655f; font-weight: normal; }
td.number { font-variant-numeric: tabular-nums; white-space: nowrap; }
details { margin: 12px 0; }
summary { cursor: pointer; color: #52514e; }
.scroll { overflow-x: auto; }
"""
