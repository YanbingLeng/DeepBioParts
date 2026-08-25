"""Unified Nature journal figure style.

Consolidates 4 duplicate implementations from:
- train_predictor_regression.py (archived)
- diffusion_model/evaluate_generative_model.py
- diffusion_model/evaluate_direct_diffusion.py

All figures across the project should use this single module for consistent
Nature/Science publication-quality styling.
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import tempfile
from xml.etree import ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm

# ---------------------------------------------------------------------------
# Nature color palette (colorblind-safe)
# ---------------------------------------------------------------------------
NATURE_SKY_BLUE = "#56B4E9"
NATURE_VERMILLION = "#D55E00"
NATURE_BLUISH_GREEN = "#009E73"
NATURE_YELLOW = "#F0E442"
NATURE_BLUE = "#2171B5"
NATURE_RED = "#D62728"
NATURE_GREEN = "#31A354"
NATURE_LIGHT_BLUE = "#6BAED6"
NATURE_DARK_GRAY = "#4D4D4D"
NATURE_LIGHT_GRAY = "#BDBDBD"

# Semantic palette: natural vs generated library comparison
# Convention (Nature skill standard): blue = hero/proposed method, warm = baseline
# For DeepBioParts: generated library = hero, natural library = baseline
GENERATED_BLUE = NATURE_BLUE       # "#2171B5" — generated / designed library (hero)
NATURAL_ORANGE = NATURE_VERMILLION  # "#D55E00" — natural / baseline library
# Backward-compatible aliases (deprecated — prefer GENERATED_BLUE / NATURAL_ORANGE)
NATURAL_TEAL = NATURE_DARK_GRAY       # [deprecated] Natural / baseline library
GENERATED_CORAL = GENERATED_BLUE      # [deprecated] Generated / designed library
OOD_ORANGE = NATURE_VERMILLION
RANDOM_GRAY = NATURE_LIGHT_GRAY

# Nature physical figure sizes
NATURE_SINGLE_COL_MM = 89
NATURE_DOUBLE_COL_MM = 183
NATURE_EXTENDED_MAX_MM = 180
NATURE_TEXT_SIZE_PT = 8.0
NATURE_PANEL_LABEL_SIZE_PT = 12.0
_SUPPLEMENTARY_PANEL_LABEL_GID_PREFIX = "supplementary-panel-label-"

# Scatter plot defaults
SCATTER_SIZE = 8
SCATTER_ALPHA = 0.6
SCATTER_EDGE = "none"

# Line defaults
DASH_LINE_WIDTH = 0.7
SOLID_LINE_WIDTH = 0.7


# =============================================================================
# 3D plotting style configuration
# =============================================================================

@dataclass
class Nature3DStyleConfig:
    """
    Nature-style 3D plot configuration.

    All 3D plots should use this configuration class to ensure a uniform style.

    Attributes:
        figure_size: figure size (width, height)
        dpi: figure resolution
        elev: 3D elevation angle (angle between the viewing direction and the xy plane, 0-90 degrees)
        azim: 3D azimuth angle (rotation of the viewing direction around the z axis, 0-360 degrees)
        labelpad: distance between axis labels and the axis (smaller values are closer)
        tick_labelsize: tick label font size
        axis_labelsize: axis title font size
        title_size: main title font size
        title_pad: spacing between the main title and the figure
        grid_on: whether to show grid lines
        pane_fill: whether to fill the axis panes
        pane_alpha: axis pane transparency (0.0 fully transparent, 1.0 fully opaque)
        aspect_ratio: 3D view aspect ratio
        surface_alpha: surface transparency (1.0 opaque, 0.0 fully transparent)
        surface_rstride: surface row stride (smaller = finer but slower to render)
        surface_cstride: surface column stride (smaller = finer but slower to render)
        surface_linewidth: surface mesh line width (0 = no mesh lines)
        scatter_size: scatter point size
        scatter_alpha: scatter point transparency
        scatter_edge: scatter point edge style
        contour_interval: contour plane spacing (default 0.2)
        contour_levels: number of contour lines
        contour_alpha: contour transparency
        contour_linewidth: contour line width
        colorbar_shrink: colorbar shrink ratio (0-1)
        colorbar_pad: spacing between colorbar and figure
        colorbar_labelsize: colorbar label font size
        colorbar_aspect: colorbar aspect ratio
        tick_pad: distance between tick labels and ticks
        tick_length: tick mark length
        tick_width: tick mark width
        tick_format: tick label format (e.g. "%.1f" for one decimal place)
        axis_line_width: axis line width
        axis_line_color: axis line color
    """
    # Base figure settings
    figure_size: tuple = (5.5, 5)      # Figure size (enlarged to prevent content overflow)
    dpi: int = 600                     # Resolution

    # 3D view settings
    elev: int = 15                     # Elevation angle (0-90 degrees) - lowered for a flatter view
    azim: int = 45                     # Azimuth angle (rotation around z, 0-360 degrees)

    # Axis label settings
    labelpad: int = -7                 # Distance from axis label to axis (negative = closer; -1 to -3 recommended)
    tick_labelsize: int = 8            # Tick label font size
    axis_labelsize: int = 8            # Axis title font size
    title_size: int = 8                # Main title font size
    title_pad: int = -25               # Spacing between title and figure (negative brings title closer)

    # Grid and pane settings
    grid_on: bool = False              # Whether to show grid lines
    pane_fill: bool = False            # Whether to fill the pane background
    pane_alpha: float = 0.0            # Pane transparency (0.0 transparent, 1.0 opaque)
    aspect_ratio: tuple = (1, 1, 1.5) # 3D aspect ratio (further reduces the height ratio)

    # Surface settings
    surface_alpha: float = 1.0        # Surface transparency (1.0 opaque, 0.0 transparent)
    surface_rstride: int = 0.1           # Surface row stride (smaller = finer but slower)
    surface_cstride: int = 0.1           # Surface column stride (smaller = finer but slower)
    surface_linewidth: float = 0.0     # Surface mesh line width (0 = hidden)

    # Scatter settings
    scatter_size: int = 5              # Scatter point size
    scatter_alpha: float = 0.65        # Scatter point transparency
    scatter_edge: str = "none"         # Scatter point edge style

    # Contour settings
    contour_interval: float = 0.2      # Contour plane spacing
    contour_levels: int = 8            # Number of contour lines
    contour_alpha: float = 0.2        # Contour transparency
    contour_linewidth: float = 0.0     # Contour line width

    # Colorbar settings
    colorbar_shrink: float = 0.7      # Colorbar shrink ratio (0-1)
    colorbar_pad: float = 0.01          # Spacing between colorbar and figure
    colorbar_labelsize: int = 8        # Colorbar label font size
    colorbar_aspect: float = 15        # Colorbar aspect ratio

    # Tick settings
    tick_pad: int = -3                 # Distance between tick labels and ticks
    tick_length: int = 3               # Tick mark length
    tick_width: float = 0.5            # Tick mark width
    tick_format: str = "%.1f"          # Tick label format (one decimal place)

    # Axis line settings
    axis_line_width: float = 0.5       # Axis line width
    axis_line_color: str = '#333333'   # Axis line color


# Default 3D style configuration
DEFAULT_3D_STYLE = Nature3DStyleConfig()


# =============================================================================
# 3D plotting style application functions
# =============================================================================

def apply_nature_3d_style(
    ax: Axes3D,
    config: Nature3DStyleConfig = None,
) -> Axes3D:
    """
    Apply Nature style to a 3D Axes.

    Args:
        ax: 3D Axes object
        config: 3D style configuration (None uses the default config)

    Returns:
        styled_ax: the Axes with style applied
    """
    if config is None:
        config = DEFAULT_3D_STYLE

    # Grid
    ax.grid(config.grid_on)

    # Axis panes
    ax.xaxis.pane.fill = config.pane_fill
    ax.yaxis.pane.fill = config.pane_fill
    ax.zaxis.pane.fill = config.pane_fill

    # Pane transparency
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, config.pane_alpha))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, config.pane_alpha))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, config.pane_alpha))

    # Axis lines
    ax.xaxis.line.set_color(config.axis_line_color)
    ax.yaxis.line.set_color(config.axis_line_color)
    ax.zaxis.line.set_color(config.axis_line_color)
    ax.xaxis.line.set_linewidth(config.axis_line_width)
    ax.yaxis.line.set_linewidth(config.axis_line_width)
    ax.zaxis.line.set_linewidth(config.axis_line_width)

    # Tick parameters
    # Note: in 3D Axes the pad argument has limited effect; matplotlib
    # automatically adjusts tick label positions to avoid overlap with the figure.
    ax.tick_params(
        axis='x',
        labelsize=config.tick_labelsize,
        pad=config.tick_pad,
        length=config.tick_length,
        width=config.tick_width,
    )
    ax.tick_params(
        axis='y',
        labelsize=config.tick_labelsize,
        pad=config.tick_pad,
        length=config.tick_length,
        width=config.tick_width,
    )
    ax.tick_params(
        axis='z',
        labelsize=config.tick_labelsize,
        pad=config.tick_pad,
        length=config.tick_length,
        width=config.tick_width,
    )

    # Additional adjustment of 3D tick label distance by modifying the
    # internal _axinfo attribute that controls tick label positioning.
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        if hasattr(axis, '_axinfo'):
            # Inward / outward factors for the tick marks
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
            # Space factor for the tick labels (smaller value brings labels
            # closer to the axis; tick_pad=-3 corresponds to a small factor).
            space_factor = max(0.3, 1.0 + config.tick_pad * 0.2)
            axis._axinfo['label']['space_factor'] = space_factor

    # Tick label format (one decimal place)
    from matplotlib.ticker import FormatStrFormatter
    z_formatter = FormatStrFormatter(config.tick_format)
    ax.zaxis.set_major_formatter(z_formatter)

    # Ensure all tick labels use Arial
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        for label in axis.get_ticklabels():
            label.set_fontfamily('Arial')

    # View angle
    ax.view_init(elev=config.elev, azim=config.azim)

    # Aspect ratio
    ax.set_box_aspect(config.aspect_ratio)

    return ax


def setup_3d_axis_labels(
    ax: Axes3D,
    xlabel: str = "X",
    ylabel: str = "Y",
    zlabel: str = "Z",
    title: str = "",
    config: Nature3DStyleConfig = None,
) -> Axes3D:
    """
    Set 3D axis labels and title.

    Args:
        ax: 3D Axes object
        xlabel: x axis label
        ylabel: y axis label
        zlabel: z axis label
        title: figure title
        config: 3D style configuration (None uses the default config)

    Returns:
        ax: the Axes with labels applied
    """
    if config is None:
        config = DEFAULT_3D_STYLE

    # Set labels, explicitly using Arial
    ax.set_xlabel(xlabel, fontsize=config.axis_labelsize, labelpad=config.labelpad,
                  fontfamily='Arial')
    ax.set_ylabel(ylabel, fontsize=config.axis_labelsize, labelpad=config.labelpad,
                  fontfamily='Arial')
    ax.set_zlabel(zlabel, fontsize=config.axis_labelsize, labelpad=config.labelpad,
                  fontfamily='Arial')

    if title:
        ax.set_title(title, fontsize=config.title_size, pad=config.title_pad,
                     fontfamily='Arial')

    return ax


def get_3d_colormap(name: str = "fitness") -> mpl.colors.Colormap:
    """
    Get a predefined 3D plotting colormap.

    Args:
        name: colormap name ("fitness", "cool_warm", "viridis")

    Returns:
        colormap: matplotlib Colormap object
    """
    if name == "fitness":
        # Fitness landscape colormap (cool to warm)
        colors = [
            (0.0,  "#0a1628"),    # Dark blue - low fitness
            (0.15, "#1a4a5a"),
            (0.3,  "#2A9D8F"),
            (0.45, "#4ecdc4"),
            (0.55, "#a8dadc"),
            (0.65, "#F0E442"),
            (0.75, "#f4a261"),
            (0.85, "#E76F51"),
            (1.0,  "#8B0000"),    # Dark red - high fitness
        ]
        return mpl.colors.LinearSegmentedColormap.from_list("nature_fitness", colors, N=256)

    elif name == "cool_warm":
        # Cool-to-warm colormap
        colors = [
            (0.0, NATURE_BLUE),
            (0.3, '#4CA6D6'),
            (0.5, NATURE_SKY_BLUE),
            (0.7, '#F4C58F'),
            (1.0, NATURE_VERMILLION),
        ]
        return mpl.colors.LinearSegmentedColormap.from_list("cool_warm", colors, N=256)

    else:
        # Default to viridis
        return plt.cm.viridis


def plot_3d_contour_planes(
    ax: Axes3D,
    x_range: tuple,
    y_range: tuple,
    z_levels: np.ndarray,
    colormap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    alpha: float = 0.12,
    grid_res: int = 40,
) -> None:
    """
    Draw horizontal contour planes in a 3D plot.

    Args:
        ax: 3D Axes object
        x_range: x axis range (min, max)
        y_range: y axis range (min, max)
        z_levels: array of z heights
        colormap: colormap
        norm: Normalize object
        alpha: plane transparency (0.0-1.0)
        grid_res: grid resolution
    """
    x_pad = (x_range[1] - x_range[0]) * 0.5  # Pad by 50%
    y_pad = (y_range[1] - y_range[0]) * 0.5

    xi = np.linspace(x_range[0] - x_pad, x_range[1] + x_pad, grid_res)
    yi = np.linspace(y_range[0] - y_pad, y_range[1] + y_pad, grid_res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    for i, z_level in enumerate(z_levels):
        if i == 0 or i == len(z_levels) - 1:
            continue
        color = colormap(norm(z_level))
        ax.plot_surface(
            xi_grid, yi_grid, np.full_like(xi_grid, z_level),
            color=color, alpha=alpha, linewidth=0, shade=False,
        )


def setup_3d_colorbar(
    fig: plt.Figure,
    ax: Axes3D,
    colormap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    label: str = "Fitness",
    config: Nature3DStyleConfig = None,
    z_ticks: np.ndarray = None,
) -> mpl.colorbar.Colorbar:
    """
    Set up the colorbar for a 3D plot.

    Args:
        fig: Figure object
        ax: Axes object
        colormap: colormap
        norm: Normalize object
        label: colorbar label
        config: 3D style configuration
        z_ticks: colorbar tick values (optional)

    Returns:
        colorbar: the Colorbar object
    """
    if config is None:
        config = DEFAULT_3D_STYLE

    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=config.colorbar_pad,
        shrink=config.colorbar_shrink,
        aspect=config.colorbar_aspect,
    )
    # Set colorbar label font to Arial
    cbar.set_label(label, fontsize=config.colorbar_labelsize, fontfamily='Arial')
    cbar.ax.tick_params(labelsize=config.colorbar_labelsize)
    # Set colorbar tick label font
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Arial')

    if z_ticks is not None:
        cbar.set_ticks(z_ticks)

    return cbar


# =============================================================================
# NPG color cycle (Nature Publishing Group standard palette)
# =============================================================================
NPG_COLOR_CYCLE = [
    "#E64B35",  # Vermilion
    "#4DBBD5",  # Cyan blue
    "#00A087",  # Bluish green
    "#3C5488",  # Dark blue
    "#F39B7F",  # Light coral
    "#8491B4",  # Grayish blue
    "#91D1C2",  # Mint green
]


# =============================================================================
# 2D global style - Nature aesthetic + fully Adobe Illustrator editable
# =============================================================================

def set_nature_ai_style() -> None:
    """Configure matplotlib global rcParams for Nature-quality, AI-editable 2D figures.

    After calling this function, subsequent plots can be exported as SVG/PNG
    output satisfies:
      - PDF text embedded as TrueType (Type 42); in AI, font, size and content
        are editable on double-click
      - SVG text retained as <text> nodes rather than <path>
      - No unnecessary clip-paths, easing Ungroup in AI
      - Nature single-column size 89 mm x 55 mm, DPI 300
      - The NPG 7-color cycle is injected into axes.prop_cycle
    """

    # -- 1. Adobe Illustrator compatibility --------------------------------
    # pdf.fonttype=42: embed fonts as TrueType outlines in the PDF (not Type 3 bitmaps),
    # so text remains editable when the PDF is opened in AI.
    mpl.rcParams["pdf.fonttype"] = 42
    # ps.fonttype=42: PostScript export also uses TrueType.
    mpl.rcParams["ps.fonttype"] = 42
    # svg.fonttype='none': keep text as <text> tags in SVG instead of <path>.
    mpl.rcParams["svg.fonttype"] = "none"

    # Disable default clipping to reduce the number of clip-paths that must be
    # released manually in AI. savefig.transparent=False avoids extra alpha layers.
    mpl.rcParams["savefig.transparent"] = False

    # -- 2. Typography and font hierarchy ---------------------------------
    # Single global font: Arial Regular (Nature requirement).
    # Listed first in sans-serif so the system matches Arial preferentially.
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["font.weight"] = "regular"

    # Math text (subscripts, superscripts, Greek letters) also uses Arial.
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Arial"
    mpl.rcParams["mathtext.it"] = "Arial:italic"
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["font.cursive"] = ["Arial"]

    # Use one nominal 8 pt size for all figure text.
    mpl.rcParams["font.size"] = NATURE_TEXT_SIZE_PT
    mpl.rcParams["axes.labelsize"] = NATURE_TEXT_SIZE_PT
    mpl.rcParams["axes.titlesize"] = NATURE_TEXT_SIZE_PT
    mpl.rcParams["xtick.labelsize"] = NATURE_TEXT_SIZE_PT
    mpl.rcParams["ytick.labelsize"] = NATURE_TEXT_SIZE_PT
    mpl.rcParams["legend.fontsize"] = NATURE_TEXT_SIZE_PT

    # Note: panel labels (A/B) must be added manually in plotting code via
    # ax.text() with fontsize=8, fontweight='bold'.
    # Example: ax.text(-0.15, 1.08, 'A', transform=ax.transAxes,
    #                  fontsize=8, fontweight='bold', fontfamily='Arial')

    # -- 3. Physical size and resolution ----------------------------------
    # Nature single-column size 89 mm x 55 mm.
    # This is a default; an explicit fig, ax = plt.subplots(figsize=...) overrides it.
    mpl.rcParams["figure.figsize"] = (89 / 25.4, 55 / 25.4)    # ~ (3.50, 2.17) inch
    # Both screen preview and export use 300 DPI.
    mpl.rcParams["figure.dpi"] = 450
    mpl.rcParams["savefig.dpi"] = 450

    # -- 4. Axes, ticks and lines (data-ink ratio) ------------------------
    # Axis linewidth: 0.8 pt (Nature print minimum).
    mpl.rcParams["axes.linewidth"] = 0.8
    # Remove top and right spines (Nature minimalist style).
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

    # Tick width: 0.5 pt; direction: out; length: 3 pt.
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["ytick.major.width"] = 0.5
    mpl.rcParams["xtick.direction"] = "out"
    mpl.rcParams["ytick.direction"] = "out"
    mpl.rcParams["xtick.major.size"] = 3
    mpl.rcParams["ytick.major.size"] = 3
    # Minor ticks slightly thinner.
    mpl.rcParams["xtick.minor.width"] = 0.3
    mpl.rcParams["ytick.minor.width"] = 0.3
    mpl.rcParams["xtick.minor.size"] = 1.5
    mpl.rcParams["ytick.minor.size"] = 1.5

    # Data line width: 1.0 pt.
    mpl.rcParams["lines.linewidth"] = 1.0
    # Round caps and joins so intersections stay smooth when zoomed in AI.
    mpl.rcParams["lines.solid_capstyle"] = "round"
    mpl.rcParams["lines.solid_joinstyle"] = "round"
    mpl.rcParams["lines.dash_capstyle"] = "round"
    mpl.rcParams["lines.dash_joinstyle"] = "round"

    # Default scatter marker size.
    mpl.rcParams["lines.markersize"] = 3

    # -- 5. NPG color cycle injection -------------------------------------
    # Set the NPG 7-color cycle as the default prop_cycle so plt.plot
    # rotates through colors automatically.
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=NPG_COLOR_CYCLE)

    # -- 6. Legend and export helper settings -----------------------------
    mpl.rcParams["legend.framealpha"] = 0.8
    mpl.rcParams["legend.edgecolor"] = "#cccccc"
    mpl.rcParams["legend.borderpad"] = 0.3
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.05


# =============================================================================
# 2D style setup function (legacy, kept for backward compatibility)
# =============================================================================

def setup_nature_style() -> None:
    """[deprecated] Configure matplotlib for Nature/Science publication-quality figures.

    Sets global rcParams including font, size, linewidth, DPI, etc.
    Call once at the start of any plotting script.

    Note: this function is deprecated. New code should use set_nature_ai_style(),
    which provides fuller AI-compatibility options (svg.fonttype='none') and the
    NPG color cycle.
    """
    import warnings
    warnings.warn(
        "setup_nature_style() is deprecated. Use set_nature_ai_style() instead.",
        DeprecationWarning, stacklevel=2,
    )
    mpl.rcParams.update({
        # Font
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": NATURE_TEXT_SIZE_PT,
        "font.weight": "regular",
        # Axes
        "axes.linewidth": 0.8,
        "axes.labelsize": NATURE_TEXT_SIZE_PT,
        "axes.titlesize": NATURE_TEXT_SIZE_PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "xtick.major.size": 3,
        "xtick.minor.size": 1.5,
        "xtick.labelsize": NATURE_TEXT_SIZE_PT,
        "ytick.major.width": 0.5,
        "ytick.minor.width": 0.3,
        "ytick.major.size": 3,
        "ytick.minor.size": 1.5,
        "ytick.labelsize": NATURE_TEXT_SIZE_PT,
        # Lines
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        # Legend
        "legend.fontsize": NATURE_TEXT_SIZE_PT,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "#cccccc",
        "legend.borderpad": 0.3,
        # Figure
        "figure.figsize": (89 / 25.4, 55 / 25.4),  # Nature single column 89 mm x 55 mm
        "figure.dpi": 450,
        "savefig.dpi": 450,
        "savefig.format": "svg",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Other
        "pdf.fonttype": 42,  # TrueType fonts in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # editable text in SVG
    })


def nature_ax(
    ax: Axes,
    spine_positions: Optional[List[str]] = None,
) -> Axes:
    """Apply Nature style to a single Axes.

    Args:
        ax: matplotlib Axes to style
        spine_positions: spines to show. Default: ['left', 'bottom']

    Returns:
        The styled Axes (same reference)
    """
    if spine_positions is None:
        spine_positions = ["left", "bottom"]

    for spine in ax.spines.values():
        spine.set_visible(False)
    for pos in spine_positions:
        ax.spines[pos].set_visible(True)
        ax.spines[pos].set_linewidth(0.5)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=3,
        width=0.5,
        labelsize=NATURE_TEXT_SIZE_PT,
    )
    ax.grid(False)

    return ax


def enforce_arial(
    fig: plt.Figure,
    font_size: Optional[float] = None,
) -> None:
    """Force figure text to Arial and optionally one nominal size.

    rcParams set the default font, but helpers such as logomaker, colorbars, or
    manually added labels can reset text objects after axes are created.
    Supplementary panel labels tagged by :func:`add_panel_label` retain the
    requested 12-pt bold style.
    """
    for text in fig.findobj(match=mpl.text.Text):
        text.set_fontfamily("Arial")
        gid = str(text.get_gid() or "")
        if gid.startswith(_SUPPLEMENTARY_PANEL_LABEL_GID_PREFIX):
            text.set_fontsize(NATURE_PANEL_LABEL_SIZE_PT)
            text.set_fontweight("bold")
        elif font_size is not None:
            text.set_fontsize(font_size)


_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_SVG_FONT_RE = re.compile(
    r"^(?P<prefix>.*?)"
    r"(?P<size>[0-9]+(?:\.[0-9]+)?)"
    r"(?P<unit>px|pt)"
    r"(?:/[^\s]+)?\s+"
    r"(?P<family>.+)$"
)
_SVG_FONT_PROPERTIES = {
    "font",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
}


def _parse_svg_style(style: str) -> Dict[str, str]:
    declarations: Dict[str, str] = {}
    for declaration in style.split(";"):
        if ":" not in declaration:
            continue
        name, value = declaration.split(":", 1)
        declarations[name.strip()] = value.strip()
    return declarations


def _format_svg_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _explicit_svg_font(element: ET.Element) -> Optional[Dict[str, Any]]:
    style = _parse_svg_style(element.get("style", ""))
    size_value = element.get("font-size") or style.get("font-size")
    font_style = element.get("font-style") or style.get("font-style")
    font_weight = element.get("font-weight") or style.get("font-weight")

    shorthand = style.get("font")
    if shorthand:
        match = _SVG_FONT_RE.match(shorthand)
        if match:
            size_value = match.group("size")
            prefix_tokens = match.group("prefix").strip().split()
            for token in prefix_tokens:
                if token in {"italic", "oblique"}:
                    font_style = token
                elif token == "bold" or token.isdigit():
                    font_weight = token

    if size_value is None:
        return None
    match = re.match(r"([0-9]+(?:\.[0-9]+)?)", str(size_value))
    if not match:
        return None
    return {
        "size": float(match.group(1)),
        "style": font_style,
        "weight": font_weight,
    }


def _set_office_svg_font(
    element: ET.Element,
    font: Dict[str, Any],
    size: float,
    family: str,
) -> None:
    style = _parse_svg_style(element.get("style", ""))
    for property_name in _SVG_FONT_PROPERTIES:
        style.pop(property_name, None)
    if style:
        element.set(
            "style",
            "; ".join(f"{name}: {value}" for name, value in style.items()),
        )
    else:
        element.attrib.pop("style", None)

    element.set("font-family", family)
    element.set("font-size", _format_svg_number(size))
    if font.get("style") and font["style"] != "normal":
        element.set("font-style", str(font["style"]))
    else:
        element.attrib.pop("font-style", None)
    if font.get("weight") and font["weight"] != "normal":
        element.set("font-weight", str(font["weight"]))
    else:
        element.attrib.pop("font-weight", None)


def _normalize_office_svg_text(element: ET.Element) -> None:
    """Remove SVG text constructs that Word imports inconsistently."""
    if element.text:
        element.text = element.text.replace("\u00a0", " ")
    if element.tail:
        element.tail = element.tail.replace("\u00a0", " ")


def _flatten_single_line_svg_text(text_element: ET.Element) -> bool:
    """Merge positioned same-baseline tspans into one Word-stable text run."""
    tspan_tag = f"{{{_SVG_NAMESPACE}}}tspan"
    tspans = list(text_element)
    if not tspans or any(element.tag != tspan_tag for element in tspans):
        return False

    y_positions = {element.get("y") for element in tspans}
    if None in y_positions or len(y_positions) != 1:
        return False

    positioned_characters = []
    for element in tspans:
        content = element.text or ""
        x_positions = element.get("x", "").split()
        if not content or len(x_positions) != len(content):
            return False
        try:
            positioned_characters.extend(
                (float(x_position), character)
                for x_position, character in zip(x_positions, content)
            )
        except ValueError:
            return False

    positioned_characters.sort(key=lambda item: item[0])
    for element in tspans:
        text_element.remove(element)
    text_element.text = "".join(
        character for _, character in positioned_characters
    )
    text_element.set("x", _format_svg_number(positioned_characters[0][0]))
    text_element.set("y", next(iter(y_positions)))
    return True


def postprocess_svg_for_office(
    path: Union[str, Path],
    font_size: float = NATURE_TEXT_SIZE_PT,
    font_family: str = "Arial",
) -> Path:
    """Make editable Matplotlib SVG text stable in Word and LibreOffice.

    Matplotlib writes text with the CSS ``font`` shorthand, which Office SVG
    importers may ignore and replace with an oversized default font. This
    function replaces the shorthand with explicit SVG presentation attributes.
    Normal text becomes exactly ``font_size``; tagged supplementary panel
    labels become 12-pt bold. Per-character tspan positions and non-breaking
    spaces are normalized because Word can otherwise compress mathematical
    labels such as ``r = 0.97`` into overlapping glyphs.
    """
    svg_path = Path(path)
    if svg_path.suffix.lower() != ".svg":
        raise ValueError(f"Expected an SVG file: {svg_path}")
    if font_size <= 0:
        raise ValueError("font_size must be positive")

    for prefix, namespace in (
        ("", _SVG_NAMESPACE),
        ("xlink", "http://www.w3.org/1999/xlink"),
        ("dc", "http://purl.org/dc/elements/1.1/"),
        ("cc", "http://creativecommons.org/ns#"),
        ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ):
        ET.register_namespace(prefix, namespace)

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(svg_path, parser=parser)
    root = tree.getroot()
    text_tag = f"{{{_SVG_NAMESPACE}}}text"
    parent_by_child = {
        child: parent
        for parent in root.iter()
        for child in parent
    }

    for text_element in root.iter(text_tag):
        current = text_element
        is_supplementary_panel_label = False
        while current is not None:
            element_id = str(current.get("id") or "")
            if element_id.startswith(_SUPPLEMENTARY_PANEL_LABEL_GID_PREFIX):
                is_supplementary_panel_label = True
                break
            current = parent_by_child.get(current)
        target_font_size = (
            NATURE_PANEL_LABEL_SIZE_PT
            if is_supplementary_panel_label
            else font_size
        )

        for element in text_element.iter():
            _normalize_office_svg_text(element)
        _flatten_single_line_svg_text(text_element)

        explicit_fonts = []
        for element in text_element.iter():
            font = _explicit_svg_font(element)
            if font is not None:
                explicit_fonts.append((element, font))
        if not explicit_fonts:
            fallback_font = {"style": None, "weight": None}
            if is_supplementary_panel_label:
                fallback_font["weight"] = "bold"
            _set_office_svg_font(
                text_element,
                fallback_font,
                target_font_size,
                font_family,
            )
            continue

        for element, font in explicit_fonts:
            if is_supplementary_panel_label:
                font = dict(font)
                font["weight"] = "bold"
            _set_office_svg_font(
                element,
                font,
                target_font_size,
                font_family,
            )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=svg_path.parent,
            prefix=f".{svg_path.stem}.",
            suffix=".svg",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        tree.write(
            temp_path,
            encoding="utf-8",
            xml_declaration=True,
        )
        os.replace(temp_path, svg_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return svg_path


def save_nature_svg(
    fig: plt.Figure,
    path: str,
    dpi: int = 450,
) -> None:
    """Save as publication-quality SVG + PNG.

    Args:
        fig: matplotlib Figure to save
        path: output file path (should end with .svg)
        dpi: resolution for embedded raster elements
    """
    base = Path(path).with_suffix("")
    enforce_arial(fig, font_size=NATURE_TEXT_SIZE_PT)

    # Ensure the output directory exists
    base.parent.mkdir(parents=True, exist_ok=True)

    # Save as editable SVG
    fig.savefig(
        str(base) + ".svg",
        format="svg",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=False,
    )
    postprocess_svg_for_office(
        str(base) + ".svg",
        font_size=NATURE_TEXT_SIZE_PT,
    )

    # Also export a PNG (bitmap preview)
    fig.savefig(
        str(base) + ".png",
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=False,
    )

    plt.close(fig)


def is_dark(hex_color: str, threshold: int = 128) -> bool:
    """Return whether a color is dark (for adaptive text color selection).

    Args:
        hex_color: hex color code (e.g. "#272727")
        threshold: brightness threshold (0-255); values below this are dark

    Returns:
        True if the color is dark (use white text)
    """
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold


def add_panel_label(
    ax: Axes,
    label: str,
    x: float = -0.06,
    y: float = 1.02,
    fontsize: int = 8,
    color: str = "black",
    fontweight: str = "bold",
    supplementary: bool = False,
    ha: str = "left",
    va: str = "bottom",
) -> mpl.text.Text:
    """Add a Nature-style panel label (A, B, C...) to the upper-left of an Axes.

    Args:
        ax: matplotlib Axes object
        label: panel label (e.g. "A", "B")
        x: x position (in Axes coords; negative is left of the axes)
        y: y position (in Axes coords; >1 is above the axes)
        fontsize: label font size (Nature standard is 8 pt bold)
        color: label color (use 'white' on dark panels)
        fontweight: font weight
        supplementary: use the requested supplementary-figure style:
            12-pt bold Arial, preserved by Office SVG post-processing
        ha: horizontal alignment
        va: vertical alignment
    """
    if supplementary:
        fontsize = NATURE_PANEL_LABEL_SIZE_PT
        fontweight = "bold"
    text = ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha=ha,
        va=va,
        fontfamily="Arial",
    )
    if supplementary:
        text.set_gid(f"{_SUPPLEMENTARY_PANEL_LABEL_GID_PREFIX}{label}")
    return text


# =============================================================================
# Demo: shows how to use set_nature_ai_style()
# =============================================================================

def _demo_nature_ai_style() -> None:
    """Generate a demo figure with multiple datasets to verify the Nature + AI-friendly style.

    Run with: python -m src.visualization.nature_style
    Output: nature_style_demo.svg and nature_style_demo.png
    """
    # 1. Apply the global style (must be called before plt.subplots)
    set_nature_ai_style()

    # 2. Create the figure - Nature single column 89 mm x 55 mm
    fig, ax = plt.subplots(figsize=(89 / 25.4, 55 / 25.4))

    # 3. Generate demo datasets
    np.random.seed(42)
    x = np.linspace(0, 10, 50)

    # The NPG color cycle rotates automatically per line, but can be set manually.
    conditions = [
        ("Condition A", lambda x: 2.0 + 0.3 * x + np.random.normal(0, 0.3, len(x))),
        ("Condition B", lambda x: 1.5 + 0.5 * x + np.random.normal(0, 0.4, len(x))),
        ("Condition C", lambda x: 3.0 + 0.1 * x + np.random.normal(0, 0.35, len(x))),
        ("Condition D", lambda x: 1.0 + 0.4 * x + np.random.normal(0, 0.25, len(x))),
    ]

    for i, (label, func) in enumerate(conditions):
        y = func(x)
        # Line width 1.0 pt is set globally via rcParams; round caps are configured.
        ax.plot(x, y, linewidth=1.0, label=label, marker='o',
                markersize=2.5, markeredgewidth=0)

    # 4. Axis labels (7 pt, controlled by rcParams)
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("Relative activity")

    # 5. Panel label (A/B, 8 pt bold - must be set manually)
    ax.text(-0.18, 1.08, 'A', transform=ax.transAxes,
            fontsize=8, fontweight='bold', fontfamily='Arial')

    # 6. Legend (6 pt, controlled by rcParams, placed as appropriate)
    ax.legend(loc='upper left', frameon=True)

    # 7. Final tick and spine confirmation (rcParams already handle this; fine-tune here)
    ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.5)

    # 8. Export the standard manuscript formats.
    output_path = "nature_style_demo.svg"
    save_nature_svg(fig, output_path)
    print("[OK] Exported SVG + PNG: nature_style_demo.{svg,png}")
    print(f"     Size: 89 mm x 55 mm, DPI: 300")
    print(f"     Open in Adobe Illustrator to verify text editability")

    plt.close(fig)


if __name__ == "__main__":
    _demo_nature_ai_style()
