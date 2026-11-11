"""Generate the figures used in the documentation.

Run from the repository root:

    uv run --group docs python scripts/generate_plots.py

Figures are written as SVG to ``docs/assets/plots`` with transparent backgrounds and
mid-tone text so they remain readable in both the light and dark site themes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from scipy.stats import binom, norm

mpl.use("Agg")
import matplotlib.pyplot as plt

from sensopy.discrimination import (
    DUO_TRIO,
    FOUR_AFC,
    THREE_AFC,
    TRIANGLE,
    TWO_AFC,
    UNSPECIFIED_TETRAD,
    DiscriminationTest,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "plots"

INK = "#8a8f98"
BLUE = "#2f6fdb"
ORANGE = "#e07a1f"
GREY = "#9aa0a8"

mpl.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "04304ad1-5950-4c2a-b5ea-58eb3234aed2",
        "legend.frameon": False,
    },
)


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg", metadata={"Date": None})
    plt.close(fig)


def thurstonian() -> None:
    """Two sensory distributions separated by d'."""
    d = 1.5
    x = np.linspace(-4, 4 + d, 400)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(x, norm.pdf(x), color=BLUE, alpha=0.25)
    ax.fill_between(x, norm.pdf(x, loc=d), color=ORANGE, alpha=0.25)
    ax.plot(x, norm.pdf(x), color=BLUE, label="Product A")
    ax.plot(x, norm.pdf(x, loc=d), color=ORANGE, label="Product B")
    top = norm.pdf(0) * 1.12
    ax.annotate("", xy=(d, top), xytext=(0, top), arrowprops={"arrowstyle": "<->", "color": INK})
    ax.text(d / 2, top + 0.015, "d′", ha="center", va="bottom", fontsize=13)
    ax.set_xlabel("Sensory intensity")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.set_ylim(0, top + 0.07)
    ax.legend(loc="upper right")
    _save(fig, "thurstonian.svg")


def psychometric() -> None:
    """P_c as a function of d' for several methods."""
    d = np.linspace(0, 5, 100)
    methods = {
        "2-AFC": TWO_AFC,
        "Duo-trio": DUO_TRIO,
        "3-AFC": THREE_AFC,
        "4-AFC": FOUR_AFC,
        "Triangle": TRIANGLE,
        "Tetrad (unspecified)": UNSPECIFIED_TETRAD,
    }
    colors = ["#2f6fdb", "#7a5cd6", "#1f9d8a", "#c2a31a", "#e07a1f", "#d1495b"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (label, method), color in zip(methods.items(), colors, strict=True):
        ax.plot(d, [method.psychometric_function(v) for v in d], label=label, color=color)
    ax.set_xlabel("d′")
    ax.set_ylabel("Probability of a correct response, $P_c$")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    _save(fig, "psychometric.svg")


def binomial_power() -> None:
    """Null and alternative binomial distributions with the critical value."""
    n = 30
    pc0 = TRIANGLE.guessing
    pc1 = TRIANGLE.psychometric_function(1.5)
    xcrit = int(binom.ppf(0.95, n, pc0)) + 1
    power = 1 - binom.cdf(xcrit - 1, n, pc1)
    x = np.arange(0, n + 1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.4
    ax.bar(x - width / 2, binom.pmf(x, n, pc0), width, color=GREY, label=f"H₀: $P_c$ = {pc0:.2f}")
    ax.bar(x + width / 2, binom.pmf(x, n, pc1), width, color=BLUE, alpha=0.8, label=f"True d′ = 1.5: $P_c$ = {pc1:.2f}")
    ax.set_xlim(-0.5, 27)
    ax.set_ylim(0, 0.24)
    ax.axvline(xcrit - 0.5, color=ORANGE, linestyle="--")
    ax.text(xcrit - 0.2, 0.235, f"critical value:\n≥ {xcrit} correct", color=ORANGE, va="top")
    ax.set_xlabel(f"Correct responses out of n = {n}")
    ax.set_ylabel("Probability")
    ax.set_title(f"Power at α = 0.05: {power:.2f}", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, "binomial-power.svg")


def power_curves() -> None:
    """Simulated power vs. d' for several panel sizes."""
    test = DiscriminationTest(TRIANGLE)
    ds = np.linspace(0, 2.5, 11)
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, color in zip((20, 40, 80), (ORANGE, BLUE, "#1f9d8a"), strict=True):
        power = [test.simulate(d, n, size=2000, rng=42).power for d in ds]
        ax.plot(ds, power, marker="o", ms=4, color=color, label=f"n = {n}")
    ax.axhline(0.05, color=GREY, linestyle=":")
    ax.axhline(0.8, color=GREY, linestyle=":")
    ax.text(ds[-1], 0.82, "80%", ha="right", va="bottom")
    ax.set_xlabel("True d′")
    ax.set_ylabel("Estimated power")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    _save(fig, "power-curves.svg")


if __name__ == "__main__":
    thurstonian()
    psychometric()
    binomial_power()
    power_curves()
