#!/usr/bin/env python3
"""Plot benchmark results from JSON files into PNG charts.

Usage::

    python -m benchmarks.plot_results                          # defaults
    python -m benchmarks.plot_results --speed results/readme_speed.json
    python -m benchmarks.plot_results --io results/readme_io.json
    python -m benchmarks.plot_results --outdir ./plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

DEFAULT_DIR = Path(__file__).resolve().parent / "results"


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ── Chart: README figure ────────────────────────────────────────────────────
# Name mapping for the speed panel — one entry per library (best config)
_SPEED_NAMES = {
    "avdec": "avdec",  # historical data keeps its original name
    "tensorcodec": "tensorcodec",
    "torchcodec(seek=approximate,thr=1)": "torchcodec (approx)",
    "torchcodec(seek=exact,thr=1)": "torchcodec (exact)",
    "decord": "decord",
    "opencv": "opencv",
    "torchvision-pyav": "torchvision",
}

# Name mapping for the I/O panel — includes both torchcodec configs
_IO_NAMES = {
    "avdec": "avdec",  # historical data keeps its original name
    "tensorcodec": "tensorcodec",
    "torchcodec(seek=approximate,thr=1)": "torchcodec (approx)",
    "torchcodec(seek=exact,thr=1)": "torchcodec (exact)",
    "decord": "decord",
    "opencv": "opencv",
    "torchvision-pyav": "torchvision",
}

# Order: avdec first (hero), then others; exact-seek variant last in I/O
_SPEED_ORDER = ["tensorcodec", "avdec", "torchcodec (approx)", "torchcodec (exact)", "decord", "opencv", "torchvision"]
_IO_ORDER = ["tensorcodec", "avdec", "torchcodec (approx)", "torchcodec (exact)"]


def plot_readme(speed_data: list[dict], io_data: list[dict], outdir: Path) -> Path:
    """Two-panel README figure: FPS + Disk I/O per frame (temporal_window).

    *speed_data* provides FPS for all decoders (best config per library).
    *io_data* provides bytes_per_frame — includes both torchcodec configs
    so readers can see the I/O cliff from exact-seek mode.
    """
    # Build FPS lookup from speed data
    fps_map: dict[str, float] = {}
    for r in speed_data:
        simple = _SPEED_NAMES.get(r["decoder"])
        if simple and r["scenario"] == "temporal_window" and simple not in fps_map:
            fps_map[simple] = r["fps"]

    # Build I/O lookup from io data (uses _IO_NAMES to include both configs)
    bpf_map: dict[str, float] = {}
    for r in io_data:
        simple = _IO_NAMES.get(r["decoder"])
        if simple and r["scenario"] == "temporal_window" and simple not in bpf_map:
            bpf = r.get("bytes_per_frame")
            if bpf and bpf > 0:
                bpf_map[simple] = bpf

    # FPS panel: all decoders in _SPEED_ORDER
    fps_labels = [n for n in _SPEED_ORDER if n in fps_map]
    fps_vals = [fps_map[n] for n in fps_labels]

    # I/O panel: decoders with data, in _IO_ORDER
    io_labels = [n for n in _IO_ORDER if n in bpf_map]
    io_vals = [bpf_map[n] for n in io_labels]

    HERO = "#2563eb"
    OTHER = "#cbd5e1"
    WARN = "#f59e0b"  # amber for the "wrong config" bar

    n_fps = len(fps_labels)
    n_io = len(io_labels)
    row_h = 0.55
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11, max(n_fps, n_io) * row_h + 1.4),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )

    # ── Panel 1: FPS ──
    y1 = list(range(n_fps))
    c1 = [HERO if lb in {"avdec", "tensorcodec"} else OTHER for lb in fps_labels]
    bars = ax1.barh(y1, fps_vals, color=c1, edgecolor="white", height=0.6)
    for bar, fps, lb in zip(bars, fps_vals, fps_labels):
        ax1.text(
            bar.get_width() + max(fps_vals) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{fps:,.0f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold" if lb in {"avdec", "tensorcodec"} else "normal",
        )
    ax1.set_yticks(y1)
    ax1.set_yticklabels(fps_labels, fontsize=12)
    ax1.invert_yaxis()
    ax1.set_xlabel("Frames per second  (higher is better)", fontsize=10)
    ax1.set_title("Decode Speed", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, max(fps_vals) * 1.22)
    ax1.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: Bytes/frame ──
    y2 = list(range(n_io))
    c2 = []
    for lb in io_labels:
        if lb in {"avdec", "tensorcodec"}:
            c2.append(HERO)
        elif "exact" in lb:
            c2.append(WARN)
        else:
            c2.append(OTHER)
    bars2 = ax2.barh(y2, io_vals, color=c2, edgecolor="white", height=0.6)
    for bar, b, lb in zip(bars2, io_vals, io_labels):
        if b >= 1024 * 1024:
            txt = f"{b / (1024 * 1024):.1f} MB"
        elif b >= 1024:
            txt = f"{b / 1024:.1f} KB"
        else:
            txt = f"{b:.0f} B"
        ax2.text(
            bar.get_width() * 1.15,
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold" if lb in {"avdec", "tensorcodec"} else "normal",
        )
    ax2.set_yticks(y2)
    ax2.set_yticklabels(io_labels, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xlabel("Disk I/O per frame  (lower is better)", fontsize=10)
    ax2.set_title("Disk Read", fontsize=13, fontweight="bold")
    ax2.set_xscale("log")
    ax2.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Random seek + clip read  ·  temporal_window scenario", fontsize=11, color="#64748b", y=1.01)
    fig.tight_layout()
    out = outdir / "readme.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results to PNG")
    parser.add_argument(
        "--speed", type=Path, default=DEFAULT_DIR / "readme_speed.json", help="Path to speed benchmark JSON"
    )
    parser.add_argument("--io", type=Path, default=DEFAULT_DIR / "readme_io.json", help="Path to I/O benchmark JSON")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_DIR, help="Output directory for PNGs")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.speed.exists() and args.io.exists():
        plot_readme(_load(args.speed), _load(args.io), args.outdir)
    else:
        missing = []
        if not args.speed.exists():
            missing.append(str(args.speed))
        if not args.io.exists():
            missing.append(str(args.io))
        print(f"Missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
