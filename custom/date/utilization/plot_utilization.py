#!/usr/bin/env python3
"""Plot TCU utilization by kernel/datatype and by K dimension."""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DTYPES = ["fp8", "fp16", "fp32"]
VARIANTS = [
    "sgemm_tcu", "sgemm_tcu_sp", "op_dense", "op_s20", "op_s50", "op_s90",
]
VARIANT_LABELS = {
    "sgemm_tcu": "SGEMM", "sgemm_tcu_sp": "SGEMM_SP",
    "op_dense": "SGEMM_OP dense", "op_s20": "SGEMM_OP 20%",
    "op_s50": "SGEMM_OP 50%", "op_s90": "SGEMM_OP 90%",
}
VARIANT_COLORS = {
    "sgemm_tcu": "#4f5d75", "sgemm_tcu_sp": "#7a5ea8",
    "op_dense": "#2f6f9f", "op_s20": "#558b2f",
    "op_s50": "#b8792d", "op_s90": "#c7533b",
}
TEST_RE = re.compile(r"^Testbench:\s*(\S+)")
B_META_RE = re.compile(
    r"(?:Sparsity Mode:(?P<mode>\d+)\s*\|\s*)?MxNxK=\d+x\d+x\d+.*?\|\s*It=(?P<dtype>\S+)"
)
SPARSITY_RE = re.compile(r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*B sparsity=(?P<b>[0-9.]+)%")
K_META_RE = re.compile(r"MxNxK=\d+x\d+x(?P<k>\d+).*?\|\s*It=(?P<dtype>\S+)")
STEP_RE = re.compile(r"BLOCK_M=(?P<m>\d+),\s*BLOCK_N=(?P<n>\d+)")
UTIL_RE = re.compile(r"^TCU utilization:\s*(?P<value>[0-9.]+)%")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stats_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent,
        help="Directory containing copied run_b_*.stat and run_k_*.stat inputs.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="PNG output (default: <stats_dir>/tcu_utilization.png); PDF is also written.",
    )
    return parser.parse_args()


def parse_run_b(path):
    row = {"test": None, "dtype": None, "mode": None, "a": None, "b": None, "util": None}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        match = TEST_RE.match(line)
        if match:
            row["test"] = match.group(1)
        match = B_META_RE.search(line)
        if match:
            row["dtype"] = match.group("dtype")
            row["mode"] = int(match.group("mode")) if match.group("mode") else None
            sparsity = SPARSITY_RE.search(line)
            if sparsity:
                row["a"], row["b"] = float(sparsity.group("a")), float(sparsity.group("b"))
        match = UTIL_RE.match(line)
        if match:
            row["util"] = float(match.group("value"))
    missing = [key for key in ("test", "dtype", "util") if row[key] is None]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    if row["test"] in ("sgemm_tcu", "sgemm_tcu_sp"):
        row["variant"] = row["test"]
    elif row["test"] == "sgemm_tcu_op" and row["mode"] == 0:
        row["variant"] = "op_dense"
    elif row["test"] == "sgemm_tcu_op" and row["a"] is not None and row["b"] is not None:
        row["variant"] = f"op_s{round((row['a'] + row['b']) / 2)}"
    else:
        raise ValueError(f"{path}: unsupported test configuration")
    return row


def parse_run_k(path):
    row = {"k": None, "block_m": None, "block_n": None, "util": None}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        match = K_META_RE.search(line)
        if match:
            row["k"] = int(match.group("k"))
        match = STEP_RE.search(line)
        if match:
            row["block_m"], row["block_n"] = int(match.group("m")), int(match.group("n"))
        match = UTIL_RE.match(line)
        if match:
            row["util"] = float(match.group("value"))
    missing = [key for key, value in row.items() if value is None]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    return row


def style(ax):
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linewidth=.7, alpha=.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(.9)


def utilization_panel(ax, rows):
    by_key = {(row["dtype"], row["variant"]): row for row in rows}
    centers = list(range(len(DTYPES)))
    group_width = .82
    width = group_width / len(VARIANTS)
    for vi, variant in enumerate(VARIANTS):
        xs = [x - group_width / 2 + width * (vi + .5) for x in centers]
        values = [by_key[(dtype, variant)]["util"] for dtype in DTYPES]
        bars = ax.bar(xs, values, width * .86, label=VARIANT_LABELS[variant],
                      color=VARIANT_COLORS[variant], edgecolor="#222222", linewidth=.65)
        for bar, value in zip(bars, values):
            high = value >= 80
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value - 1.5 if high else value + 1.4, f"{value:.1f}%",
                    ha="center", va="top" if high else "bottom", rotation=90,
                    fontsize=6.2, color="white" if high else "#111111",
                    fontweight="bold" if high else "normal")
    ax.set_xlabel("Input Data Type", fontsize=9)
    ax.set_ylabel("TCU Multiplier Utilization (%)", fontsize=9)
    ax.set_xticks(centers, DTYPES)
    ax.legend(title="Kernel / OP sparsity mode", loc="lower center",
              bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=6.4,
              title_fontsize=7, frameon=True, columnspacing=.7,
              handlelength=1.3, handletextpad=.3, borderpad=.3)
    style(ax)


def k_panel(ax, rows):
    k_values = sorted({row["k"] for row in rows})
    steps = sorted({(row["block_m"], row["block_n"]) for row in rows})
    by_key = {(row["k"], row["block_m"], row["block_n"]): row for row in rows}
    centers = list(range(len(k_values)))
    group_width = .58
    width = group_width / len(steps)
    colors = ["#2f6f9f", "#c7533b"]
    for si, (block_m, block_n) in enumerate(steps):
        xs = [x - group_width / 2 + width * (si + .5) for x in centers]
        values = [by_key[(k, block_m, block_n)]["util"] for k in k_values]
        bars = ax.bar(xs, values, width * .86, label=f"Step {block_m}x{block_n}",
                      color=colors[si], edgecolor="#222222", linewidth=.7)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value - 2.0, f"{value:.1f}%",
                    ha="center", va="top", rotation=90, fontsize=6.8, color="white",
                    fontweight="bold")
    ax.set_xlabel("K Dimension", fontsize=9)
    ax.set_ylabel("TCU Multiplier Utilization (%)", fontsize=9)
    ax.set_xticks(centers, [str(k) for k in k_values])
    ax.legend(title="Outer-product step shape", loc="lower center",
              bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=7,
              title_fontsize=7, frameon=True, columnspacing=.9,
              handlelength=1.4, borderpad=.3)
    style(ax)


def main():
    args = arguments()
    stats_dir = args.stats_dir.resolve()
    output = args.output or stats_dir / "tcu_utilization.png"
    run_b = [parse_run_b(stats_dir / f"run_b_{run}.stat") for run in range(1, 19)]
    run_k = [parse_run_k(stats_dir / f"run_k_{run}.stat") for run in range(1, 9)]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.15), gridspec_kw={"width_ratios": [1.08, 1]})
    utilization_panel(axes[0], run_b)
    k_panel(axes[1], run_k)
    fig.subplots_adjust(left=.09, right=.995, top=.72, bottom=.25, wspace=.30)
    for ax, marker in zip(axes, ("(a)", "(b)")):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, .015, marker, ha="center", va="bottom", fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
