#!/usr/bin/env python3
"""Plot total kernel instructions on a logarithmic scale."""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DTYPES = ["fp8", "fp16", "fp32"]
VARIANTS = [
    "sgemm_tcu", "sgemm_tcu_sp", "sgemm_tcu_op_dense",
    "sgemm_tcu_op_s20", "sgemm_tcu_op_s50", "sgemm_tcu_op_s90",
]
LABELS = {
    "sgemm_tcu": "SGEMM",
    "sgemm_tcu_sp": "SGEMM_SP",
    "sgemm_tcu_op_dense": "SGEMM_OP dense",
    "sgemm_tcu_op_s20": "SGEMM_OP 20%",
    "sgemm_tcu_op_s50": "SGEMM_OP 50%",
    "sgemm_tcu_op_s90": "SGEMM_OP 90%",
}
COLORS = {
    "sgemm_tcu": "#4f5d75",
    "sgemm_tcu_sp": "#7a5ea8",
    "sgemm_tcu_op_dense": "#2f6f9f",
    "sgemm_tcu_op_s20": "#558b2f",
    "sgemm_tcu_op_s50": "#b8792d",
    "sgemm_tcu_op_s90": "#c7533b",
}

TEST_RE = re.compile(r"^Testbench:\s*(\S+)")
META_RE = re.compile(
    r"(?:Sparsity Mode:(?P<mode>\d+)\s*\|\s*)?"
    r"MxNxK=\d+x\d+x\d+.*?\|\s*It=(?P<dtype>\S+)"
)
SPARSITY_RE = re.compile(
    r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*B sparsity=(?P<b>[0-9.]+)%"
)
INSTRUCTIONS_RE = re.compile(r"^Total kernel instructions:\s*(\d+)")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stats_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent,
        help="Directory containing run_b_1.stat through run_b_18.stat.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="PNG output path (default: <stats_dir>/instructions.png).",
    )
    return parser.parse_args()


def parse_stat(path):
    row = {"test": None, "dtype": None, "mode": None, "a": None, "b": None,
           "instructions": None}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        match = TEST_RE.match(line)
        if match:
            row["test"] = match.group(1)
        match = META_RE.search(line)
        if match:
            row["dtype"] = match.group("dtype")
            row["mode"] = int(match.group("mode")) if match.group("mode") else None
            sparsity = SPARSITY_RE.search(line)
            if sparsity:
                row["a"] = float(sparsity.group("a"))
                row["b"] = float(sparsity.group("b"))
        match = INSTRUCTIONS_RE.match(line)
        if match:
            row["instructions"] = int(match.group(1))

    missing = [key for key in ("test", "dtype", "instructions") if row[key] is None]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")

    if row["test"] in ("sgemm_tcu", "sgemm_tcu_sp"):
        row["variant"] = row["test"]
    elif row["test"] == "sgemm_tcu_op" and row["mode"] == 0:
        row["variant"] = "sgemm_tcu_op_dense"
    elif row["test"] == "sgemm_tcu_op" and row["a"] is not None and row["b"] is not None:
        row["variant"] = f"sgemm_tcu_op_s{round((row['a'] + row['b']) / 2)}"
    else:
        raise ValueError(f"{path}: unsupported or incomplete test configuration")
    return row


def plot(rows, output):
    by_key = {(row["dtype"], row["variant"]): row for row in rows}
    centers = list(range(len(DTYPES)))
    group_width = 0.84
    bar_width = group_width / len(VARIANTS)

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    for variant_index, variant in enumerate(VARIANTS):
        positions = [
            center - group_width / 2 + bar_width * (variant_index + 0.5)
            for center in centers
        ]
        values = [by_key[(dtype, variant)]["instructions"] for dtype in DTYPES]
        ax.bar(
            positions, values, width=bar_width * 0.88,
            label=LABELS[variant], color=COLORS[variant],
            edgecolor="#222222", linewidth=0.8,
        )

    ax.set_yscale("log")
    ax.set_ylim(10 ** 3, max(row["instructions"] for row in rows) * 1.35)
    ax.set_xlabel("Input Data Type")
    ax.set_ylabel("Total Kernel Instructions (log scale)")
    ax.set_xticks(centers, DTYPES)
    ax.grid(True, axis="y", which="both", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Kernel / OP Sparsity Mode", loc="upper center",
        bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True,
        fancybox=False, edgecolor="#333333",
    )
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf


def main():
    args = arguments()
    stats_dir = args.stats_dir.resolve()
    output = args.output or stats_dir / "instructions.png"
    rows = [parse_stat(stats_dir / f"run_b_{run}.stat") for run in range(1, 19)]
    pdf = plot(rows, output)
    print(f"wrote {output}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
