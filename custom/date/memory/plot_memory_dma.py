#!/usr/bin/env python3
"""Build the paper's side-by-side memory-request and DMA-transaction figure."""

import argparse
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DTYPES = ["fp8", "fp16", "fp32"]
VARIANTS = ["sgemm_tcu", "sgemm_tcu_sp", "op_dense", "op_s20", "op_s50", "op_s90"]
LABELS = {
    "sgemm_tcu": "SGEMM", "sgemm_tcu_sp": "SGEMM_SP",
    "op_dense": "SGEMM_OP dense", "op_s20": "SGEMM_OP 20%",
    "op_s50": "SGEMM_OP 50%", "op_s90": "SGEMM_OP 90%",
}
HATCHES = {
    "sgemm_tcu": "", "sgemm_tcu_sp": "..", "op_dense": "//",
    "op_s20": "\\\\", "op_s50": "xx", "op_s90": "++",
}
MEMORY_RE = re.compile(r"^(?:PERF2:\s*)?memory:\s*reqs=(\d+)\s*\(r=(\d+),\s*w=(\d+)\)")
TEST_RE = re.compile(r"^Testbench:\s*(\S+)")
META_RE = re.compile(
    r"(?:Sparsity Mode:(?P<mode>\d+)\s*\|\s*)?.*?"
    r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*B sparsity=(?P<b>[0-9.]+)%.*?\|\s*It=(?P<dtype>\S+)"
)
FALLBACK_META_RE = re.compile(r"MxNxK=\d+x\d+x\d+.*?\|\s*It=(?P<dtype>\S+)")
DMA_META_RE = re.compile(
    r"Sparsity Mode:(?P<mode>\d+)\s*\|\s*MxNxK=(?P<m>\d+)x(?P<n>\d+)x(?P<k>\d+)\s*\|\s*"
    r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*B sparsity=(?P<b>[0-9.]+)%\s*\|\s*It=(?P<dtype>\S+)"
)
DMA_RE = re.compile(r"^PERF6: dxa: transfers=\d+, gmem_reads=(?P<reads>\d+),")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stats_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent,
        help="Directory containing the copied run_b_*.stat and run_d_*.stat files.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="PNG output (default: <stats_dir>/memory_dma.png); a PDF is also written.",
    )
    return parser.parse_args()


def parse_memory(path):
    row = {"test": None, "dtype": None, "mode": None, "a": 0.0, "b": 0.0,
           "reads": None, "writes": None}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        match = TEST_RE.match(line)
        if match:
            row["test"] = match.group(1)
        match = META_RE.search(line)
        if match:
            row["dtype"] = match.group("dtype")
            row["mode"] = int(match.group("mode")) if match.group("mode") else None
            row["a"], row["b"] = float(match.group("a")), float(match.group("b"))
        elif row["dtype"] is None:
            match = FALLBACK_META_RE.search(line)
            if match:
                row["dtype"] = match.group("dtype")
        match = MEMORY_RE.match(line)
        if match and row["reads"] is None:
            row["reads"], row["writes"] = int(match.group(2)), int(match.group(3))

    missing = [key for key in ("test", "dtype", "reads", "writes") if row[key] is None]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    if row["test"] in ("sgemm_tcu", "sgemm_tcu_sp"):
        row["variant"] = row["test"]
    elif row["test"] == "sgemm_tcu_op" and row["mode"] == 0:
        row["variant"] = "op_dense"
    elif row["test"] == "sgemm_tcu_op":
        row["variant"] = f"op_s{round((row['a'] + row['b']) / 2)}"
    else:
        raise ValueError(f"{path}: unsupported testbench {row['test']}")
    return row


def ratio(dtype):
    return {"fp32": 1, "fp16": 2, "bf16": 2, "fp8": 4}[dtype]


def tile_sizes(total, size):
    return [min(size, total - offset) for offset in range(0, total, size)]


def theoretical_dma(row):
    packing = ratio(row["dtype"])
    mt, nt, kt = tile_sizes(row["m"], 32), tile_sizes(row["n"], 32), tile_sizes(row["k"], 32 * packing)
    a_dense = sum(math.ceil(m * k / (16 * packing)) * len(nt) for m in mt for k in kt)
    b_dense = sum(math.ceil(k * n / (16 * packing)) * len(mt) for n in nt for k in kt)

    def compressed(dense, sparsity):
        return dense * (1 - sparsity / 100) + dense / (32 / packing)

    a_count = compressed(a_dense, row["a"]) if row["mode"] == 2 else a_dense
    b_count = compressed(b_dense, row["b"]) if row["mode"] else b_dense
    return a_count + b_count


def parse_dma(path):
    row = {"mode": None, "m": None, "n": None, "k": None, "a": None, "b": None,
           "dtype": None, "reads": None}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        match = DMA_META_RE.search(line)
        if match:
            row.update(mode=int(match.group("mode")), m=int(match.group("m")),
                       n=int(match.group("n")), k=int(match.group("k")),
                       a=float(match.group("a")), b=float(match.group("b")),
                       dtype=match.group("dtype"))
        match = DMA_RE.match(line)
        if match:
            row["reads"] = int(match.group("reads"))
    missing = [key for key, value in row.items() if value is None]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    row["theory"] = theoretical_dma(row)
    row["case"] = f"A: {round(row['a'])}%, B: {round(row['b'])}%"
    row["mode_label"] = {0: "Dense", 1: "A raw, B compressed", 2: "A and B compressed"}[row["mode"]]
    return row


def style_axis(ax):
    ax.grid(True, axis="y", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.9)


def memory_panel(ax, rows):
    by_key = {(row["dtype"], row["variant"]): row for row in rows}
    group_width, width = 0.72, 0.72 / len(VARIANTS)
    centers = list(range(len(DTYPES)))
    maximum = max(row["reads"] + row["writes"] for row in rows)
    baselines = {dtype: sum(by_key[(dtype, "sgemm_tcu")][k] for k in ("reads", "writes")) for dtype in DTYPES}
    for vi, variant in enumerate(VARIANTS):
        xs, reads, writes = [], [], []
        for di, dtype in enumerate(DTYPES):
            row = by_key[(dtype, variant)]
            xs.append(di - group_width / 2 + width * (vi + 0.5))
            reads.append(row["reads"])
            writes.append(row["writes"])
        bars = ax.bar(xs, writes, width * 0.88, color="#c7533b", edgecolor="#222222",
                      linewidth=0.65, hatch=HATCHES[variant])
        ax.bar(xs, reads, width * 0.88, bottom=writes, color="#2f6f9f", edgecolor="#222222",
               linewidth=0.65, hatch=HATCHES[variant])
        for bar, dtype, read, write in zip(bars, DTYPES, reads, writes):
            total = read + write
            ax.text(bar.get_x() + bar.get_width() / 2, total + maximum * .018,
                    f"{baselines[dtype] / total:.1f}x", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.set_ylabel(r"Memory Requests ($\times 10^3$)", fontsize=9)
    ax.set_xlabel("Input Type", fontsize=9, labelpad=1)
    ax.set_xticks(centers, DTYPES)
    ax.tick_params(axis="both", labelsize=8, pad=1.5)
    ax.set_yticks([0, 50000, 100000, 150000], ["0", "50", "100", "150"])
    ax.set_ylim(0, maximum * 1.28)
    handles = [Patch(facecolor="#2f6f9f", edgecolor="#222222", label="Read requests"),
               Patch(facecolor="#c7533b", edgecolor="#222222", label="Write requests")]
    handles += [Patch(facecolor="white", edgecolor="#222222", hatch=HATCHES[v], label=LABELS[v]) for v in VARIANTS]
    ax.legend(handles=handles, title="Request type and kernel / sparsity mode",
              loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=4,
              fontsize=6.6, title_fontsize=7, frameon=True, columnspacing=.65,
              handlelength=1.25, handletextpad=.3, borderpad=.3, labelspacing=.25)
    style_axis(ax)


def dma_panel(ax, rows):
    colors = {"Dense": "#4f5d75", "A and B compressed": "#2f6f9f",
              "A raw, B compressed": "#c7533b"}
    xs, width = list(range(len(rows))), 0.25
    maximum = max(max(row["reads"], row["theory"]) for row in rows)
    seen = set()
    for x, row in zip(xs, rows):
        label = row["mode_label"] if row["mode_label"] not in seen else None
        seen.add(row["mode_label"])
        ax.bar(x - width / 2, row["reads"], width, color=colors[row["mode_label"]], label=label)
        ax.bar(x + width / 2, row["theory"], width, facecolor="white", edgecolor="#222222",
               hatch="///", linewidth=1, label="Theoretical minimum" if x == 0 else None)
        ax.text(x - width / 2, row["reads"] + maximum * .018, f"{row['reads']}",
                ha="center", va="bottom", fontsize=7, rotation=90)
        ax.text(x + width / 2, row["theory"] + maximum * .018, f"{row['theory']:.0f}",
                ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_ylabel("DMA Transactions", fontsize=9)
    ax.set_xlabel("A/B Sparsity (%)", fontsize=9, labelpad=0)
    ax.set_xticks(xs, [f"{round(row['a'])}/{round(row['b'])}" for row in rows],
                  rotation=35, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8, pad=1.5)
    ax.set_ylim(0, maximum * 1.25)
    handles, labels = ax.get_legend_handles_labels()
    explanatory = {
        "Dense": "Dense",
        "Theoretical minimum": "Theoretical minimum",
        "A and B compressed": "A and B compressed",
        "A raw, B compressed": "A uncompressed, B compressed",
    }
    ax.legend(handles, [explanatory[label] for label in labels],
              title="DMA storage mode / reference", loc="lower center",
              bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=6.6,
              title_fontsize=7, frameon=True, columnspacing=.8,
              handlelength=1.3, handletextpad=.35, borderpad=.3, labelspacing=.25)
    style_axis(ax)


def main():
    args = arguments()
    stats_dir = args.stats_dir.resolve()
    output = args.output or stats_dir / "memory_dma.png"
    memory_rows = [parse_memory(stats_dir / f"run_b_{run}.stat") for run in range(1, 19)]
    dma_rows = [parse_dma(stats_dir / f"run_d_{run}.stat") for run in range(1, 8)]

    # A compact landscape layout keeps both panels readable without the height of
    # the original 16 x 5.4 inch figure.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), gridspec_kw={"width_ratios": [1.08, 1]})
    memory_panel(axes[0], memory_rows)
    dma_panel(axes[1], dma_rows)
    fig.subplots_adjust(left=.09, right=.995, top=.72, bottom=.26, wspace=.30)
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
