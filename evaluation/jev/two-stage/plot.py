"""Publication-style scatter of retrieval quality and extra decision latency."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).parent
src = json.loads((P / "plot-data.json").read_text())
pts = src["existing_points"]
src["quality_only_methods"] = {"naive": 65.37916666666666, "colbert": 58.7}
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none", "axes.linewidth": 0.8}
)
fig, ax = plt.subplots(figsize=(12, 7.8))
fig.subplots_adjust(left=0.105, right=0.975, top=0.86, bottom=0.22)
ink = "#202733"
green = "#087f6d"
blue = "#486c99"
gray = "#7f8894"
ax.set_xlim(-1.1, 25.5)
ax.set_ylim(55, 92)
ax.set_xticks([0, 5, 10, 15, 20, 25])
ax.set_yticks([60, 65, 70, 75, 80, 85, 90])
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#a8afb6")
ax.tick_params(colors=ink, labelsize=11, direction="out", length=4)
ax.grid(axis="y", color="#e9ecf0", lw=0.7)
ax.set_axisbelow(True)
ax.set_xlabel("Additional model-call latency (s)", labelpad=12, fontsize=13)
ax.set_ylabel("Average Recall@5 (%)", labelpad=12, fontsize=13)
# No reranking or decision API is required by the dense-retrieval baselines.
naive = [
    ("NV-Embed-v2", 73.1),
    ("BGE-large-en-v1.5", src["quality_only_methods"]["naive"]),
    ("ColBERTv2", src["quality_only_methods"]["colbert"]),
]
for name, y in naive:
    ax.scatter(0, y, marker="s", s=67, facecolor="white", edgecolor=gray, lw=1.5, zorder=4)
    ax.annotate(
        f"Naive RAG · {name}  ({y:.2f}%)",
        (0, y),
        xytext=(14, -4),
        textcoords="offset points",
        fontsize=11,
        color="#515a65",
    )
# The dotted guide is the nondominated set at the reference latency estimates.
ax.plot(
    [0, pts["jev_single"]["latency"], pts["jev_two"]["latency"]],
    [73.1, pts["jev_single"]["quality"], pts["jev_two"]["quality"]],
    ls=(0, (3, 3)),
    lw=1.5,
    color=green,
    alpha=0.75,
    zorder=2,
    label="Pareto frontier",
)
ax.legend(loc="upper right", frameon=False, fontsize=11, handlelength=2.4, labelcolor=green)
for key, p in pts.items():
    x, y = p["latency"], p["quality"]
    if key == "jev_two":
        ax.scatter(x, y, s=180, marker="*", color=green, edgecolor=green, zorder=5)
        ax.annotate(
            "Vector Graph RAG + Jev",
            (x, y),
            xytext=(20, 19),
            textcoords="offset points",
            fontsize=14,
            weight="bold",
            color=green,
        )
        ax.annotate(
            "Two-stage · 86.07% · 3.13 s",
            (x, y),
            xytext=(20, 0),
            textcoords="offset points",
            fontsize=12,
            color=green,
        )
    elif key == "jev_single":
        ax.scatter(x, y, s=62, marker="D", facecolor="white", edgecolor=green, lw=1.5, zorder=4)
        ax.annotate(
            "VGRAG + Jev · relations only",
            (x, y),
            xytext=(16, -16),
            textcoords="offset points",
            fontsize=11,
            color=green,
        )
        ax.annotate(
            "80.04% · 2.28 s",
            (x, y),
            xytext=(16, -32),
            textcoords="offset points",
            fontsize=10.5,
            color=green,
        )
    else:
        ax.scatter(x, y, s=65, marker="o", color=blue, zorder=4)
        if key == "gpt-5-mini":
            text = "VGRAG + GPT-5-mini\n83.56% · ≈20 s"
            off = (0, 15)
            ha = "center"
        elif key == "hippo2":
            text = "HippoRAG 2\n82.55% · ≈8.3 s"
            off = (12, 4)
            ha = "left"
        else:
            text = "VGRAG + GPT-4o-mini\n77.88% · ≈7 s"
            off = (30, -23)
            ha = "left"
        ax.annotate(
            text,
            (x, y),
            xytext=off,
            textcoords="offset points",
            ha=ha,
            fontsize=11,
            color=blue,
            linespacing=1.45,
        )
fig.text(0.105, 0.938, "Retrieval quality–latency trade-off", fontsize=21, weight="bold", color=ink)
fig.text(
    0.105,
    0.895,
    "MuSiQue + 2Wiki  |  Equal-weight average  |  1,000 queries per dataset",
    fontsize=11.5,
    color="#606973",
)
ax.text(
    0.98,
    0.04,
    "Better: upper left",
    transform=ax.transAxes,
    ha="right",
    fontsize=11,
    color="#606973",
)
fig.text(
    0.105,
    0.102,
    "Jev: measured means. LLMs: estimated range midpoints (≈). Frontier uses these reference times.",
    fontsize=9.5,
    color="#606973",
)
fig.text(
    0.105,
    0.067,
    "0 s = no extra model call. Embedding, retrieval and answer generation are excluded for all methods.",
    fontsize=9.5,
    color="#606973",
)
for ext in ["png", "svg"]:
    fig.savefig(
        P.parents[2] / f"docs/assets/evaluation/quality-latency.{ext}", dpi=320, facecolor="white"
    )

# Normalize generated SVG whitespace for clean source diffs.
svg = P.parents[2] / "docs/assets/evaluation/quality-latency.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
