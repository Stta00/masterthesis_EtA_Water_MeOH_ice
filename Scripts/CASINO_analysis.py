import numpy as np
import matplotlib.pyplot as plt

# ── Parse Max Depth from each trajectory ──────────────────────────────────────
max_depths = []
with open("/mnt/user-data/uploads/simulation_ice.dat", "r") as f:
    for line in f:
        if line.strip().startswith("no\tno\tno"):
            parts = line.split("\t")
            try:
                max_depths.append(float(parts[5]))
            except (IndexError, ValueError):
                pass

max_depths = np.array(max_depths)

# ── CDF ───────────────────────────────────────────────────────────────────────
sorted_depths = np.sort(max_depths)
cdf = np.arange(1, len(sorted_depths) + 1) / len(sorted_depths)

methanol_end = 130.0
water_end    = 195.0
plot_max     = 500.0

frac_in_layers = np.sum(max_depths <= water_end) / len(max_depths)

# ── Style matching Fig2 ───────────────────────────────────────────────────────
COLOR_METHANOL     = "#E07B00"
COLOR_WATER        = "#1A3A8F"
COLOR_ETHANOLAMINE = "#1A6B2A"

LABEL_FS = 16
TICK_FS  = 14
ANNOT_FS = 13
LW_AXIS  = 1.5

plt.rcParams.update({"font.family": "sans-serif", "axes.linewidth": LW_AXIS})

fig, ax = plt.subplots(figsize=(7, 5.5))

ax.axvspan(0,            methanol_end, color=COLOR_METHANOL,     alpha=0.15, zorder=0)
ax.axvspan(methanol_end, water_end,    color=COLOR_WATER,        alpha=0.15, zorder=0)
ax.axvspan(water_end,    plot_max,     color=COLOR_ETHANOLAMINE, alpha=0.15, zorder=0)

ax.plot(sorted_depths, cdf, color="black", linewidth=2.0, zorder=3)

ax.text(methanol_end / 2,             0.50, "CH$_3$OH\nLayer",
        ha="center", va="center", rotation=90, fontsize=ANNOT_FS,
        color=COLOR_METHANOL, fontweight="bold")
ax.text((methanol_end + water_end)/2, 0.50, "H$_2$O\nLayer",
        ha="center", va="center", rotation=90, fontsize=ANNOT_FS,
        color=COLOR_WATER, fontweight="bold")
ax.text(water_end + 60,               0.50, "Ethanolamine\nLayer",
        ha="center", va="center", rotation=90, fontsize=ANNOT_FS,
        color=COLOR_ETHANOLAMINE, fontweight="bold")

ax.annotate(">99.8% of electrons implanted in\nCH$_3$OH and H$_2$O ice layers",
            xy=(water_end + 3, frac_in_layers),
            xytext=(255, 0.78),
            fontsize=ANNOT_FS - 1,
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
            ha="left", va="center")

ax.set_xlim(0, plot_max)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Depth (nm)", fontsize=LABEL_FS)
ax.set_ylabel("Fraction of Implanted Electrons", fontsize=LABEL_FS)
ax.tick_params(axis="both", which="both",
               direction="in", top=True, right=True,
               labelsize=TICK_FS, width=LW_AXIS, length=5)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/simulation_ice_plot.png", dpi=150, bbox_inches="tight")
print("Done.")