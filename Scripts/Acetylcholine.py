from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch

# ---------------------------------------------------------------------------
# KONFIGURATION & STYLING
# ---------------------------------------------------------------------------
FIGSIZE = (11, 8)
DPI = 400
OUTPUT_DIR = Path("/home/tara/PyCharmMiscProject/MA_MPE/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(r"/home/tara/Dokumente/MA/2 - Data/Excel-files")

SPECTRA_CONFIG = [
    {
        "name": "Acetylcholine-chloride",
        "description": "Acetylcholine",
        "file": "20240629_AcetylCholine_300K_sample1.CSV",
        "peaks": [
            {"wavenumber": 3448, "label": "1"},
            {"wavenumber": 3377, "label": "2"},
            {"wavenumber": 3009, "label": "3"},
            {"wavenumber": 2979, "label": "4", "offset": (3, 15)},
            {"wavenumber": 2924, "label": "5", "offset": (3, 15)},
            {"wavenumber": 1734, "label": "6"},
            {"wavenumber": 1489, "label": "7"},
            {"wavenumber": 1457, "label": "8"},
            {"wavenumber": 1424, "label": "9"},
            {"wavenumber": 1378, "label": "10", "offset": (0,35)},
            {"wavenumber": 1366, "label": "11", "offset": (8, 20)},
            {"wavenumber": 1234, "label": "12", "offset": (40, -10)},
            {"wavenumber": 1145, "label": "13", "offset": (-5, 20)},
            {"wavenumber": 1088, "label": "14", "offset": (-2, 20)},
            {"wavenumber": 1053, "label": "15", "offset": (0, 30)},
            {"wavenumber": 1028, "label": "16", "offset": (2, 35)},
            {"wavenumber": 1007, "label": "17", "offset": (4, 15)},
            {"wavenumber": 959, "label": "18"},
            {"wavenumber": 919, "label": "19", "offset": (2, 20)},
            {"wavenumber": 877, "label": "20", "offset": (0, 25)},
            {"wavenumber": 848, "label": "21", "offset": (3, 20)},
            {"wavenumber": 825, "label": "22", "offset": (15, 25)},
            {"wavenumber": 718, "label": "23", "offset": (0, 25)},
            {"wavenumber": 644, "label": "24", "offset": (0, 25)},
            {"wavenumber": 608, "label": "25", "offset": (4, 20)},
        ],
    },
]

TEXT_X_POS = 2650
TEXT_OFFSETS = {"Acetylcholine-chloride": 0.0001}
X_MIN, X_MAX = 550, 3800
CSV_SEP, CSV_DECIMAL = ";", ","

PEAK_TEXT_STYLE = {"fontsize": 16, "color": "black", "ha": "center", "va": "bottom"}
PEAK_TEXT_OFFSET = (0, 20)
PEAK_ARROW_STYLE = {"arrowstyle": "-", "color": "black", "lw": 1}

SPECTRA_COLORS = {"Acetylcholine-chloride": "#b70f3b"}
SCALING_FACTORS = {"Acetylcholine-chloride": 1}

# ---------------------------------------------------------------------------
# FUNKTIONEN
# ---------------------------------------------------------------------------

def create_figure(xlabel=r"Wavenumber / cm$^{-1}$", ylabel="Absorbance (a.u.)", labelsize=24, ticksize=20):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    # Ticks nach innen
    ax.tick_params(axis="both", which="major", labelsize=ticksize, direction="in", top=True, right=True)
    return fig, ax

def load_csv_spectrum(path):
    df = pd.read_csv(path, sep=CSV_SEP, decimal=CSV_DECIMAL)
    return df.iloc[:, 0].values, df.iloc[:, 1].values

def get_peak_coordinates(x, y, wn):
    idx = np.abs(x - wn).argmin()
    return x[idx], y[idx]

def annotate_peaks(ax, x, y, peaks):
    for p in peaks:
        xp, yp = get_peak_coordinates(x, y, p["wavenumber"])
        current_offset = p.get("offset", PEAK_TEXT_OFFSET)
        ax.annotate(p["label"], xy=(xp, yp), xytext=current_offset, textcoords="offset points",
                    arrowprops=PEAK_ARROW_STYLE, **PEAK_TEXT_STYLE)

# ---------------------------------------------------------------------------
def main():
    fig, ax = create_figure()
    spectra_data, max_range = [], 0

    for spec in SPECTRA_CONFIG:
        x, y = load_csv_spectrum(DATA_DIR / spec["file"])
        mask = (x >= X_MIN) & (x <= X_MAX)
        x, y = x[mask], y[mask] * SCALING_FACTORS[spec["name"]]
        spectra_data.append({**spec, "x": x, "y": y})
        max_range = max(max_range, y.max() - y.min())

    offsets = {"Acetylcholine-chloride": 0}

    # -------- Hauptplot --------
    for spec in spectra_data:
        y_off = spec["y"] + offsets[spec["name"]]
        ax.plot(spec["x"], y_off, color=SPECTRA_COLORS[spec["name"]])
        ax.text(TEXT_X_POS, np.mean(y_off) + TEXT_OFFSETS[spec["name"]], spec["description"],
                fontsize=16, color=SPECTRA_COLORS[spec["name"]], ha='left', va='center')

        if spec["name"] == "Acetylcholine-chloride":
            peaks_main = [p for p in spec["peaks"] if p["wavenumber"] > 1550]
            annotate_peaks(ax, spec["x"], y_off, peaks_main)
        else:
            annotate_peaks(ax, spec["x"], y_off, spec["peaks"])

    # -------- Choline Zoom --------
    choline = next(s for s in spectra_data if s["name"] == "Acetylcholine-chloride")
    y_ch = choline["y"] + offsets["Acetylcholine-chloride"]
    mask_zoom = (choline["x"] >= 550) & (choline["x"] <= 1550)
    y_zoom = y_ch[mask_zoom]

    # -------- Inset --------
    axins = inset_axes(ax, width="60%", height="37%", bbox_to_anchor=(-0.19, 0.3, 1, 1),
                       bbox_transform=ax.transAxes, loc="center")
    axins.plot(choline["x"], y_ch, color=SPECTRA_COLORS["Acetylcholine-chloride"])
    axins.set_xlim(1550, 565)
    axins.set_ylim(y_zoom.min(), y_zoom.max() * 1.05)

    # Inset Styling: Ticks nach innen und Beschriftung
    axins.tick_params(axis="both", which="major", labelsize=12, direction="in", top=True, right=True)
    axins.set_yticks([])
    axins.set_xlabel(r"Wavenumber / cm$^{-1}$", fontsize=14)
    # Optional: ylabel falls gewünscht: axins.set_ylabel("Abs.", fontsize=10)

    peaks_zoom = [p for p in choline["peaks"] if 565 <= p["wavenumber"] <= 1550]
    annotate_peaks(axins, choline["x"], y_ch, peaks_zoom)

    # -------- Rechteck & Verbindung --------
    x0, x1, y0, y1 = 565, 1560, y_zoom.min(), y_zoom.max()
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="gray", facecolor="none",
                     linestyle="--", lw=0.6, alpha=0.4)
    ax.add_patch(rect)

    con1 = ConnectionPatch(xyA=(565, y1+0.012), xyB=(565, y1), coordsA=axins.transData,
                           coordsB=ax.transData, color="gray", linestyle="--", alpha=0.4, lw=0.6)
    fig.add_artist(con1)
    con2 = ConnectionPatch(xyA=(1550, y0), xyB=(1560, y0), coordsA=axins.transData,
                           coordsB=ax.transData, color="gray", linestyle="--", alpha=0.4, lw=0.6)
    fig.add_artist(con2)

    ax.set_xlim(X_MAX, X_MIN)
    ax.set_ylim(0.025, 0.35)
    # -------- Speichern --------
    file_base = "Acetylcholine"
    plt.savefig(OUTPUT_DIR / f"{file_base}.png", transparent=True, dpi=DPI, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{file_base}.pdf", transparent=True, bbox_inches="tight")

    print(f"Dateien erfolgreich in {OUTPUT_DIR} gespeichert.")
    plt.show()

if __name__ == "__main__":
    main()