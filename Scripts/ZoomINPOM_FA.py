from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_helper_spectra import create_figure

# ---------------------------------------------------------------------------
# PFAD-KONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR_COMPARISON = Path(
    r"C:\Users\taras\OneDrive\Dokumente\MSc\MA\2 - Data\Excel-files\Comparison_All"
)

DATA_DIR_IRR = Path(
    r"C:\Users\taras\OneDrive\Dokumente\MSc\MA\2 - Data\Excel-files\MeOH_EtA_Irr"
)

# ---------------------------------------------------------------------------
# PLOT-BEREICH / DARSTELLUNG
# ---------------------------------------------------------------------------

# Fokus-Bereich: 900–1500 cm^-1 (IR-typisch invertiert: hohe Wellenzahl links)
X_LEFT = 1500
X_RIGHT = 900

CSV_SEP = ";"
CSV_DECIMAL = ","

# Skalierung Irradiation
IRR_FACTOR = 55

# Farben (fix)
SPECTRA_COLORS_FIXED = {
    "EtA-H_2O-MeOH": "#c243a2",  # pink
}

# ---------------------------------------------------------------------------
# SPEKTREN-KONFIGURATION
# ---------------------------------------------------------------------------

STATIC_SPECTRA = [
    {
        "name": "EtA-H_2O-MeOH",
        "description": "EtA + H₂O + \nCH$_3$OH",
        "file": "20240626_ice_Deposited_Etha_H2O_Methanol_20K_sample1.csv",
        "dir": DATA_DIR_COMPARISON,
        "scale": 1.0,
    },
]

IRR_SPECTRA = [
    {
        "name": "Irr 60min",
        "description": "Irr. 60 min",
        "file": "20240626_Irradiation_20K_sample6.csv",
        "dir": DATA_DIR_IRR,
        "scale": IRR_FACTOR,
    },
    {
        "name": "Irr 30min",
        "description": "Irr. 30 min",
        "file": "20240626_Irradiation_20K_sample5.csv",
        "dir": DATA_DIR_IRR,
        "scale": IRR_FACTOR,
    },
    {
        "name": "Irr 15min",
        "description": "Irr. 15 min",
        "file": "20240626_Irradiation_20K_sample4.csv",
        "dir": DATA_DIR_IRR,
        "scale": IRR_FACTOR,
    },
    {
        "name": "Irr 5min",
        "description": "Irr. 5 min",
        "file": "20240626_Irradiation_20K_sample3.csv",
        "dir": DATA_DIR_IRR,
        "scale": IRR_FACTOR,
    },
    {
        "name": "Irr 1min",
        "description": "Irr. 1 min",
        "file": "20240626_Irradiation_20K_sample2.csv",
        "dir": DATA_DIR_IRR,
        "scale": IRR_FACTOR,
    },
]

ORDERED_SPECTRA = STATIC_SPECTRA + IRR_SPECTRA

# ---------------------------------------------------------------------------
# PEAKS (Vertikallinien)
# ---------------------------------------------------------------------------

HCOOH_PEAKS = [1067, 1210, 1380]
POM_PEAKS = [990, 1095]
ALL_PEAKS = HCOOH_PEAKS + POM_PEAKS

# ---------------------------------------------------------------------------
# MANUELLE LABEL-POSITIONEN (x,y in Datenkoordinaten!)
# -> Diese Positionen kannst du frei anpassen.
# -> x_anchor ist der Wellenzahl-Ankerpunkt auf dem Irr-60min Spektrum.
# -> y_anchor_offset schiebt den Startpunkt etwas weg vom Spektrum.
# -> fontsize optional pro Label.
# ---------------------------------------------------------------------------

PEAK_LABELS_MANUAL = {
    1067: {"text": "HCOOH", "x": 1035, "y": 3.20, "x_anchor": 1067, "fontsize": 18},
    1210: {"text": "HCOOH", "x": 1160, "y": 3.20, "x_anchor": 1210, "fontsize": 18},
    1380: {"text": "HCOOH", "x": 1330, "y": 3.50, "x_anchor": 1380, "fontsize": 18},

    990:  {"text": "POM",   "x": 950,  "y": 3.00, "x_anchor": 990,  "fontsize": 18},
    1095: {"text": "POM",   "x": 1125, "y": 3.00, "x_anchor": 1095, "fontsize": 18},
}

# ---------------------------------------------------------------------------
# LAYOUT-TUNING
# ---------------------------------------------------------------------------

# Abstand zwischen den Spektren (kleiner = enger)
OFFSET_FACTOR = 0.45

# Extra Abstand nur für EtA+H2O+MeOH (in offset_step-Einheiten)
TOP_EXTRA_GAP_STEPS = 1.4

# Kleiner Abstand zwischen y-Achse (unten) und dem untersten Spektrum
Y_BOTTOM_PAD = 0.05  # Daten-Einheiten, klein halten

# Unteres Ende der gestrichelten Linien (damit sie sichtbar bleiben)
DASHED_LINE_YMIN = 0.0  # du hattest zuvor -0.5; hier auf 0 gesetzt

# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------------------------------

def load_csv_spectrum(path: Path, x_col=0, y_col=1):
    df = pd.read_csv(path, sep=CSV_SEP, decimal=CSV_DECIMAL)
    x = df.iloc[:, x_col].astype(float).values
    y = df.iloc[:, y_col].astype(float).values
    return x, y


def label_spectrum_left(ax, x_data, y_data, text, color, x_target, y_text_offset=0):
    """
    Label links platzieren (bei invertierter IR-Achse ist 'links' hohe Wellenzahl).
    """
    idx = int(np.abs(x_data - x_target).argmin())
    ax.annotate(
        text,
        xy=(x_data[idx], y_data[idx]),
        xytext=(0, -40),
        textcoords="offset points",
        color=color,
        fontsize=18,
        ha="left",
        va="center",
        clip_on=False,
    )


def draw_dashed_lines_from_top_to_zero(ax, x_data_top, y_data_top, peaks, ymin=0.0, color="black", lw=1.0):
    """
    Zeichnet für jeden Peak px eine gestrichelte Linie von y(top_spectrum@px) bis ymin (z.B. 0).
    """
    for px in peaks:
        idx = int(np.abs(x_data_top - px).argmin())
        x0 = float(x_data_top[idx])
        y_top = float(y_data_top[idx])

        ax.plot(
            [x0, x0],
            [ymin, y_top],
            linestyle="--",
            color=color,
            lw=lw,
            zorder=1,
        )


def add_manual_peak_labels_with_leaders(
    ax,
    labels_dict,
    x_data_anchor,
    y_data_anchor,
    color="black",
    fontsize=18,
    line_lw=1.0,
):
    """
    Setzt Labels an manuellen (x,y)-Positionen und zeichnet eine Leader-Line
    vom Peak auf dem ANCHOR-Spektrum (hier: Irr 60 min) zur Label-Position.
    """
    for _, cfg in labels_dict.items():
        text = cfg["text"]
        x_text = cfg["x"]
        y_text = cfg["y"]

        x_anchor = cfg.get("x_anchor", x_text)

        idx = int(np.abs(x_data_anchor - x_anchor).argmin())
        x0 = float(x_data_anchor[idx])
        y0 = float(y_data_anchor[idx]) + cfg.get("y_anchor_offset", 0.0)

        ax.annotate(
            text,
            xy=(x0, y0),                 # Startpunkt: Irr 60 min Spektrum
            xytext=(x_text, y_text),     # Textposition: manuell
            textcoords="data",
            color=color,
            fontsize=cfg.get("fontsize", fontsize),
            ha="center",
            va="bottom",
            arrowprops=dict(
                arrowstyle="-",          # nur Linie, kein Pfeilkopf
                lw=line_lw,
                color=color,
            ),
            clip_on=True,
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    fig, ax = create_figure()

    # Irradiation-Farben (stabil)
    irr_names = [s["name"] for s in IRR_SPECTRA]
    cmap = plt.get_cmap("magma")
    irr_cols = cmap(np.linspace(0.05, 0.75, len(irr_names)))
    irr_color_map = {name: irr_cols[i] for i, name in enumerate(irr_names)}

    def get_color(spec_name: str):
        if spec_name in SPECTRA_COLORS_FIXED:
            return SPECTRA_COLORS_FIXED[spec_name]
        if spec_name in irr_color_map:
            return irr_color_map[spec_name]
        return "black"

    # ----------------- Daten laden (900–1500 cm^-1) -----------------
    spectra_data = []
    max_range = 0.0

    for spec in ORDERED_SPECTRA:
        filepath = spec["dir"] / spec["file"]
        if not filepath.exists():
            print(f"Warnung: Datei nicht gefunden: {filepath}")
            continue

        x, y = load_csv_spectrum(filepath)

        lo, hi = min(X_RIGHT, X_LEFT), max(X_RIGHT, X_LEFT)
        mask = (x >= lo) & (x <= hi)
        x = x[mask]
        y = y[mask]

        if x.size == 0:
            print(f"Warnung: Keine Daten im Bereich {lo}–{hi} cm^-1 in {filepath}")
            continue

        y_scaled = y * float(spec.get("scale", 1.0))

        spectra_data.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "x": x,
                "y": y_scaled,
                "color": get_color(spec["name"]),
            }
        )

        max_range = max(max_range, float(y_scaled.max() - y_scaled.min()))

    if not spectra_data:
        print("Keine Spektren geladen – bitte Pfade/Dateinamen prüfen.")
        return

    # ----------------- Offsets -----------------
    offset_step = max_range * OFFSET_FACTOR if max_range > 0 else 0.15
    n = len(spectra_data)

    # Labels links
    x_label_target = X_LEFT - 5

    # Spektren-Anker für gestrichelte Linien (EtA+H2O+MeOH)
    top_x = top_y_off = None

    # Spektren-Anker für Leader-Lines (Irr 60 min)
    irr60_x = irr60_y_off = None

    # Track min/max (für tight y-limits)
    plotted_min = None
    plotted_max = None

    # ----------------- Plotten -----------------
    for i, s in enumerate(spectra_data):
        offset = (n - 1 - i) * offset_step

        # Extra Abstand nur für EtA+H2O+MeOH
        if s["name"] == "EtA-H_2O-MeOH":
            offset += TOP_EXTRA_GAP_STEPS * offset_step

        y_off = s["y"] + offset

        ax.plot(s["x"], y_off, color=s["color"], zorder=2)

        label_spectrum_left(
            ax,
            s["x"],
            y_off,
            s["description"],
            s["color"],
            x_target=x_label_target,
            y_text_offset=0,
        )

        # Merke EtA+H2O+MeOH (für gestrichelte Linien bis y=0)
        if s["name"] == "EtA-H_2O-MeOH":
            top_x = s["x"]
            top_y_off = y_off

        # Merke Irr 60 min (für Leader-Lines zu den Labels)
        if s["name"] == "Irr 60min":
            irr60_x = s["x"]
            irr60_y_off = y_off

        # y-range tracking
        y_min_local = float(np.min(y_off))
        y_max_local = float(np.max(y_off))
        plotted_min = y_min_local if plotted_min is None else min(plotted_min, y_min_local)
        plotted_max = y_max_local if plotted_max is None else max(plotted_max, y_max_local)

    # ----------------- Gestrichelte Linien: EtA+H2O+MeOH -> y=0 -----------------
    if top_x is not None and top_y_off is not None:
        draw_dashed_lines_from_top_to_zero(
            ax,
            top_x,
            top_y_off,
            ALL_PEAKS,
            ymin=DASHED_LINE_YMIN,
            color="black",
            lw=1.0,
        )
    else:
        print("Warnung: Oberes Spektrum (EtA-H_2O-MeOH) nicht gefunden – Linien nicht gezeichnet.")

    # ----------------- Manuelle Peak-Labels + Leader-Lines (vom Irr 60 min Spektrum) -----------------
    if irr60_x is not None and irr60_y_off is not None:
        add_manual_peak_labels_with_leaders(
            ax,
            PEAK_LABELS_MANUAL,
            irr60_x,
            irr60_y_off,
            color="black",
            fontsize=18,
            line_lw=1.0,
        )
    else:
        print("Warnung: Irr 60min nicht gefunden – Leader-Lines/Labels nicht gesetzt.")

    # ----------------- Achsen -----------------
    ax.set_xlim(X_LEFT, X_RIGHT)
    ax.margins(x=0)

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.tick_params(
        axis="both",
        which="both",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )

    # Tight y-limits:
    # - kleiner Abstand zum untersten Spektrum
    # - y=0 muss sichtbar sein (wegen gestrichelter Linien)
    y_bottom = min(DASHED_LINE_YMIN, plotted_min - Y_BOTTOM_PAD)
    y_top = plotted_max + 0.05
    ax.set_ylim(y_bottom, y_top)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
