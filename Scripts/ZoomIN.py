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

# gewünschter Bereich: 970–820 cm^-1
X_LEFT = 900
X_RIGHT = 830

CSV_SEP = ";"
CSV_DECIMAL = ","

# gestrichelte Linien
VLINES_RED = [895, 850]
VLINES_CHOLINE = [893, 865]
VLINES_ETA = [877]


# Skalierungen (Anforderung)
IRR_FACTOR = 50
CHOLINE_FACTOR = 5

# Farben (deine festgelegten Farben bleiben erhalten)
SPECTRA_COLORS_FIXED = {
    "Choline-Chloride": "#40004d",   # lila
    "EtA-H_2O-MeOH": "#c243a2",      # pink
    "Ethanolamine": "#006400",       # grün
    # Irradiation-Farben werden unten festgelegt (stabil im Skript)
}

# ---------------------------------------------------------------------------
# SPEKTREN-KONFIGURATION
# ---------------------------------------------------------------------------

STATIC_SPECTRA = [
    {
        "name": "Choline-Chloride",
        "description": "Choline",
        "file": "20240627_CholineChloride_300K_sample1.csv",
        "dir": DATA_DIR_COMPARISON,
        "scale": CHOLINE_FACTOR,
    },
    {
        "name": "EtA-H_2O-MeOH",
        "description": "EtA + H₂O + MeOH",
        "file": "20240626_ice_Deposited_Etha_H2O_Methanol_20K_sample1.csv",
        "dir": DATA_DIR_COMPARISON,
        "scale": 1.0,
    },
    {
        "name": "Ethanolamine",
        "description": "Ethanolamine",
        "file": "20240626_Deposition_ZnSe_20K_sample1.csv",
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

# gewünschte Reihenfolge von oben nach unten:
ORDERED_SPECTRA = STATIC_SPECTRA + IRR_SPECTRA


# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------------------------------

def load_csv_spectrum(path: Path, x_col=0, y_col=1):
    df = pd.read_csv(path, sep=CSV_SEP, decimal=CSV_DECIMAL)
    x = df.iloc[:, x_col].astype(float).values
    y = df.iloc[:, y_col].astype(float).values
    return x, y


def label_spectrum_right(ax, x_data, y_data, text, color, x_target, y_text_offset=0):
    """
    Label rechts platzieren (bei invertierter IR-Achse ist 'rechts' niedrige Wellenzahl).
    ha='right' => Text endet am Ankerpunkt und läuft nach links in den Plot hinein.
    """
    idx = int(np.abs(x_data - x_target).argmin())
    ax.annotate(
        text,
        xy=(x_data[idx], y_data[idx]),
        xytext=(-0.2, y_text_offset),
        textcoords="offset points",
        color=color,
        fontsize=18,
        ha="right",
        va="center",
        clip_on=False,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    fig, ax = create_figure()

    # Irradiation-Farben "fest" im Skript definieren (stabil, reproduzierbar)
    irr_names = [s["name"] for s in IRR_SPECTRA]
    cmap = plt.get_cmap("magma")
    irr_cols = cmap(np.linspace(0.05, 0.75, len(irr_names)))
    irr_color_map = {name: irr_cols[i] for i, name in enumerate(irr_names)}

    # kombiniertes Color-Mapping
    def get_color(spec_name: str):
        if spec_name in SPECTRA_COLORS_FIXED:
            return SPECTRA_COLORS_FIXED[spec_name]
        if spec_name in irr_color_map:
            return irr_color_map[spec_name]
        return "black"

    # ----------------- Daten laden (nur 970–820 cm^-1) -----------------
    spectra_data = []
    max_range = 0.0

    for spec in ORDERED_SPECTRA:
        filepath = spec["dir"] / spec["file"]
        if not filepath.exists():
            print(f"Warnung: Datei nicht gefunden: {filepath}")
            continue

        x, y = load_csv_spectrum(filepath)

        # Fenster 820..970 (unabhängig von Plot-Inversion)
        lo, hi = min(X_RIGHT, X_LEFT), max(X_RIGHT, X_LEFT)
        mask = (x >= lo) & (x <= hi)
        x = x[mask]
        y = y[mask]

        if x.size == 0:
            print(f"Warnung: Keine Daten im Bereich {X_LEFT}–{X_RIGHT} cm^-1 in {filepath}")
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

    # ----------------- Offset berechnen -----------------
    offset_step = max_range * 1.4 if max_range > 0 else 0.5

    # ----------------- Plotten: oben -> unten -----------------
    # Wir geben dem ersten (oben) den größten Offset.
    n = len(spectra_data)

    # Labels rechts: bei invertierter Achse ist rechts ~ X_RIGHT
    x_label_target = X_RIGHT + 2  # leicht innerhalb, damit der Punkt sicher im Fenster ist

    for i, spec in enumerate(spectra_data):
        # i=0 soll oben sein => größter Offset
        offset = (n - 1 - i) * offset_step
        y_off = spec["y"] + offset

        ax.plot(spec["x"], y_off, color=spec["color"])

        # Beschriftung rechts
        label_spectrum_right(
            ax,
            spec["x"],
            y_off,
            spec["description"],
            spec["color"],
            x_target=x_label_target,
            y_text_offset=10,
        )

    # ----------------- Vertikale gestrichelte Linien -----------------
    # red lines
    for v in VLINES_RED:
        ax.axvline(v, linestyle="--", linewidth=1.2, color="red", alpha=0.8)

    # choline-colored lines
    for v in VLINES_CHOLINE:
        ax.axvline(
            v,
            linestyle="--",
            linewidth=1.2,
            color=SPECTRA_COLORS_FIXED["Choline-Chloride"],
            alpha=0.8,
        )

    # ethanolamine-colored line
    for v in VLINES_ETA:
        ax.axvline(
            v,
            linestyle="--",
            linewidth=1.2,
            color=SPECTRA_COLORS_FIXED["Ethanolamine"],
            alpha=0.8,
        )

    # ----------------- Achsen -----------------
    # IR-typisch invertiert: 970 links, 820 rechts
    ax.set_xlim(X_LEFT, X_RIGHT)
    # IR-typisch invertierte x-Achse
    ax.margins(x=0)
    # ax.set_xlim(X_MAX, X_MIN)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True
    )

    # Ticks on all sides
    ax.tick_params(
        axis="both",
        which="both",
        top=True,
        right=True,
        labeltop=False,
        labelright=False
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
