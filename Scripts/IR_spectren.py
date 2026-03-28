from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_helper_spectra import create_figure


# ---------------------------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\taras\OneDrive\Dokumente\MSc\MA\2 - Data\Excel-files\MeOH_EtA_Irr"
)

SPECTRA_CONFIG = [
    {
        "name": "Irr 1min",
        "description": "Irr. 1 min",
        "file": "20240626_Irradiation_20K_sample2.csv",
        "factor": 0.8,
        "label_x": 3600,            # anchor near 3600 cm^-1
        "label_offset": (0, -20),  # move left 30, down 5
        "peaks": [
            {"wavenumber": 3100, "label": "1"},
            {"wavenumber": 2964, "label": "2"},
            {"wavenumber": 2848, "label": "4"},
            {"wavenumber": 2787, "label": "5"},
            {"wavenumber": 2340, "label": "6"},
            {"wavenumber": 2140, "label": "8"},
            {"wavenumber": 1720, "label": "9", "text_offset": (-5, 20)},
            {"wavenumber": 1660, "label": "10"},
            {"wavenumber": 1502, "label": "11", "text_offset": (-7, 15)},
            {"wavenumber": 1380, "label": "12", "text_offset": (-4, 10)},
            {"wavenumber": 1352, "label": "13", "text_offset": (6, 25)},
            {"wavenumber": 1302, "label": "14", "text_offset": (14, 40)},
            {"wavenumber": 1210, "label": "15"},
            {"wavenumber": 1067, "label": "17", "text_offset": (10, 23)},
            {"wavenumber": 990,  "label": "18", "text_offset": (15, 37)},
            {"wavenumber": 895,  "label": "19", "text_offset": (8, 10)},
            {"wavenumber": 3337, "label": "24", "text_offset": (0, 20)},
            {"wavenumber": 1095, "label": "25", "text_offset": (-2, 10)},
            {"wavenumber": 1437, "label": "27", "text_offset": (-8, 40)},
        ],
    },
    {
        "name": "Irr 5min",
        "description": "Irr. 5 min",
        "file": "20240626_Irradiation_20K_sample3.csv",
        "factor": 1.2,
        "label_x": 3600,  # anchor near 3600 cm^-1
        "label_offset": (0, -28),  # move left 30, down 5
        "peaks": [
            {"wavenumber": 2890, "label": "3"},
            {"wavenumber": 1151, "label": "16"},
            {"wavenumber": 850,  "label": "20"},
            {"wavenumber": 1123,  "label": "23", "text_offset": (15, 20)},
            {"wavenumber": 783,  "label": "21", "text_offset": (5, 20)},
            {"wavenumber": 3009, "label": "26"},
        ],
    },
    {
        "name": "Irr 15min",
        "description": "Irr. 15 min",
        "file": "20240626_Irradiation_20K_sample4.csv",
        "factor": 1.4,
        "label_x": 3600,  # anchor near 3600 cm^-1
        "label_offset": (0, -30),  # move left 30, down 5
        "peaks": [
            {"wavenumber": 2165, "label": "7"},
            {"wavenumber": 655,  "label": "22"},
        ],
    },
    {
        "name": "Irr 30min",
        "description": "Irr. 30 min",
        "file": "20240626_Irradiation_20K_sample5.csv",
        "factor": 1.6,
        "label_x": 3600,  # anchor near 3600 cm^-1
        "label_offset": (0, -40),  # move left 30, down 5
        "peaks": [],
    },
    {
        "name": "Irr 60min",
        "description": "Irr. 60 min",
        "file": "20240626_Irradiation_20K_sample6.csv",
        "factor": 1.8,
        "label_x": 3600,  # anchor near 3600 cm^-1
        "label_offset": (0, -55),  # move left 30, down 5
        "peaks": [],
    },
]

X_MIN = 550
X_MAX = 3700

CSV_SEP = ";"
CSV_DECIMAL = ","

# --- TPD-like Peak Styling ---
PEAK_TEXT_STYLE = {"fontsize": 18, "color": "black", "ha": "center", "va": "bottom"}
PEAK_TEXT_OFFSET = (0, 10)
PEAK_ARROW_STYLE = {
    "arrowstyle": "-",
    "color": "black",
    "lw": 1,
}

SPECTRA_CMAP = plt.get_cmap("magma")


# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------------------------------

def load_csv_spectrum(path, x_col=0, y_col=1):
    df = pd.read_csv(path, sep=CSV_SEP, decimal=CSV_DECIMAL)
    x = df.iloc[:, x_col].astype(float).values
    y = df.iloc[:, y_col].astype(float).values
    return x, y


def get_peak_coordinates(x_data, y_data, wavenumber):
    if x_data.size == 0:
        return None
    idx = int(np.abs(x_data - wavenumber).argmin())
    return float(x_data[idx]), float(y_data[idx])


def annotate_peaks(ax, x_data, y_data, peaks, text_defaults, text_offset, arrow_defaults):
    if not peaks:
        return

    for idx, peak in enumerate(peaks, start=1):
        wavenumber = peak.get("wavenumber")
        if wavenumber is None:
            continue

        coords = get_peak_coordinates(x_data, y_data, wavenumber)
        if coords is None:
            continue

        peak_x, peak_y = coords

        text_style = {**text_defaults, **peak.get("text_style", {})}
        arrow_style = {**arrow_defaults, **peak.get("arrow_style", {})}

        label = str(peak.get("label") or idx)
        offset = peak.get("text_offset", text_offset)
        if not isinstance(offset, (tuple, list)) or len(offset) != 2:
            offset = text_offset

        ax.annotate(
            label,
            xy=(peak_x, peak_y),
            xytext=tuple(offset),
            textcoords="offset points",
            arrowprops=arrow_style,
            **text_style,
        )


def label_spectrum(
    ax,
    x_data,
    y_data,
    text,
    color,
    x_target=3600,
    text_offset=(-4, -15),
):
    if x_data.size == 0:
        return

    idx = int(np.abs(x_data - x_target).argmin())

    ax.annotate(
        text,
        xy=(x_data[idx], y_data[idx]),
        xytext=text_offset,
        textcoords="offset points",
        color=color,
        fontsize=18,
        va="center",
        ha="left",
        clip_on=False,
    )


# ---------------------------------------------------------------------------
# HAUPTFUNKTION
# ---------------------------------------------------------------------------

def main():
    if not SPECTRA_CONFIG:
        print("Keine Spektren konfiguriert – bitte SPECTRA_CONFIG ausfüllen.")
        return

    # gleiche Maße/Fonts wie TPD über den Helper
    fig, ax = create_figure()

    spectra_data = []
    max_range = 0.0

    # ----------------- CSVs einlesen & vorbereiten -----------------
    for spec in SPECTRA_CONFIG:
        name = spec.get("name", "unbenannt")
        description = spec.get("description", name)
        filename = spec.get("file")

        if not filename:
            print(f"Warnung: Kein Dateiname für {name} gesetzt.")
            continue

        # Dein ursprünglicher Skalierungsansatz bleibt erhalten
        factor = spec.get("factor", 1.0) * 15

        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Warnung: Datei nicht gefunden: {filepath}")
            continue

        x, y = load_csv_spectrum(filepath)

        if X_MIN is not None and X_MAX is not None:
            mask = (x >= X_MIN) & (x <= X_MAX)
            x = x[mask]
            y = y[mask]

        if x.size == 0:
            print(f"Warnung: Keine Daten im Bereich {X_MIN}–{X_MAX} cm^-1 in {filepath}")
            continue

        y_scaled = y * factor

        spectra_data.append(
            {
                "name": name,
                "description": description,  # <<< wichtig für sichtbare Beschriftungen
                "x": x,
                "y": y_scaled,
                "factor": factor,
                "peaks": spec.get("peaks", []),
                "label_x": spec.get("label_x", 3600),
                "label_offset": spec.get("label_offset", (-4, -15)),
            }
        )

        y_range = float(y_scaled.max() - y_scaled.min())
        max_range = max(max_range, y_range)

    if not spectra_data:
        print("Keine Spektren geladen – bitte Pfad, Dateinamen und Spalten prüfen.")
        return

    colors = SPECTRA_CMAP(np.linspace(0.75, 0.05, len(spectra_data)))

    # ----------------- y-Offset bestimmen -----------------
    offset_step = max_range * 1.2 if max_range > 0 else 1.0

    # ----------------- Plotten mit y-Offset -----------------
    for i, spec in enumerate(spectra_data):
        x = spec["x"]
        y_scaled = spec["y"]
        offset = i * offset_step
        y_offset = y_scaled + offset

        color = colors[i]
        ax.plot(x, y_offset, color=color)

        # Beschriftung in gleicher Farbe pro Spektrum
        # Falls 3600 bei dir ungünstig ist, probier z.B. 700:
        # label_spectrum(ax, x, y_offset, spec["description"], color, x_target=700)
        label_spectrum(
            ax,
            x,
            y_offset,
            spec["description"],
            color,
            x_target=spec.get("label_x", 3600),
            text_offset=spec.get("label_offset", (-4, -15)),
        )

        # Peak-Annotation im TPD-Stil
        annotate_peaks(
            ax,
            x,
            y_offset,
            spec["peaks"],
            PEAK_TEXT_STYLE,
            PEAK_TEXT_OFFSET,
            PEAK_ARROW_STYLE,
        )

    # IR-typisch invertierte x-Achse
    ax.margins(x=0)
    ax.set_xlim(X_MAX, X_MIN)

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

