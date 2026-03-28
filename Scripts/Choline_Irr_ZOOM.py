from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_helper_spectra import create_figure


# ---------------------------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    r"/home/tara/Dokumente/MA/2 - Data/Excel-files/Choline_Irr"
)
OUTPUT_DIR = Path("/home/tara/PyCharmMiscProject/MA_MPE/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


X_MIN = 580
X_MAX = 1670

CSV_SEP = ";"
CSV_DECIMAL = ","


# --- Peak Styling (identisch) ---
PEAK_TEXT_STYLE = {"fontsize": 18, "color": "black", "ha": "center", "va": "bottom"}
PEAK_TEXT_OFFSET = (0, 10)
PEAK_ARROW_STYLE = {"arrowstyle": "-", "color": "black", "lw": 1}

# --- Farben/Look (identisch) ---
SPECTRA_CMAP = plt.get_cmap("viridis")




SPECTRA_CONFIG = [
    {
        "name": "sample1.1",
        "description": "Irr. 1min",
        "file": "20240628_Irradiation_Choline_H2O_20K_sample1.1.CSV",
        "factor": 1.0,
        "label_x": 3600,
        "label_offset": (5, -18),
        "peaks": [],
    },
    {
        "name": "sample1.2",
        "description": "Irr. 5 min",
        "file": "20240628_Irradiation_Choline_H2O_20K_sample1.2.CSV",
        "factor": 1.0,
        "label_x": 3600,
        "label_offset": (5, -20),
        "peaks": [
    {"wavenumber": 1270, "label": "26", "text_offset": (-15, 15)},
    {"wavenumber": 1248, "label": "27", "text_offset": (15, 15)},
    {"wavenumber": 967, "label": "37"},
    {"wavenumber": 1332, "label": "*", "text_offset": (0, -30)},
    {"wavenumber": 1087, "label": "*", "text_offset": (0, -35)},
    {"wavenumber": 1151, "label": "*", "text_offset": (0, -45)},
    {"wavenumber": 1014, "label": "*", "text_offset": (0, -40)},
    {"wavenumber": 962, "label": "*", "text_offset": (0, -40)},
    {"wavenumber": 953, "label": "*", "text_offset": (7, -25)},
    {"wavenumber": 634, "label": "*", "text_offset": (0, -35)}
        ],
    },
    {
        "name": "sample1.3",
        "description": "Irr. 15min",
        "file": "20240628_Irradiation_Choline_H2O_20K_sample1.3.CSV",
        "factor": 1.0,
        "label_x": 3600,
        "label_offset": (5, -20),
        "peaks": [
    {"wavenumber": 1481, "label": "*", "text_offset": (0, -40)},   # choline
    {"wavenumber": 1350, "label": "*", "text_offset": (0, -20)},   # choline
  # choline
    {"wavenumber": 1460, "label": "*", "text_offset": (0, -30)},
    {"wavenumber": 1028, "label": "35"},
    {"wavenumber": 1134, "label": "32", "text_offset": (-5, 20)},
    {"wavenumber": 892, "label": "*", "text_offset": (0, -30)},
    {"wavenumber": 865, "label": "*", "text_offset": (0, -55)},
    {"wavenumber": 845, "label": "*", "text_offset": (0, -30)},
                  ]  # choline],
    },
    {
        "name": "sample1.4",
        "description": "Irr. 30 min",
        "file": "20240628_Irradiation_Choline_H2O_20K_sample1.4.CSV",
        "factor": 1.0,
        "label_x": 3600,
        "label_offset": (5, -20),
        "peaks": [
    {"wavenumber": 1356, "label": "17", "text_offset": (8, 45)},
    {"wavenumber": 1379, "label": "18", "text_offset": (-6, 64)},
    {"wavenumber": 1439, "label": "19", "text_offset": (-20, 15)},
    {"wavenumber": 1430, "label": "20", "text_offset": (-7, 30)},
    {"wavenumber": 1417, "label": "21", "text_offset": (4, 45)},
    {"wavenumber": 1336, "label": "23", "text_offset": (25, 25)},
    {"wavenumber": 1298, "label": "25", "text_offset": (33, 5)},
    {"wavenumber": 903, "label": "38", "text_offset": (0, 10)},
    {"wavenumber": 655, "label": "39", "text_offset": (0, 10)},
    {"wavenumber": 777, "label": "40", "text_offset": (0, 10)},
    {"wavenumber": 610, "label": "22", "text_offset": (0, 15)},
    {"wavenumber": 1143, "label": "*", "text_offset": (0, -40)},
    {"wavenumber": 1056, "label": "*", "text_offset": (0, -40)},
    ],
    }, ## ICH HAB PEAK NUMMER 22 RAUSGENOMMEN DIESER FEHLT JETZT!!
    {
        "name": "sample1.5",
        "description": "Irr. 60 min",
        "file": "20240628_Irradiation_Choline_H2O_20K_sample1.5.CSV",
        "factor": 1.0,
        "label_x": 3600,
        "label_offset": (5, -20),
        "peaks": [
            {"wavenumber": 1327, "label": "24", "text_offset": (-25, 25)},
            {"wavenumber": 1227, "label": "28", "text_offset": (-20, 10)},
            {"wavenumber": 1219, "label": "29", "text_offset": (-15, 30)},
            {"wavenumber": 1189, "label": "30", "text_offset": (-7, 42)},
            {"wavenumber": 1160, "label": "31", "text_offset": (5, 20)},
            {"wavenumber": 1107, "label": "33", "text_offset": (10, 40)},
            {"wavenumber": 1074, "label": "34", "text_offset": (14, 20)},
            {"wavenumber": 1001, "label": "36", "text_offset": (20, 35)},
        ],
    },
    ]




# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN (identisch)
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


def label_spectrum(ax, x_data, y_data, text, color, x_target=3600, text_offset=(-4, -15)):
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
# HAUPTFUNKTION (identisch)
# ---------------------------------------------------------------------------

def main():
    if not SPECTRA_CONFIG:
        print("Keine Spektren konfiguriert – bitte SPECTRA_CONFIG ausfüllen.")
        return

    fig, ax = create_figure()

    spectra_data = []
    max_range = 0.0

    for spec in SPECTRA_CONFIG:
        name = spec.get("name", "unbenannt")
        description = spec.get("description", name)
        filename = spec.get("file")

        if not filename:
            print(f"Warnung: Kein Dateiname für {name} gesetzt.")
            continue

        # identisch: factor * 15
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
                "description": description,
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

    offset_step = max_range * 1.2 if max_range > 0 else 1.0

    for i, spec in enumerate(spectra_data):
        x = spec["x"]
        y_scaled = spec["y"]
        offset = i * offset_step
        y_offset = y_scaled + offset

        color = colors[i]
        ax.plot(x, y_offset, color=color)

        label_spectrum(
            ax,
            x,
            y_offset,
            spec["description"],
            color,
            x_target=spec.get("label_x", 3600),
            text_offset=spec.get("label_offset", (-4, -10)),
        )

        annotate_peaks(
            ax,
            x,
            y_offset,
            spec["peaks"],
            PEAK_TEXT_STYLE,
            PEAK_TEXT_OFFSET,
            PEAK_ARROW_STYLE,
        )

    ax.margins(x=0)
    ax.set_xlim(X_MAX, X_MIN)
    ax.set_ylim(-0.05, 0.75)

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="both", top=True, right=True, labeltop=False, labelright=False)

    plt.tight_layout()

    out_base = OUTPUT_DIR / "Choline_Irr_Zoom"
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
