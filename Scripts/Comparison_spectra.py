from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_helper_spectra import create_figure


# ---------------------------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\taras\OneDrive\Dokumente\MSc\MA\2 - Data\Excel-files\Comparison_All"
)

SPECTRA_CONFIG = [
    {
        "name": "Ethanolamine",
        "description": "Ethanolamine",
        "file": "20240626_Deposition_ZnSe_20K_sample1.csv",
        "peaks": [
            {"wavenumber": 3350, "label": "1"},
            {"wavenumber": 3284, "label": "2"},
            {"wavenumber": 3198, "label": "3"},
            {"wavenumber": 2966, "label": "4","text_offset": (-3,15)},
            {"wavenumber": 2928, "label": "5"},
            {"wavenumber": 2861, "label": "6"},
            {"wavenumber": 2777, "label": "7"},
            {"wavenumber": 2722, "label": "8"},
            {"wavenumber": 1607, "label": "9"},
            {"wavenumber": 1457, "label": "10", "text_offset":(-15, 5)},
            {"wavenumber": 1398, "label": "11", "text_offset":(-20, 37)},
            {"wavenumber": 1377, "label": "12", "text_offset":(-10, 50)},
            {"wavenumber": 1362, "label": "13", "text_offset":(4, 10)},
            {"wavenumber": 1269, "label": "14", "text_offset":(-5, 44)},
            {"wavenumber": 1238, "label": "15", "text_offset":(3, 10)},
            {"wavenumber": 1175, "label": "16", "text_offset":(2, 25)},
            {"wavenumber": 1085, "label": "17", "text_offset":(-7, 17)},
            {"wavenumber": 1050, "label": "18", "text_offset":(-5, 30)},
            {"wavenumber": 1035, "label": "19", "text_offset":(13, 40)},
            {"wavenumber": 998, "label": "20", "text_offset":(15, 33)},
            {"wavenumber": 960, "label": "21", "text_offset":(30, 32)},
            {"wavenumber": 877, "label": "22", "text_offset":(25, 5)},
        ],
    },
    {
        "name": "H_2O",
        "description": "H₂O",
        "file": "20240626_Deposition_H2O_20K_sample1.csv",
        "peaks": [
            {"wavenumber": 3271, "label": "1", "text_offset":(0, 20)},
            {"wavenumber": 1670, "label": "2", "text_offset":(0, 20)},
            {"wavenumber": 748, "label": "3", "text_offset":(0, 20)},
        ],
    },
    {
        "name": "MeOH",
        "description": "CH$_3$OH",
        "file": "20240626_Deposition_Methanol_20K_sample1.csv",
        "peaks": [
            {"wavenumber": 3258, "label": "1", "text_offset":(0, 20)},
            {"wavenumber": 2958, "label": "2", "text_offset":(0, 15)},
            {"wavenumber": 2915, "label": "3", "text_offset":(0, 20)},
            {"wavenumber": 2828, "label": "4", "text_offset":(0, 20)},
            {"wavenumber": 1483, "label": "5", "text_offset":(-3, 20)},
            {"wavenumber": 1443, "label": "6", "text_offset":(3, 20)},
            {"wavenumber": 1127, "label": "7", "text_offset":(0, 20)},
            {"wavenumber": 1079, "label": "8", "text_offset":(0, 20)},
            {"wavenumber": 1026, "label": "9", "text_offset":(0, 20)},
            {"wavenumber": 758, "label": "10", "text_offset":(0, 20)},
        ],
    },
    {
        "name": "EtA-H_2O-MeOH",
        "description": "EtA + H₂O + CH$_3$OH",
        "file": "20240626_ice_Deposited_Etha_H2O_Methanol_20K_sample1.csv",
        "peaks": [],
    },
    {
        "name": "Choline-Chloride",
        "description": "Choline",
        "file": "20240627_CholineChloride_300K_sample1.csv",
        "peaks": [],
    },
]

X_MIN = 550
X_MAX = 3700

CSV_SEP = ";"
CSV_DECIMAL = ","

PEAK_TEXT_STYLE = {"fontsize": 18, "color": "black", "ha": "center", "va": "bottom"}
PEAK_TEXT_OFFSET = (0, 10)
PEAK_ARROW_STYLE = {"arrowstyle": "-", "color": "black", "lw": 1}

SPECTRA_COLORS = {
    "Ethanolamine": "#006400",
    "H_2O": "#00008B",
    "MeOH": "#FF8C00",
    "EtA-H_2O-MeOH": "#c243a2",
    "Choline-Chloride": "#40004d",
}

SCALING_FACTORS = {
    "MeOH": 15,
    "H_2O": 10,
    "Choline-Chloride": 5,
}


# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------------------------------

def load_csv_spectrum(path, x_col=0, y_col=1):
    df = pd.read_csv(path, sep=CSV_SEP, decimal=CSV_DECIMAL)
    x = df.iloc[:, x_col].astype(float).values
    y = df.iloc[:, y_col].astype(float).values
    return x, y


def get_peak_coordinates(x_data, y_data, wavenumber):
    idx = int(np.abs(x_data - wavenumber).argmin())
    return float(x_data[idx]), float(y_data[idx])


def annotate_peaks(ax, x_data, y_data, peaks, text_defaults, text_offset, arrow_defaults):
    for peak in peaks:
        x_p, y_p = get_peak_coordinates(x_data, y_data, peak["wavenumber"])

        # individuelles Offset pro Peak erlauben
        offset = peak.get("text_offset", text_offset)

        ax.annotate(
            peak["label"],
            xy=(x_p, y_p),
            xytext=offset,
            textcoords="offset points",
            arrowprops=arrow_defaults,
            **text_defaults,
        )


def label_spectrum(ax, x_data, y_data, text, color, x_target, y_text_offset=8):
    idx = int(np.abs(x_data - x_target).argmin())
    ax.annotate(
        text,
        xy=(x_data[idx], y_data[idx]),
        xytext=(0, y_text_offset),
        textcoords="offset points",
        color=color,
        fontsize=18,
        ha="center",
        va="center",
        clip_on=False,
    )


# ---------------------------------------------------------------------------
# HAUPTFUNKTION
# ---------------------------------------------------------------------------

def main():
    fig, ax = create_figure()

    spectra_data = []
    max_range = 0.0

    for spec in SPECTRA_CONFIG:
        x, y = load_csv_spectrum(DATA_DIR / spec["file"])
        mask = (x >= X_MIN) & (x <= X_MAX)
        x, y = x[mask], y[mask]

        factor = SCALING_FACTORS.get(spec["name"], 1.0)
        y_scaled = y * factor

        spectra_data.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "x": x,
                "y": y_scaled,
                "peaks": spec["peaks"],
            }
        )

        max_range = max(max_range, y_scaled.max() - y_scaled.min())

    offset_step = max_range * 1.2
    x_center = 0.5 * (X_MIN + X_MAX)  # <<< FIX

    LABEL_X = {
        "Ethanolamine": 2131.5,
        "H_2O": 2330.8,
        "MeOH": 2278.5,
        "EtA-H_2O-MeOH": 2030,
        "Choline-Chloride": 2252.2,
    }

    LABEL_Y_OFFSET = {
        "Ethanolamine": 15,
        "H_2O": 10,
        "MeOH": 10,
        "EtA-H_2O-MeOH": 21,
        "Choline-Chloride": 10,
    }

    for i, spec in enumerate(spectra_data):
        y_offset = spec["y"] + i * offset_step
        color = SPECTRA_COLORS.get(spec["name"], "black")

        ax.plot(spec["x"], y_offset, color=color)

        label_spectrum(
            ax,
            spec["x"],
            y_offset,
            spec["description"],
            color,
            x_target=LABEL_X.get(spec["name"], x_center),
            y_text_offset=LABEL_Y_OFFSET.get(spec["name"], 8),
        )

        annotate_peaks(
            ax,
            spec["x"],
            y_offset,
            spec["peaks"],
            PEAK_TEXT_STYLE,
            PEAK_TEXT_OFFSET,
            PEAK_ARROW_STYLE,
        )

    ax.set_xlim(X_MAX, X_MIN)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
