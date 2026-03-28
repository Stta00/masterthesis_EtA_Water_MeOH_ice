from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from plot_helper_spectra import create_figure

DATA_DIR = Path(
    r"/home/tara/Dokumente/MA/2 - Data/Excel-files/Spektren_EtA_MeOH_H2O_TPD"
)

TEMPERATURES = [40, 80, 120, 160, 180, 190, 230, 270, 300]
FILENAME_TEMPLATE = "20240626_TPD_After_Irradiation_{temp}K_sample1.CSV"

X_MIN = 550
X_MAX = 3700


def load_CSV_spectrum(path, x_col=0, y_col=1):
    df = pd.read_csv(path, sep=";", decimal=",")
    x = df.iloc[:, x_col].astype(float).values
    y = df.iloc[:, y_col].astype(float).values
    return x, y


def normalize_spectrum(y):
    """Baseline correct + normalize spectrum to 0–1."""
    y = y - np.min(y)
    max_val = np.max(y)
    if max_val != 0:
        y = y / max_val
    return y


def main():
    fig, ax = create_figure()

    spectra = []

    for temp in TEMPERATURES:
        filename = FILENAME_TEMPLATE.format(temp=temp)
        filepath = DATA_DIR / filename

        if not filepath.exists():
            print(f"Warnung: Datei nicht gefunden: {filepath}")
            continue

        x, y = load_CSV_spectrum(filepath)

        mask = (x >= X_MIN) & (x <= X_MAX)
        x = x[mask]
        y = y[mask]

        if x.size == 0:
            print(f"Warnung: Keine Daten im Bereich {X_MIN}–{X_MAX} cm^-1 in {filepath}")
            continue

        # normalize spectrum
        y = normalize_spectrum(y)

        spectra.append((temp, x, y))

    if not spectra:
        print("Keine Spektren geladen – bitte Pfad, Dateinamen und Spalten prüfen.")
        return

    spectra.sort(key=lambda item: item[0])

    # constant offset for normalized spectra
    offset_step = 1.5


    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(vmin=min(TEMPERATURES), vmax=max(TEMPERATURES))

    temp_to_data = {}

    for i, (temp, x, y) in enumerate(spectra):
        offset = i * offset_step
        color = cmap(norm(temp))
        ax.plot(x, y + offset, color=color, lw=1.5)

        temp_to_data[temp] = {"x": x, "y": y, "offset": offset}

    ax.set_xlim(X_MAX, X_MIN)

    # --- COLORBAR ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Temperature / K", fontsize=18)
    cbar.set_ticks(TEMPERATURES)
    cbar.set_ticklabels([f"{T}" for T in TEMPERATURES])
    cbar.ax.tick_params(labelsize=18)

    # --- PEAK ANNOTATION ---
    def annotate_peak(temp, peak_x, label_num):
        if temp not in temp_to_data:
            print(f"Peak {label_num}: Temperatur {temp} K nicht geladen.")
            return

        data = temp_to_data[temp]
        x_arr = data["x"]
        y_arr = data["y"]
        offset = data["offset"]

        idx = int(np.argmin(np.abs(x_arr - peak_x)))
        x_val = x_arr[idx]
        y_val = y_arr[idx] + offset

        text_offset_y = 10
        text_offset_x = 0

        if label_num == 12:
            text_offset_y = 15
            text_offset_x = 18

        if label_num == 2:
            text_offset_y = 6
            text_offset_x = 10

        ax.annotate(
            str(label_num),
            xy=(x_val, y_val),
            xytext=(text_offset_x, text_offset_y),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=18,
            color="black",
            arrowprops=dict(
                arrowstyle="-",
                color="black",
                lw=1,
            ),
        )

    peak_specs = [
        (1, 2964, 120),
        (2, 2958, 190),
        (3, 1355, 80),
        (4, 1481, 120),
        (5, 1376, 120),
        (6, 1466, 160),
        (7, 1714, 180),
        (8, 1178, 180),
        (9, 1084, 80),
        (10, 984, 80),
        (11, 928, 180),
        (12, 906, 180),
        (13, 1080, 180),
    ]

    for num, x_peak, temp in peak_specs:
        annotate_peak(temp, x_peak, num)

    # --- save as PDF ---
    output_dir = Path(
        r"/home/tara/PyCharmMiscProject/MA_MPE/output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "TPD_spectra.png"
    fig.savefig(output_path, format="png", bbox_inches="tight")

    plt.tight_layout()
    #plt.show()


if __name__ == "__main__":
    main()