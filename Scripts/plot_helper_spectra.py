# plot_helper_spectra.py
import matplotlib.pyplot as plt

# Globale Plot-Einstellungen (TPD-Standard)
FIGSIZE = (11, 8)  # Breite, Höhe in Zoll
DPI = 400          # Auflösung

def create_figure(
    xlabel=r"Wavenumber / cm$^{-1}$",
    ylabel="Absorbance (a.u.)",
    labelsize=24,
    ticksize=20,
):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.tick_params(axis="both", which="major", labelsize=ticksize)
    return fig, ax

