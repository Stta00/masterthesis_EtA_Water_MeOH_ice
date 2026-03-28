"""
plot_helper.py

- Einheitlicher Matplotlib-Style
- Molekül-Insets über RDKit (2D-Rendering)
- Optional: Molekül als PNG auf die Platte schreiben (save_molecule_png)

Hinweis:
- Für ein PDF/PNG in einem normalen Python-Skript ist RDKit-2D deutlich robuster
  als py3Dmol, das v.a. auf Jupyter-Notebooks ausgelegt ist.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -------------------------------------------------------------
# RDKit-Import (für Molekülobjekte & 2D-Rendering)
# -------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from PIL import Image
    _HAS_RDKIT = True
except ImportError:
    Chem = AllChem = Draw = Image = None
    _HAS_RDKIT = False
    warnings.warn(
        "RDKit (und/oder Pillow) ist nicht installiert. "
        "Molekül-Rendering ist deaktiviert.",
        RuntimeWarning
    )


# ============================================================
# 1. Allgemeine Plot-Einstellungen
# ============================================================

def setup_matplotlib():
    """
    Setzt einen einheitlichen Matplotlib-Style für alle Plots.
    Top- und Right-Spines werden NICHT global deaktiviert.
    """
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": False,
        "savefig.dpi": 300,
        "figure.autolayout": True,
    })


# Optionales Farbschema
COLORS = {
    "EtA_sim":   "#98FB98",  # pale green
    "EtA_data":  "#006400",  # dark green
    "H2O_sim":   "#ADD8E6",  # light blue
    "H2O_data":  "#00008B",  # dark blue
    "MeOH_sim":  "#FFD580",  # light orange
    "MeOH_data": "#FF8C00",  # dark orange
}


# ============================================================
# 2. RDKit-Helfer
# ============================================================

def _check_rdkit():
    if not _HAS_RDKIT:
        raise RuntimeError(
            "RDKit/Pillow ist nicht verfügbar. "
            "Installiere sie z.B. mit 'conda install -c conda-forge rdkit pillow'."
        )


def _mol_from_smiles(smiles: str):
    """Erzeugt ein RDKit-Molekülobjekt aus einem SMILES-String."""
    _check_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES konnte nicht gelesen werden: {smiles}")
    mol = Chem.AddHs(mol)
    return mol


def mol_to_image_array(smiles: str, size=(300, 300)) -> np.ndarray:
    """
    Erzeugt ein 2D-Molekülbild (ähnlich Ball-and-Stick) als numpy-Array (RGB)
    aus einem SMILES-String mithilfe von RDKit.
    """
    mol = _mol_from_smiles(smiles)
    # 2D-Koordinaten berechnen
    AllChem.Compute2DCoords(mol)

    # RDKit-Zeichnung – standardmäßig bereits mit Atomen + Bindungen
    img = Draw.MolToImage(mol, size=size)
    return np.array(img)


def save_molecule_png(smiles: str, filename: str, size=(600, 600)):
    """
    Speichert ein Molekülbild als PNG-Datei. Entspricht deinem Wunsch,
    einen „internen Screenshot“ zu erzeugen und als echtes PNG zu nutzen.

    Parameter
    ---------
    smiles : str
        SMILES-String des Moleküls.
    filename : str
        Dateipfad (.png), z.B. 'ethanolamine.png'
    size : tuple
        Pixelgröße (Breite, Höhe).
    """
    _check_rdkit()
    arr = mol_to_image_array(smiles, size=size)
    img = Image.fromarray(arr)
    # Ordner anlegen falls nötig
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    img.save(filename)


# ============================================================
# 3. Molekülstruktur in Matplotlib-Inset einfügen
# ============================================================

def add_molecule_inset_from_smiles(
    ax,
    smiles: str,
    loc: str = "upper right",
    width: str = "25%",
    height: str = "25%",
    border: bool = False,
    size_px=(300, 300),
):
    """
    Fügt in eine bestehende Achse 'ax' ein Molekülbild als Inset ein,
    das aus einem SMILES-String mit RDKit erzeugt wird (2D).

    Parameter
    ---------
    ax : matplotlib.axes.Axes
        Achse, in die das Inset soll.
    smiles : str
        SMILES-String des Moleküls.
        Beispiele:
          - Ethanolamine: "OCCN"
          - Wasser:       "O"
          - Methanol:     "CO"
    loc : str
        Position des Inset-Fensters (z.B. "upper right").
    width, height : str
        Größe des Insets in Prozent der Achse ("25%", "30%", ...).
    border : bool
        Wenn False, werden Achsen-Ränder ausgeblendet.
    size_px : tuple
        Pixelgröße des generierten Molekülbildes.
    """
    inset_ax = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        borderpad=0.8,
    )

    if not _HAS_RDKIT:
        # Fallback: Text-Platzhalter, damit das Skript trotz fehlendem RDKit läuft
        inset_ax.text(
            0.5, 0.5,
            f"SMILES:\n{smiles}\n(RDKit fehlt)",
            ha="center",
            va="center",
            fontsize=8,
            wrap=True,
        )
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_visible(False)
        return inset_ax

    # RDKit verfügbar → Molekülbild erzeugen
    img_array = mol_to_image_array(smiles, size=size_px)
    inset_ax.imshow(img_array)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    if not border:
        for spine in inset_ax.spines.values():
            spine.set_visible(False)

    return inset_ax
