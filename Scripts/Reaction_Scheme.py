import os
from typing import Dict, Tuple, Any, Optional

try:
    from PIL import Image
except ImportError:  # Pillow ist optional, nur fürs Zuschneiden nötig
    Image = None

import plotly.io as pio
import plotly.graph_objects as go  # for go.Figure / go.Sankey

# ===== Plot-/Export-Konfiguration =====
FIG_LAYOUT: Dict[str, Any] = {
    "width": 1500,
    "height": 1000,
    "margin": dict(l=17, r=20, t=0, b=160),  # mehr Platz rechts für Achsenbeschriftungen
    "font": dict(size=24),
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
}

EXPORT_CONFIG: Dict[str, Any] = {
    "pdf_name": "Reaction_System.pdf",
    "png_name": "Reactiontest.png",
    "scale": 10,  # höhere Auflösung für PNG
}

CROP_SETTINGS: Dict[str, Any] = {
    "enabled": False,  # bei Bedarf auf True setzen
    # (left, upper, right, lower) in Pixeln relativ zum gespeicherten PNG
    # Beispiel: (50, 20, 1050, 620)
    "box": (0, 0, 0, 0),
}

# -------- Hilfsfunktionen --------
def apply_layout(fig: go.Figure, layout_cfg: Dict[str, Any]) -> None:
    """Überträgt Layout-Optionen auf die Plotly-Figur."""
    fig.update_layout(**layout_cfg)


def crop_png(path: str, crop_cfg: Dict[str, Any]) -> None:
    """Schneidet das gespeicherte PNG mit Pillow zu, falls aktiviert."""
    if not crop_cfg.get("enabled"):
        return
    if Image is None:
        print("Pillow nicht installiert – Zuschneiden wird übersprungen.")
        return

    box: Tuple[int, int, int, int] = tuple(crop_cfg.get("box", (0, 0, 0, 0)))
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        print("CROP_SETTINGS.box ist ungültig – Zuschneiden wird übersprungen.")
        return

    with Image.open(path) as img:
        cropped = img.crop(box)
        cropped.save(path)
        print(f"PNG auf Box {box} zugeschnitten.")


def export_figure(fig: go.Figure, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, EXPORT_CONFIG["pdf_name"])
    png_path = os.path.join(out_dir, EXPORT_CONFIG["png_name"])

    pio.write_image(fig, pdf_path, scale=EXPORT_CONFIG["scale"])
    pio.write_image(fig, png_path, scale=EXPORT_CONFIG["scale"])

    crop_png(png_path, CROP_SETTINGS)
    print("Saved to:", out_dir)


# ===== Farb­schema =====
color_eta    = "#1B5E20"  # EtA (grün)
color_h2o    = "#0D47A1"  # WATER (blau)
color_meoh   = "#E65100"  # METHANOL / C-chemie (orange)

# eigene Farben für spezielle Knoten
color_pom      = "#6A1B9A"  # POM (lila)
color_Hnode    = "#FBC02D"  # H (gelb)
color_OHnode   = "#00ACC1"  # OH (türkis)
color_choline  = "#1B5E20"  # grün wie EtA


# =====Farben für reactionen ====
# ===== Highly distinct reaction colors (color-blind safe) =====

color_catalyst        = "#4D4D4D"  # dark neutral grey
color_hydrogenation   = "#F57D05"  # orange
color_dehydrogenation = "#02F5C9"  # sky blue
color_dissociation    = "#F51E05"  # vermillion / red-orange
color_radical_radical = "#599E03"  # bluish green
color_methylation     = "#C4B104"  # strong yellow
color_protonation     = "#6C09ED"  # purple / magenta
color_ionisation      = "#094FED"  # deep blue
color_acid_base       = "#FC039E"  # black (max contrast)



# ===== Nodes =====
labels = [
    "NH₂CH₂CH₂OH<br>EtA",
    "init. H₂O",
    "init. CH₃OH",
    "NH₂·",
    "CH₂CH₂OH·",
    "CH₂OH·",
    "NH₂CH₂CH₂·",
    "NH₂CH₂·",
    "H·",
    "OH·",
    "ETA·",
    "CH₃·",
    "CH₃O·",
    "CH₂OH·",
    "<b>NH₃</b>",
    "(CH₃)NHCH₂CH₂OH",
    "(CH₃)₂NCH₂CH₂OH",
    "<b>CO₂</b>",
    "<b>CO</b>",
    "<b>H₂CO</b>",
    "<b>POM</b>",
    "<b>HCOOH</b>",
    "<b>H₂O</b>", #22
    "<b>HCO·</b>",
    "<b>H₂O₂</b>",
    "NH₄⁺",
    "HNCO",
    "<b>OCN⁻</b>",
    "H₂O⁺",
    "H₃O⁺",
    "CH₃⁺", # 30
    "(CH₃)NH",
    "(CH₃)₂N",
    "<b>[(CH₃)₃NCH₂CH₂OH]⁺-rel.</b>",
    "CH₃COH",
    "(CH₃)₃N⁺",
    "HOCO",
    "<b>CH₃CHO</b>",
    "<b>CH₃OH</b>", #38
    "C",
    "HCCO", #40
    "[(CH₃)₂NCH₂CH₂OH]⁺",
    "O",
    "<b>NH₄⁺CN⁻</b>", #43
    "HCN", #44
    "CN⁻",
]



# ===== Node-Farben =====
colors = [
    color_eta,      # 0 EtA
    color_h2o,      # 1 WATER
    color_meoh,     # 2 METHANOL

    # EtA-Fragmentierung
    color_eta,      # 3 NH2
    color_eta,      # 4 CH2CH2OH
    color_eta,      # 5 CH2OH (EtA)
    color_eta,      # 6 NH2CH2CH2
    color_eta,      # 7 NH2CH2

    # H / OH bekommen eigene Farben
    color_Hnode,    # 8 H
    color_OHnode,   # 9 OH

    # weiteres EtA
    color_eta,      #10 ETA-H

    # MeOH-deriviert
    color_meoh,     #11 CH3
    color_meoh,     #12 CH3O
    color_meoh,     #13 CH2OH (MeOH)

    # EtA-N-Produkte
    color_eta,      #14 NH3
    color_eta,      #15 MEtA
    color_eta,      #16 DEtA

    # C/O-Spezies (orange)
    color_meoh,     #17 CO2
    color_meoh,     #18 CO
    color_meoh,     #19 H2CO
    color_meoh,      #20 POM (eigene Farbe)
    color_meoh,     #21 HCOOH

    # Wasser-basierte Spezies
    color_h2o,      #22 H2O
    color_meoh,     #23 HCO (≙ CHO)
    color_h2o,      #24 H2O2

    # weitere N/O
    color_eta,      #25 NH4+
    color_eta,      #26 HNCO
    color_eta,      #27 OCN-
    color_h2o,      #28 H2O+
    color_h2o,      #29 H3O+

    # andere C/N-Spezies
    color_meoh,     #30 CH3+
    color_eta,      #31 (CH3)NH
    color_eta,      #32 (CH3)2N
    color_eta,      #33 CHOLINE
    color_meoh,     #34 CH3COH
    color_eta,      #35 (CH3)3N+
    color_meoh,     #36 HOCO
    color_meoh,     #37 CH3CHO
    color_meoh,     #38 CH3OH
    color_meoh,     #39 C
    color_meoh,     #40 HCCO
    color_eta,      #41 DEtA+
    color_meoh,     #42 O (aus CO, gleiche “Familie” wie CO/CO2/HCO…)
    color_eta,      #43 NH4+CN-
    color_eta,      #44 (CH3)N
    color_eta,
]

# ===== NODE COLOR OVERRIDES =====
sources = []
targets = []
values  = []

# ===== Primäre Fragmentierungen =====

# EtA → NH2, CH2CH2OH, OH, NH2CH2CH2, NH2CH2, CH2OH, H
sources += [0, 0, 0, 0, 0, 0, 0]
targets += [3, 4, 9, 6, 7, 5, 8]
values  += [4, 2, 2, 2, 2, 2, 2]

# WATER → OH, H
sources += [1, 1]
targets += [9, 8]
values  += [2, 2]

# METHANOL → CH3, OH, CH3O, H, CH2OH(MeOH)
sources += [2, 2, 2, 2, 2]
targets += [11, 9, 12, 8, 13]
values  += [1, 1, 1, 1, 1]

# ===== Reaktionen =====

# NH2 → NH3  und  H → NH3
sources += [3, 8]
targets += [14, 14]
values  += [1, 1]

# CH3 + EtA → MEtA
sources += [11, 0]
targets += [15, 15]
values  += [1, 1]

# NH3 → POM
sources += [14, 45, 19]
targets += [20, 20, 20]
values  += [1, 1, 1]

sources += [44, 22, 44, 22]
targets += [45, 29, 29, 45]
values  += [1, 1, 1, 1]

sources += [18]
targets += [38]
values  += [1]


# H2CO → POM
sources += [19]
targets += [20]
values  += [1]

# MEtA + CH3 → DEtA
sources += [15, 11]
targets += [16, 16]
values  += [1, 1]

# DEtA → DEtA+
sources += [16]
targets += [41]
values  += [1]


# CO + OH → CO2
sources += [18, 9]
targets += [17, 17]
values  += [1, 1]

# CO + H → HCO
sources += [18, 8]
targets += [23, 23]
values  += [1, 1]

# METHANOL → CO
sources += [2]
targets += [18]
values  += [1]

# CH2OH(MeOH) → H2CO
sources += [13]
targets += [19]
values  += [1]

# CH2OH(MeOH) → H
sources += [13]
targets += [8]
values  += [1]

# WATER → H2O+, H3O+
sources += [1, 1]
targets += [28, 29]
values  += [1, 1]

# H2O+ → H3O+, OH
sources += [28, 28]
targets += [29, 9]
values  += [1, 1]

# DEtA → CHOLINE
sources += [16]
targets += [33]
values  += [1]

# CH3 → CHOLINE
sources += [11]
targets += [33]
values  += [1]

# NH2 → (CH3)NH
sources += [3]
targets += [31]
values  += [1]

# (CH3)NH → (CH3)2N
sources += [31]
targets += [32]
values  += [1]

# CH3 → (CH3)2N
sources += [11]
targets += [32]
values  += [1]

# CH2CH2OH → CHOLINE
sources += [4]
targets += [33]
values  += [1]

# (CH3)2N → (CH3)3N+
sources += [32]
targets += [35]
values  += [1]

# (CH3)3N+ → CHOLINE
sources += [35]
targets += [33]
values  += [1]

# H + OH → H2O
sources += [8, 9]
targets += [22, 22]
values  += [1, 1]

# H2O + H → H2O2
sources += [9]
targets += [24]
values  += [2]

# CO + H → CH3OH
sources += [18]
targets += [38]
values  += [3]

# CH2OH(MeOH) + H → CH3OH
sources += [13, 8]
targets += [38, 38]
values  += [1, 1]

# HCO (≙ CHO) + OH → HCOOH
sources += [23, 9]
targets += [21, 21]
values  += [1, 1]

# HOCO + H → HCOOH
sources += [36, 8]
targets += [21, 21]
values  += [1, 1]

# CH3 + HCO → CH3CHO
sources += [11, 23]
targets += [37, 37]
values  += [1, 1]

sources += [23, 8]
targets += [19, 19]
values  += [1, 1]

# CH3OH + C → HCCO
sources += [23, 39]
targets += [40, 40]
values  += [1, 1]

# HCCO + H → CH3CHO
sources += [40, 8]
targets += [37, 37]
values  += [1, 2]

# H2CO → H und HOCO
sources += [19, 19]
targets += [8, 36]
values  += [1, 1]

# NEU: H2O+ → DEtA+ und DEtA+ → CHOLINE
sources += [28]
targets += [41]
values  += [1]

sources += [41]
targets += [33]
values  += [1]

# NEU: O und C aus CO
# CO → C
sources += [18]
targets += [39]
values  += [1]

# CO → O
sources += [18]
targets += [42]
values  += [1]

sources += [0]
targets += [10]
values += [3]

sources += [26]
targets += [25]
values += [1]

sources += [14]
targets += [25]
values += [1]

sources += [14]
targets += [27]
values += [1]

sources += [26]
targets += [27]
values += [1]

sources += [18]
targets += [26]
values += [1]

sources += [3]
targets += [26]
values += [1]

sources += [28]
targets += [35]
values += [1]

sources += [11]
targets += [31]
values += [1]

sources += [31, 31]
targets += [44, 8]
values += [1, 1]

sources += [44, 14]
targets += [43, 43]
values += [1, 1]



# === DICKEN-ANPASSUNGEN ===
# CH3CHO dick (alle Links zu/von Node 37 x3)
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 37 or t == 37:
        values[i] *= 3

# H → H2O doppelt so groß wie andere von H
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 8 and t == 22:
        values[i] *= 2

# OH → H2O doppelt so groß wie andere von OH
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 9 and t == 22:
        values[i] *= 2

# EtA → H viermal so groß wie andere von EtA
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 0 and t == 8:
        values[i] *= 4

# METHANOL → CH3 halb so groß wie andere von METHANOL
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 2 and t == 11:
        values[i] *= 0.9

# ===== Link-Farben: wie Quellknoten, aber 80% transparent =====
def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ===== LINK COLORS =====
# Default: link color follows source node color (transparent)
link_colors = [hex_to_rgba(colors[s], alpha=0.2) for s in sources]

# --- Specific overrides ---

# NH3 -> POM : dark grey catalytic step
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 14 and t == 20:   # NH3 -> POM
        link_colors[i] = hex_to_rgba(color_catalyst, alpha=0.2)
    if s == 19 and t == 20:   # NH3 -> POM
        link_colors[i] = hex_to_rgba(color_catalyst, alpha=0.2)

#dissociation
for i, (s, t) in enumerate(zip(sources, targets)):
    if t == 3 or t==4 or t==5 or t== 6 or t==7 or t==11 or t== 12 or t==18:  # CHOLINE
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)
    elif s == 0 and t == 9:   #DetA
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)
    elif s == 1 and t == 9:  # DetA
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)
    elif s == 3 and t == 9:  # DetA
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)
    elif s == 18 and t == 42:  # DetA
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)
    elif s == 18 and t == 39:  # DetA
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)
    elif s == 44 and t == 45:  # DetA
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)
    elif s == 22 and t == 45:  # DetA
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)
    elif s == 44 and t == 29:  # DetA
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)
    elif s == 22 and t == 29:  # DetA
        link_colors[i] = hex_to_rgba(color_dissociation, alpha=0.2)

# hydrogenation and dehydrogenation chemistry
for i, (s, t) in enumerate(zip(sources, targets)):
    if t == 37:   # +H
        link_colors[i] = hex_to_rgba(color_hydrogenation, alpha=0.2)
    elif t == 8 or t == 10 or s == 36: # -H
        link_colors[i] = hex_to_rgba(color_dehydrogenation, alpha=0.2)
    elif s == 23 and t == 19: # -H
        link_colors[i] = hex_to_rgba(color_hydrogenation, alpha=0.2)
    elif s == 8 and t == 19: # -H
        link_colors[i] = hex_to_rgba(color_hydrogenation, alpha=0.2)
    elif s == 13 and t == 38:  # +H
        link_colors[i] = hex_to_rgba(color_hydrogenation, alpha=0.2)
    elif s == 3 and t == 14:  # +H
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 18 and t == 23:  # +H
        link_colors[i] = hex_to_rgba(color_hydrogenation, alpha=0.2)
    elif s == 18 and t == 38:  # +H
        link_colors[i] = hex_to_rgba(color_hydrogenation, alpha=0.2)
    elif s == 8 and t == 14:  # +H
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 31 and t == 8:  # +H
        link_colors[i] = hex_to_rgba(color_dehydrogenation, alpha=0.2)
    elif s == 31 and t == 44:  # +H
        link_colors[i] = hex_to_rgba(color_dehydrogenation, alpha=0.2)


# radical-radical
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 8 and t == 22:  # water
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 9 and t == 22 :  # water
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 9 and t == 21:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 23 and t == 21:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 11 and t == 37:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 23 and t == 37:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 41 and t == 33:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 11 and t == 33:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 35 and t == 33:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 4 and t == 33:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 18 and t == 17:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 18 and t == 8:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 9 and t == 17:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 9 and t == 8:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 3 and t == 26:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 18 and t == 26:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 23 and t == 40:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 39 and t == 40:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 9 and t == 24:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 45 and t == 20:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)
    elif s == 19 and t == 20:
        link_colors[i] = hex_to_rgba(color_radical_radical, alpha=0.2)

for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 13 and t == 19:   #
        link_colors[i] = hex_to_rgba(color_dehydrogenation, alpha=0.2)
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 13 and t == 8:   #
        link_colors[i] = hex_to_rgba(color_dehydrogenation, alpha=0.2)
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 36 and t == 21:   #
        link_colors[i] = hex_to_rgba(color_hydrogenation, alpha=0.2)

#methylation
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 11 and t == 15:   #MEta
        link_colors[i] = hex_to_rgba(color_methylation, alpha=0.2)
    elif s == 11 and t == 16:   #DetA
        link_colors[i] = hex_to_rgba(color_methylation, alpha=0.2)
    elif s == 0 and t == 15:  # DetA
        link_colors[i] = hex_to_rgba(color_methylation, alpha=0.2)
    elif s == 3 and t == 31:
        link_colors[i] = hex_to_rgba(color_methylation, alpha=0.2)
    elif s == 31 and t == 32:  # DetA
        link_colors[i] = hex_to_rgba(color_methylation, alpha=0.2)
    elif s == 11 and t == 32:  # DetA
        link_colors[i] = hex_to_rgba(color_methylation, alpha=0.2)
    elif s == 11 and t == 31:  # DetA
        link_colors[i] = hex_to_rgba(color_methylation, alpha=0.2)
    elif s == 15 and t == 16:  # DetA
        link_colors[i] = hex_to_rgba(color_methylation, alpha=0.2)
    elif s == 16 and t == 33:  # DetA
        link_colors[i] = hex_to_rgba(color_methylation, alpha=0.2)


#protonation & ionisation
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 1 and t == 28:   #MEta
        link_colors[i] = hex_to_rgba(color_ionisation, alpha=0.2)
    elif s == 1 and t == 29:
        link_colors[i] = hex_to_rgba(color_protonation, alpha=0.2)
    elif s == 28 and t == 29:  # DetA
        link_colors[i] = hex_to_rgba(color_protonation, alpha=0.2)
    elif s == 32 and t == 35:
        link_colors[i] = hex_to_rgba(color_protonation, alpha=0.2)
    elif s == 28 and t == 9:  # DetA
        link_colors[i] = hex_to_rgba(color_protonation, alpha=0.2)
    elif s == 16 and t == 41:
        link_colors[i] = hex_to_rgba(color_protonation, alpha=0.2)
    elif s == 28 and t == 41:  # DetA
        link_colors[i] = hex_to_rgba(color_protonation, alpha=0.2)
    elif s == 28 and t == 35:  # DetA
        link_colors[i] = hex_to_rgba(color_protonation, alpha=0.2)

#acidbase
for i, (s, t) in enumerate(zip(sources, targets)):
    if s == 26 and t == 25:   #MEta
        link_colors[i] = hex_to_rgba(color_acid_base, alpha=0.2)
    elif s == 26 and t == 27:
        link_colors[i] = hex_to_rgba(color_acid_base, alpha=0.2)
    elif s == 14 and t == 25:   #MEta
        link_colors[i] = hex_to_rgba(color_acid_base, alpha=0.2)
    elif s == 14 and t == 27:
        link_colors[i] = hex_to_rgba(color_acid_base, alpha=0.2)
    elif s == 44 and t == 43:   #MEta
        link_colors[i] = hex_to_rgba(color_acid_base, alpha=0.2)
    elif s == 14 and t == 43:
        link_colors[i] = hex_to_rgba(color_acid_base, alpha=0.2)

NODE_POS: Dict[int, Tuple[float, float]] = {
    # x = 0.0
    0: (0.0, 0.0),
    1: (0.0, 0.3),
    2: (0.0, 0.9),

    # x = 0.2
    3: (0.2, 0.1),
    4: (0.2, 0.15),
    5: (0.2, 0.2),
    6: (0.2, 0.25),
    7: (0.2, 0.3),
    10: (0.2, 0.4),
    28: (0.2, 0.5),
    11: (0.2, 0.6),
    12: (0.2, 0.7),
    13: (0.2, 0.8),
    18: (0.2, 1.0),

    # x = 0.4
    8: (0.345, 0.5),
    9: (0.35, 0.75),

    # x = 0.5
    14: (0.4, 0.1),
    15: (0.4, 0.25),
    30: (0.30, 0.35),
    29: (0.4, 0.9),
    37: (0.4, 1.0),
    22: (0.4, 0.55),
    # x = 0.55

    40: (0.5, 0.9),
    17: (0.5, 1.0),
    16: (0.5, 0.2),
    27: (0.5, 0.4),
    24: (0.5, 0.55),
    26: (0.45, 0.7),
    36: (0.5, 0.8),


    39: (0.6, 0.1),
    25: (0.6, 0.3),
    31: (0.6, 0.5),
    23: (0.6, 0.9),
    # 43: (0.6, 0.75),
    # x = 0.7
    42: (0.6, 0.75),
    33: (0.7, 0.15),
    41: (0.7, 0.3),
    19: (0.7, 0.4),

    # x = 0.8

    35: (0.9, 0.3),
    34: (0.8, 0.45),
    44: (0.8, 0.3),
    43: (0.8, 0.6),
    38: (0.7, 0.65),

    # x = 1.0

    20: (1.0, 0.8),
    21: (1.0, 0.5),
    32: (1.0, 0.2),

}

def build_positions_from_node_pos(n: int, node_pos: Dict[int, Tuple[float, float]]):
    # Default: None = Plotly darf platzieren, aber wir nutzen fixed -> daher setzen wir Defaults sinnvoll
    xs = [0.0] * n
    ys = [0.0] * n

    # einfache Defaults: alle links/mittig (du kannst das später verbessern)
    for i in range(n):
        xs[i] = 0.1
        ys[i] = 0.5

    # Overrides anwenden
    for idx, (xv, yv) in node_pos.items():
        if 0 <= idx < n:
            xs[idx] = float(xv)
            ys[idx] = float(yv)
        else:
            print(f"Warnung: NODE_POS index {idx} existiert nicht (0..{n-1}).")

    # Clipping auf 0..1
    xs = [min(1.0, max(0.0, v)) for v in xs]
    ys = [min(1.0, max(0.0, v)) for v in ys]
    return xs, ys

node_x, node_y = build_positions_from_node_pos(len(labels), NODE_POS)


fig = go.Figure(data=[go.Sankey(
    arrangement="fixed",   # <-- entscheidend!
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=colors,
        x=node_x,
        y=node_y,
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors
    )
)])


def add_custom_legend(fig):
    """
    Adds a custom legend box (nodes + links) at the bottom of the Sankey plot.
    Coordinates are in paper space (0–1).
    """

    # ---- Legend box ----
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.00, x1=1,
        y0=-0.18, y1=-0.06,
        line=dict(color="black", width=0.5),
        fillcolor="white",
        layer="above",
    )

    # ---- Node legend entries (squares) ----
    node_entries = [
        ("EtA-rel. species", color_eta),
        ("H₂O-rel. species",color_h2o),
        ("CH₃OH-rel. species", color_meoh),
    ]

    x_start = 0.015
    y_node = -0.10
    dx = 0.21

    for i, (label, color) in enumerate(node_entries):
        x = x_start + i * dx

        # colored square
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=x, x1=x + 0.02,
            y0=y_node, y1=y_node + 0.02,
            fillcolor=color,
            line=dict(color=color, width=0.5),
        )

        # text
        fig.add_annotation(
            xref="paper", yref="paper",
            x=x + 0.025,
            y=y_node + 0.015,
            text=label,
            showarrow=False,
            font=dict(size=24),
            xanchor="left",
            yanchor="middle",
        )

    # ---- Link legend entries (lines) ----
    link_entries = [
        ("hydrogenation", color_hydrogenation),
        ("dehydrogenation", color_dehydrogenation),
        ("catalytic polymerisation", color_catalyst),
        ("acid-base reaction", color_acid_base),
        ("radical–radical", color_radical_radical),
        ("dissociation", color_dissociation),
        ("methylation", color_methylation),
        ("protonation", color_protonation),
        ("radiolytic induced ionisation", color_ionisation)
    ]

    y_link_top = -0.13
    y_link_bottom = -0.16
    dx = 0.21

    for i, (label, color) in enumerate(link_entries):
        row = i // 5  # 0 or 1
        col = i % 5

        x = x_start + col * dx
        y = y_link_top if row == 0 else y_link_bottom

        fig.add_shape(
            type="line",
            xref="paper", yref="paper",
            x0=x, x1=x + 0.03,
            y0=y, y1=y,
            line=dict(
                color=hex_to_rgba(color, alpha=0.3),
                width=7
            ),
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=x + 0.035,
            y=y,
            text=label,
            showarrow=False,
            font=dict(size=20),
            xanchor="left",
            yanchor="middle",
        )


apply_layout(fig, FIG_LAYOUT)

add_custom_legend(fig)

fig.show()


# ----- Optional: speichern -----
output_dir = r"/home/tara/PyCharmMiscProject/MA_MPE/output"
export_figure(fig, output_dir)
