import os
import re
import numpy as np
import matplotlib.pyplot as plt


# === Path & filenames ===
folder = r"C:\Users\taras\OneDrive\Desktop\MA\Subtraction spectra\Methanol"
basenames = [f"Subtractionspectra_Methanol{i}" for i in range(1, 6)]
basenames.append("20240626_Deposition_Methanol_20K_sample1")
ext_candidates = [".csv", ".txt", ".dat", ".dpt", ".asc"]

# === Plot x-window (adjust if you want) ===
x_left, x_right = 1085, 1175


def find_file(folder, base, exts):
    for ext in exts:
        p = os.path.join(folder, base + ext)
        if os.path.exists(p):
            return p
    p2 = os.path.join(folder, base)
    if os.path.exists(p2):
        return p2
    return None


def read_two_columns(path):
    # Try simple splitting
    for sep in [None, ";", ",", "\t"]:
        try:
            wn, y = [], []
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(sep) if sep is not None else line.split()
                    if len(parts) < 2:
                        continue
                    a = parts[0].replace(",", ".")
                    b = parts[1].replace(",", ".")
                    try:
                        wn.append(float(a))
                        y.append(float(b))
                    except Exception:
                        continue
            if len(wn) >= 5:
                return np.array(wn, float), np.array(y, float)
        except Exception:
            pass

    # Regex fallback
    number_re = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
    wn, y = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            nums = number_re.findall(line)
            if len(nums) >= 2:
                a = nums[0].replace(",", ".")
                b = nums[1].replace(",", ".")
                try:
                    wn.append(float(a))
                    y.append(float(b))
                except Exception:
                    continue

    return np.array(wn, float), np.array(y, float)


# === Load spectra ===
spectra = []
for base in basenames:
    path = find_file(folder, base, ext_candidates)
    if path is None:
        raise FileNotFoundError(f"Not found: {base} (+ {ext_candidates}) in\n{folder}")

    wn, y = read_two_columns(path)
    if wn.size == 0:
        raise ValueError(f"No valid numeric data read from: {path}")

    # sort + remove duplicate wavenumbers
    idx = np.argsort(wn)
    wn = wn[idx]
    y = y[idx]
    wn_u, uniq_idx = np.unique(wn, return_index=True)
    y_u = y[uniq_idx]

    spectra.append((base, wn_u, y_u))

# === Common x-grid = first spectrum ===
wn_ref = spectra[0][1]
Y = []
for name, wn, y in spectra:
    y_i = np.interp(wn_ref, wn, y)
    Y.append(y_i)
Y = np.column_stack(Y)


# === Integration window ===
a, b = 1142.0, 1110.0
lo, hi = min(a, b), max(a, b)
mask_int = (wn_ref >= lo) & (wn_ref <= hi)
wn_int = wn_ref[mask_int]
if wn_int.size < 2:
    raise ValueError(f"Too few points in integration window {a}–{b} cm^-1.")
order_int = np.argsort(wn_int)


# === Plot (NO stacking) + mark integration region to y=0 ===
fig, ax = plt.subplots()

print(f"Integration over {a}–{b} cm^-1 (relative to y=0):")
print("------------------------------------------------")
for i, (name, _, _) in enumerate(spectra):
    y = Y[:, i]

    # line
    ax.plot(wn_ref, y, label=name)

    # shaded integration area between curve and y=0 in the window
    ax.fill_between(wn_ref[mask_int], y[mask_int], 0.0, alpha=0.25)

    # integrals (NumPy 2.x)
    area_signed = np.trapezoid(y[mask_int][order_int], wn_int[order_int])
    area_abs    = np.trapezoid(np.abs(y[mask_int][order_int]), wn_int[order_int])

    # difference between absolute and signed (magnitude) integral
    diff = area_abs - abs(area_signed)

    print(
        f"{name}: signed = {area_signed:.6e}   |abs| = {area_abs:.6e}   "
        f"diff(|abs|-|signed|) = {diff:.6e}"
    )

ax.axhline(0.0, linewidth=1)
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_xlim(x_left, x_right)
ax.set_ylim(-0.005, 0.005)
ax.set_title("Methanol spectra (same y-axis, no stacking) + integration area")
ax.invert_xaxis()
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

