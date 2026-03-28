import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import os

from plot_helper import setup_matplotlib

NA = 6.02214076e23  # Avogadro-Zahl in mol^-1

# -------------------------------------------------------------------
# Output-Ordner für den gestapelten Plot
# -------------------------------------------------------------------
output_dir = r"C:\Users\taras\PycharmProjects\masterthesis_mpe\output"
os.makedirs(output_dir, exist_ok=True)
stacked_png = os.path.join(output_dir, "stacked_parent_simulation_nomol.png")
stacked_pdf = os.path.join(output_dir, "stacked_parent_simulation_nomol.pdf")


def compute_band_strength_A(S0, sigma_S0,
                            thickness_cm, sigma_thickness_cm,
                            density_g_cm3, sigma_density_g_cm3,
                            molar_mass_g_mol, sigma_molar_mass_g_mol):
    ln10 = np.log(10.0)
    A0 = (ln10 * molar_mass_g_mol * S0) / (density_g_cm3 * NA * thickness_cm)

    rel_S0  = sigma_S0 / S0 if S0 != 0 else 0.0
    rel_M   = sigma_molar_mass_g_mol / molar_mass_g_mol if molar_mass_g_mol != 0 else 0.0
    rel_rho = sigma_density_g_cm3 / density_g_cm3 if density_g_cm3 != 0 else 0.0
    rel_d   = sigma_thickness_cm / thickness_cm if thickness_cm != 0 else 0.0

    rel_A2 = rel_S0**2 + rel_M**2 + rel_rho**2 + rel_d**2
    sigma_A0 = A0 * np.sqrt(rel_A2)

    return A0, sigma_A0


def propagate_error_deltaN(S, sigma_S, A, sigma_A):
    deltaN = S / A
    term_S = (sigma_S / A) ** 2
    term_A = (S * sigma_A / A**2) ** 2
    sigma_deltaN = np.sqrt(term_S + term_A)
    return deltaN, sigma_deltaN


def propagate_error_Nphi(N0, sigma_N0, deltaN, sigma_deltaN):
    Nphi = N0 + deltaN
    sigma_Nphi = np.sqrt(sigma_N0**2 + sigma_deltaN**2)
    return Nphi, sigma_Nphi


def fit_parent_species(label,
                       thickness_cm, sigma_thickness_cm,
                       density_g_cm3, sigma_density_g_cm3,
                       molar_mass_g_mol, sigma_molar_mass_g_mol,
                       S0, sigma_S0,
                       S_irr, sigma_S_irr,
                       times_min,
                       ylim=None):

    A0, sigma_A0 = compute_band_strength_A(
        S0, sigma_S0,
        thickness_cm, sigma_thickness_cm,
        density_g_cm3, sigma_density_g_cm3,
        molar_mass_g_mol, sigma_molar_mass_g_mol
    )

    print(f"\n=== {label} ===")
    print("Bandstärke A0: {:.3e} ± {:.3e}".format(A0, sigma_A0))

    N0, sigma_N0 = propagate_error_deltaN(S0, sigma_S0, A0, sigma_A0)
    print("Initial N0: {:.3e} ± {:.3e} molec/cm^2".format(N0, sigma_N0))

    times_s = times_min * 60.0

    t_list = [0.0]
    N_list = [N0]
    sigma_N_list = [sigma_N0]

    print("\nIrradiation steps:")
    for t_i, S_t, sigma_S_t in zip(times_s, S_irr, sigma_S_irr):
        deltaN_t, sigma_deltaN_t = propagate_error_deltaN(S_t, sigma_S_t, A0, sigma_A0)
        N_t, sigma_N_t = propagate_error_Nphi(N0, sigma_N0, deltaN_t, sigma_deltaN_t)

        t_list.append(t_i)
        N_list.append(N_t)
        sigma_N_list.append(sigma_N_t)

        print(f"t = {t_i:.1f} s ({t_i/60:.1f} min)")
        print("  S(t)      = {:.3e} ± {:.3e}".format(S_t, sigma_S_t))
        print("  DeltaN(t) = {:.3e} ± {:.3e}".format(deltaN_t, sigma_deltaN_t))
        print("  N(t)      = {:.3e} ± {:.3e} molec/cm^2".format(N_t, sigma_N_t))

    t = np.array(t_list)
    N = np.array(N_list)

    def exp_model(t_, k_eff, N_inf):
        return N_inf + (N0 - N_inf) * np.exp(-k_eff * t_)

    k0_guess = 1e-5
    N_min = N.min()
    N_max = N.max()
    N_inf_guess = N_min

    p0 = [k0_guess, N_inf_guess]
    bounds = ([0.0, 0.0],
              [1e-2, 2.0 * N_max])

    popt, pcov = curve_fit(
        exp_model,
        t, N,
        p0=p0,
        bounds=bounds,
        maxfev=10000
    )

    k_eff_fit, N_inf_fit = popt
    dk_eff, dN_inf = np.sqrt(np.diag(pcov))

    print("\nFit-Ergebnis (N(t) = N_inf + (N0 - N_inf)*exp(-k t)):")
    print(f"{label}: k_eff = {k_eff_fit:.3e} ± {dk_eff:.3e} 1/s")
    print(f"{label}: N_inf = {N_inf_fit:.3e} ± {dN_inf:.3e} molec/cm^2")

    fig, ax = plt.subplots()
    t_fit = np.linspace(t.min(), t.max(), 200)
    N_fit = exp_model(t_fit, k_eff_fit, N_inf_fit)

    ax.plot(t, N, 'o', label=f'{label} data')
    ax.plot(t_fit, N_fit, '-', label=f'{label} exp fit')

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel('t (s)')
    ax.set_ylabel(f'N$_{{{label}}}$ (molecules cm$^2$)')
    ax.legend()

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(True)

    fig.tight_layout()
    plt.show()

    return k_eff_fit, dk_eff, N_inf_fit, dN_inf, t, N, N0


def fit_choline_from_areas(label, S_chol, sigma_S_chol, times_min):
    times_s = times_min * 60.0
    S0 = 0.0

    def form_model(t_, k_form, S_inf):
        return S_inf - (S_inf - S0) * np.exp(-k_form * t_)

    k0_guess = 1e-4
    S_inf_guess = S_chol[-1]

    p0 = [k0_guess, S_inf_guess]

    bounds = ([0.0, 0.0],
              [1e-1, 10.0 * max(S_chol)])

    popt, pcov = curve_fit(
        form_model,
        times_s, S_chol,
        p0=p0,
        sigma=sigma_S_chol,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=10000
    )

    k_form, S_inf = popt
    dk_form, dS_inf = np.sqrt(np.diag(pcov))

    print(f"\n=== {label} (Cholin) ===")
    print(f"k_form = {k_form:.3e} ± {dk_form:.3e} 1/s")
    print(f"S_inf  = {S_inf:.3e} ± {dS_inf:.3e} (arb. units)")

    return k_form, dk_form, S_inf, dS_inf


def eta_ode(t, y, k_eta, Ninf_eta, f_rad, f_cholin):
    E, R, C = y
    loss_E = k_eta * (E - Ninf_eta)
    dE_dt = -loss_E
    dR_dt = f_rad    * loss_E
    dC_dt = f_cholin * loss_E
    return [dE_dt, dR_dt, dC_dt]


def meoh_ode(t, y, k_meoh, Ninf_meoh, f_dehyd, f_CH3, f_OH):
    M, D, CH3, OH = y
    loss_M = k_meoh * (M - Ninf_meoh)
    dM_dt   = -loss_M
    dD_dt   = f_dehyd * loss_M
    dCH3_dt = f_CH3   * loss_M
    dOH_dt  = f_OH    * loss_M
    return [dM_dt, dD_dt, dCH3_dt, dOH_dt]


if __name__ == "__main__":
    setup_matplotlib()

    thickness_EtA_cm = 1.034557e-04
    sigma_thickness_EtA_cm = 5.172787e-06

    thickness_H2O_cm = 4.593441e-06
    sigma_thickness_H2O_cm = 2.296720e-07

    thickness_MeOH_cm = 1.022621e-05
    sigma_thickness_MeOH_cm = 5.113105e-07

    density_eta = 1.01
    sigma_density_eta = 0.00
    M_eta = 61.08
    sigma_M_eta = 0.0

    S0_eta = 1.714
    sigma_S0_eta = 3.784

    S_irr_eta = np.array([-0.02041, -0.03002, -0.03612, -0.04502, -0.05304])
    sigma_S_irr_eta = np.array([0.03741, 0.01394, 0.01101, 0.02872, 0.07946])

    times_min_eta = np.array([1, 5, 15, 30, 60])

    k_eta, dk_eta, Ninf_eta, dNinf_eta, t_eta, N_eta, N0_eta = fit_parent_species(
        label="EtA",
        thickness_cm=thickness_EtA_cm,
        sigma_thickness_cm=sigma_thickness_EtA_cm,
        density_g_cm3=density_eta,
        sigma_density_g_cm3=sigma_density_eta,
        molar_mass_g_mol=M_eta,
        sigma_molar_mass_g_mol=sigma_M_eta,
        S0=S0_eta,
        sigma_S0=sigma_S0_eta,
        S_irr=S_irr_eta,
        sigma_S_irr=sigma_S_irr_eta,
        times_min=times_min_eta,
        ylim=(0.415e18, 0.45e18)
    )

    density_h2o = 0.93
    sigma_density_h2o = 0.0
    M_h2o = 18.015
    sigma_M_h2o = 0.0

    S0_h2o = 0.5182
    sigma_S0_h2o = 0.8677

    S_irr_h2o = np.array([0.1323, 0.1464, 0.2097, 0.2292, 0.2351])
    sigma_S_irr_h2o = np.array([0.1464, 0.0128, 0.1367, 0.2273, 0.4032])

    times_min_h2o = np.array([1, 5, 15, 30, 60])

    k_h2o, dk_h2o, Ninf_h2o, dNinf_h2o, t_h2o, N_h2o, N0_h2o = fit_parent_species(
        label="H2O",
        thickness_cm=thickness_H2O_cm,
        sigma_thickness_cm=sigma_thickness_H2O_cm,
        density_g_cm3=density_h2o,
        sigma_density_g_cm3=sigma_density_h2o,
        molar_mass_g_mol=M_h2o,
        sigma_molar_mass_g_mol=sigma_M_h2o,
        S0=S0_h2o,
        sigma_S0=sigma_S0_h2o,
        S_irr=S_irr_h2o,
        sigma_S_irr=sigma_S_irr_h2o,
        times_min=times_min_h2o,
        ylim=None
    )

    density_meoh = 0.779
    sigma_density_meoh = 0.0
    M_meoh = 32.04
    sigma_M_meoh = 0.0

    S0_meoh = 0.4015
    sigma_S0_meoh = 0.048

    S_irr_meoh = np.array([-0.03, -0.06, -0.08, -0.09, -0.11])
    sigma_S_irr_meoh = np.array([0.173, 0.1443, 0.1538, 0.1035, 0.0518])

    times_min_meoh = np.array([1, 5, 15, 30, 60])

    k_meoh, dk_meoh, Ninf_meoh, dNinf_meoh, t_meoh, N_meoh, N0_meoh = fit_parent_species(
        label="MeOH",
        thickness_cm=thickness_MeOH_cm,
        sigma_thickness_cm=sigma_thickness_MeOH_cm,
        density_g_cm3=density_meoh,
        sigma_density_g_cm3=sigma_density_meoh,
        molar_mass_g_mol=M_meoh,
        sigma_molar_mass_g_mol=sigma_M_meoh,
        S0=S0_meoh,
        sigma_S0=sigma_S0_meoh,
        S_irr=S_irr_meoh,
        sigma_S_irr=sigma_S_irr_meoh,
        times_min=times_min_meoh,
        ylim=None
    )

    print("\nZusammenfassung der k_eff:")
    print(f"EtA : k_eff = {k_eta:.3e} ± {dk_eta:.3e} 1/s")
    print(f"H2O : k_eff = {k_h2o:.3e} ± {dk_h2o:.3e} 1/s")
    print(f"MeOH: k_eff = {k_meoh:.3e} ± {dk_meoh:.3e} 1/s")

    times_min_chol = np.array([1, 5, 15, 30, 60])

    S_chol1 = np.array([0.01689, 0.02257, 0.02653, 0.03318, 0.04023])
    sigma_S_chol1 = np.array([0.01689, 0.05065, 0.05174, 0.08414, 0.15482])

    S_chol2 = np.array([0.00781, 0.01144, 0.01196, 0.01410, 0.01621])
    sigma_S_chol2 = np.array([0.00203, 0.0321, 0.03124, 0.04835, 0.08518])

    k_chol1, dk_chol1, S_inf1, dS_inf1 = fit_choline_from_areas(
        label="Cholin Peak 1",
        S_chol=S_chol1,
        sigma_S_chol=sigma_S_chol1,
        times_min=times_min_chol
    )

    k_chol2, dk_chol2, S_inf2, dS_inf2 = fit_choline_from_areas(
        label="Cholin Peak 2",
        S_chol=S_chol2,
        sigma_S_chol=sigma_S_chol2,
        times_min=times_min_chol
    )

    print("\nCholin-Zusammenfassung:")
    print(f"Peak 1: k_form = {k_chol1:.3e} ± {dk_chol1:.3e} 1/s")
    print(f"Peak 2: k_form = {k_chol2:.3e} ± {dk_chol2:.3e} 1/s")

    t_span = (0.0, 3600.0)
    t_eval = np.linspace(t_span[0], t_span[1], 300)

    f_rad_eta    = 0.5
    f_cholin_eta = 0.3

    y0_eta = [N0_eta, 0.0, 0.0]

    sol_eta = solve_ivp(
        fun=lambda t, y: eta_ode(t, y, k_eta, Ninf_eta, f_rad_eta, f_cholin_eta),
        t_span=t_span,
        y0=y0_eta,
        t_eval=t_eval,
        method="LSODA"
    )

    t_sim = sol_eta.t
    E_sim_eta, R_sim_eta, C_sim_eta = sol_eta.y

    W_sim = Ninf_h2o + (N0_h2o - Ninf_h2o) * np.exp(-k_h2o * t_sim)

    f_dehyd = 0.3
    f_CH3   = 0.4
    f_OH    = 0.3

    y0_meoh = [N0_meoh, 0.0, 0.0, 0.0]

    sol_meoh = solve_ivp(
        fun=lambda t, y: meoh_ode(t, y, k_meoh, Ninf_meoh, f_dehyd, f_CH3, f_OH),
        t_span=t_span,
        y0=y0_meoh,
        t_eval=t_eval,
        method="LSODA"
    )

    M_sim, D_sim, CH3_sim, OH_sim = sol_meoh.y

    fig, axes = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(6, 8),
        constrained_layout=True
    )

    ax_eta = axes[0]
    ax_eta.plot(t_sim, E_sim_eta, color="#98FB98", label="EtA simulation")
    ax_eta.plot(t_eta, N_eta, 'o', color="#006400", label="EtA data")
    ax_eta.set_ylabel(r"N$_{\mathrm{EtA}}$ (molecules cm$^{-2}$)")
    ax_eta.set_ylim(4.3e17, 4.5e17)
    ax_eta.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax_eta.spines[spine].set_visible(True)

    ax_h2o = axes[1]
    ax_h2o.plot(t_sim, W_sim, color="#ADD8E6", label="H$_2$O simulation")
    ax_h2o.plot(t_h2o, N_h2o, 'o', color="#00008B", label="H$_2$O data")
    ax_h2o.set_ylabel(r"N$_{\mathrm{H_2O}}$ (molecules cm$^{-2}$)")
    ax_h2o.legend(loc="center right", frameon=False)
    for spine in ("top", "right"):
        ax_h2o.spines[spine].set_visible(True)

    ax_meoh = axes[2]
    ax_meoh.plot(t_sim, M_sim, color="#FFD580", label="MeOH simulation")
    ax_meoh.plot(t_meoh, N_meoh, 'o', color="#FF8C00", label="MeOH data")
    ax_meoh.set_ylabel(r"N$_{\mathrm{MeOH}}$ (molecules cm$^{-2}$)")
    ax_meoh.set_xlabel("t (s)")
    ax_meoh.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax_meoh.spines[spine].set_visible(True)

    fig.savefig(stacked_png, dpi=300)
    fig.savefig(stacked_pdf)

    plt.show()

    print(f"\nGestapelter Plot gespeichert als:")
    print("  PNG:", stacked_png)
    print("  PDF:", stacked_pdf)
