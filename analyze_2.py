import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ssig
import fetch_data

# ----------------------------
# Signal helpers
# ----------------------------
def build_displacement(data, hub):
    t = np.asarray(data["Time"], float)
    pitch_deg = np.asarray(data["PtfmPitch"], float)
    surge = np.asarray(data["PtfmSurge"], float)

    x_loc = surge + hub * np.sin(np.deg2rad(pitch_deg))
    x = ssig.detrend(x_loc, type="linear")  # dynamic displacement
    return t, x

def calc_velocity(t, x, method="savgol", window=201, poly=3):
    dt = float(np.mean(np.diff(t)))
    if method == "gradient":
        return np.gradient(x, dt)

    w = min(window, len(x) - (1 - len(x) % 2))
    if w < 5:
        return np.gradient(x, dt)
    if w % 2 == 0:
        w -= 1
    p = min(poly, w - 2)
    return ssig.savgol_filter(x, window_length=w, polyorder=p, deriv=1, delta=dt, mode="interp")

# ----------------------------
# PSD helpers
# ----------------------------
def fft_psd_density(t, x):
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    n = len(x)

    X = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(n, d=dt)

    # one-sided PSD density [x^2/Hz]
    psd = (1.0 / (fs * n)) * np.abs(X)**2
    if n % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    return f, psd

def welch_psd_density(t, x):
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    f, psd = ssig.welch(
        x, fs=fs, window="hann", nperseg=8192,
        noverlap=8192//2, detrend="constant", scaling="density",
    )
    return f, psd

def band_metrics(f, Pxx, frange):
    m = (f >= frange[0]) & (f <= frange[1])
    if np.sum(m) < 2:
        return np.nan, np.nan, np.nan
    fpk = float(f[m][np.argmax(Pxx[m])])
    var_band = float(np.trapezoid(Pxx[m], f[m]))
    std_band = np.sqrt(max(var_band, 0.0))
    Aeq_band = np.sqrt(2.0) * std_band
    return fpk, std_band, Aeq_band

def stack_psd(spec_list):
    # spec_list: list of (f, Pxx), returns aligned f, mean, std
    f_ref = spec_list[0][0]
    P = []
    for f, p in spec_list:
        if np.array_equal(f, f_ref):
            P.append(p)
        else:
            P.append(np.interp(f_ref, f, p))
    P = np.vstack(P)
    P_mean = np.mean(P, axis=0)
    P_std = np.std(P, axis=0, ddof=1) if P.shape[0] > 1 else np.zeros_like(P_mean)
    return f_ref, P_mean, P_std

# ----------------------------
# Plot helper
# ----------------------------
def plot_condition_2x2(
    t_list, x_list, v_list, Tp,
    f_x_fft, P_x_fft_mean, P_x_fft_std,
    f_x_wel, P_x_wel_mean, P_x_wel_std,
    f_v_fft, P_v_fft_mean, P_v_fft_std,
    f_v_wel, P_v_wel_mean, P_v_wel_std,
    title, out_png
):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_t_x, ax_t_v = axs[0, 0], axs[0, 1]
    ax_p_x, ax_p_v = axs[1, 0], axs[1, 1]

    # top row: seed traces only
    for i, (t, x) in enumerate(zip(t_list, x_list)):
        ax_t_x.plot(t, x, lw=1.0, label=f"Seed {i+1}")
    for i, (t, v) in enumerate(zip(t_list, v_list)):
        ax_t_v.plot(t, v, lw=1.0, label=f"Seed {i+1}")

    ax_t_x.set_title("Displacement (seed traces)")
    ax_t_x.set_xlabel("Time [s]")
    ax_t_x.set_ylabel("x [m]")
    ax_t_x.grid(alpha=0.3)

    ax_t_v.set_title("Velocity (seed traces)")
    ax_t_v.set_xlabel("Time [s]")
    ax_t_v.set_ylabel("v [m/s]")
    ax_t_v.grid(alpha=0.3)

    # bottom row: PSD mean ± std
    eps = np.finfo(float).tiny

    # displacement PSD
    ax_p_x.loglog(f_x_fft[1:], P_x_fft_mean[1:], "C0-", lw=2, label="FFT mean")
    ax_p_x.axvline(x = 1 / Tp, linestyle = "dashed", color = "k")
    ax_p_x.loglog(f_x_wel[1:], P_x_wel_mean[1:], "C1--", lw=2, label="Welch mean")

    ax_p_x.set_title("Displacement PSD (ensemble mean ± std)")
    ax_p_x.set_xlabel("Frequency [Hz]")
    ax_p_x.set_ylabel("PSD [m²/Hz]")
    ax_p_x.grid(alpha=0.3)
    ax_p_x.legend()

    # velocity PSD
    ax_p_v.loglog(f_v_fft[1:], P_v_fft_mean[1:], "C0-", lw=2, label="FFT mean")
    ax_p_v.axvline(x = 1 / Tp, linestyle = "dashed", color = "k")
    ax_p_v.loglog(f_v_wel[1:], P_v_wel_mean[1:], "C1--", lw=2, label="Welch mean")

    ax_p_v.set_title("Velocity PSD (ensemble mean ± std)")
    ax_p_v.set_xlabel("Frequency [Hz]")
    ax_p_v.set_ylabel("PSD [(m/s)²/Hz]")
    ax_p_v.grid(alpha=0.3)
    ax_p_v.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Main analysis
# ----------------------------
def run_analysis(
    cases, fetch_data,
    LF_frange=(1/300, 1/30),
    WF_frange=(1/24, 1/1),
    vel_method="savgol",
    out_csv="condition_summary_stats_all_cases.csv",
    fig_dir="fig_ensemble"
):
    os.makedirs(fig_dir, exist_ok=True)
    summary_rows = []

    cases = cases[cases["HWindSpeed"].isin([10.5, 5.0, 25.0])]

    for (U, Hs, Tp), g in cases.groupby(["HWindSpeed", "WaveHs", "WaveTp"], sort=True):
        if g.empty:
            continue

        print(f"Selected cases: HWindSpeed = {U}, WaveHs = {Hs}, WaveTp = {Tp}")

        t_list, x_list, v_list = [], [], []
        x_fft_specs, x_wel_specs = [], []
        v_fft_specs, v_wel_specs = [], []
        seed_metric_rows = []

        for _, row in g.iterrows():
            case_name = row["case_name"]
            file = fetch_data.DATA_PATH + "outfiles/" + case_name + ".out"
            data = fetch_data.load_outfile(file)

            t, x = build_displacement(data, fetch_data.HUB)
            v = calc_velocity(t, x, method=vel_method)

            fx_fft, Px_fft = fft_psd_density(t, x)
            fx_wel, Px_wel = welch_psd_density(t, x)
            fv_fft, Pv_fft = fft_psd_density(t, v)
            fv_wel, Pv_wel = welch_psd_density(t, v)

            # store for ensemble spectra
            x_fft_specs.append((fx_fft, Px_fft))
            x_wel_specs.append((fx_wel, Px_wel))
            v_fft_specs.append((fv_fft, Pv_fft))
            v_wel_specs.append((fv_wel, Pv_wel))

            # store for top-row traces
            t_list.append(t); x_list.append(x); v_list.append(v)

            # # per-seed band metrics (for mean/std across seeds)
            # seed_m = {}

            # seed_m["x_fft_f_lf"], _, seed_m["x_fft_A_lf"] = band_metrics(fx_fft, Px_fft, LF_frange)
            # seed_m["x_fft_f_wf"], _, seed_m["x_fft_A_wf"] = band_metrics(fx_fft, Px_fft, WF_frange)
            # seed_m["x_wel_f_lf"], _, seed_m["x_wel_A_lf"] = band_metrics(fx_wel, Px_wel, LF_frange)
            # seed_m["x_wel_f_wf"], _, seed_m["x_wel_A_wf"] = band_metrics(fx_wel, Px_wel, WF_frange)

            # seed_m["v_fft_f_lf"], _, seed_m["v_fft_A_lf"] = band_metrics(fv_fft, Pv_fft, LF_frange)
            # seed_m["v_fft_f_wf"], _, seed_m["v_fft_A_wf"] = band_metrics(fv_fft, Pv_fft, WF_frange)
            # seed_m["v_wel_f_lf"], _, seed_m["v_wel_A_lf"] = band_metrics(fv_wel, Pv_wel, LF_frange)
            # seed_m["v_wel_f_wf"], _, seed_m["v_wel_A_wf"] = band_metrics(fv_wel, Pv_wel, WF_frange)

            # seed_metric_rows.append(seed_m)

        # ensemble PSDs
        fx_fft_e, Px_fft_mean, Px_fft_std = stack_psd(x_fft_specs)
        fx_wel_e, Px_wel_mean, Px_wel_std = stack_psd(x_wel_specs)
        fv_fft_e, Pv_fft_mean, Pv_fft_std = stack_psd(v_fft_specs)
        fv_wel_e, Pv_wel_mean, Pv_wel_std = stack_psd(v_wel_specs)

        # ensemble band metrics
        row_out = {"HWindSpeed": U, "WaveHs": Hs, "WaveTp": Tp, "n_seeds": len(g)}

        row_out["x_fft_ens_f_lf"], _, row_out["x_fft_ens_A_lf"] = band_metrics(fx_fft_e, Px_fft_mean, LF_frange)
        row_out["x_fft_ens_f_wf"], _, row_out["x_fft_ens_A_wf"] = band_metrics(fx_fft_e, Px_fft_mean, WF_frange)
        row_out["x_wel_ens_f_lf"], _, row_out["x_wel_ens_A_lf"] = band_metrics(fx_wel_e, Px_wel_mean, LF_frange)
        row_out["x_wel_ens_f_wf"], _, row_out["x_wel_ens_A_wf"] = band_metrics(fx_wel_e, Px_wel_mean, WF_frange)

        row_out["v_fft_ens_f_lf"], _, row_out["v_fft_ens_A_lf"] = band_metrics(fv_fft_e, Pv_fft_mean, LF_frange)
        row_out["v_fft_ens_f_wf"], _, row_out["v_fft_ens_A_wf"] = band_metrics(fv_fft_e, Pv_fft_mean, WF_frange)
        row_out["v_wel_ens_f_lf"], _, row_out["v_wel_ens_A_lf"] = band_metrics(fv_wel_e, Pv_wel_mean, LF_frange)
        row_out["v_wel_ens_f_wf"], _, row_out["v_wel_ens_A_wf"] = band_metrics(fv_wel_e, Pv_wel_mean, WF_frange)

        # Av from Ax*2πf (ensemble)
        row_out["Av_from_Ax_wf_fft_ens"] = row_out["x_fft_ens_A_wf"] * 2*np.pi * row_out["x_fft_ens_f_wf"]
        row_out["Av_from_Ax_wf_wel_ens"] = row_out["x_wel_ens_A_wf"] * 2*np.pi * row_out["x_wel_ens_f_wf"]

        # seed mean/std of band metrics
        # seed_df_local = pd.DataFrame(seed_metric_rows)
        # for c in seed_df_local.columns:
        #     row_out[f"{c}_seed_mean"] = seed_df_local[c].mean()
        #     row_out[f"{c}_seed_std"] = seed_df_local[c].std(ddof=1)

        summary_rows.append(row_out)

        # make plot
        title = f"HWindSpeed={U}, WaveHs={Hs}, WaveTp={Tp}, n={len(g)}"
        out_png = os.path.join(fig_dir, f"U{U}_Hs{Hs}_Tp{Tp}.png")
        plot_condition_2x2(
            t_list, x_list, v_list, Tp,
            fx_fft_e, Px_fft_mean, Px_fft_std,
            fx_wel_e, Px_wel_mean, Px_wel_std,
            fv_fft_e, Pv_fft_mean, Pv_fft_std,
            fv_wel_e, Pv_wel_mean, Pv_wel_std,
            title, out_png
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_csv, index=False)
    return summary_df

cases = fetch_data.get_cases()

summary_df = run_analysis(
    cases=cases,
    fetch_data=fetch_data,
    LF_frange=(1/300, 1/30),
    WF_frange=(1/24, 1/1),
    vel_method="savgol",
    out_csv="condition_summary_stats_all_cases.csv",
    fig_dir="fig_ensemble"
)

print(summary_df.head())