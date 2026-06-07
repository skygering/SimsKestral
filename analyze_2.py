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
    t = np.asarray(data["Time"], dtype=float)
    pitch_deg = np.asarray(data["PtfmPitch"], dtype=float)
    surge = np.asarray(data["PtfmSurge"], dtype=float)

    x_loc = surge + hub * np.sin(np.deg2rad(pitch_deg))
    x = ssig.detrend(x_loc, type="linear")
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
    return ssig.savgol_filter(
        x, window_length=w, polyorder=p, deriv=1, delta=dt, mode="interp"
    )

# ----------------------------
# PSD helpers
# ----------------------------
def fft_psd_density(t, x):
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    n = len(x)

    X = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(n, d=dt)

    psd = (1.0 / (fs * n)) * np.abs(X) ** 2
    if n % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    return f, psd

def welch_psd_density(t, x, nperseg=4096, noverlap=None):
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt

    nperseg_eff = min(int(nperseg), len(x))
    if nperseg_eff < 8:
        return np.array([np.nan]), np.array([np.nan])

    if noverlap is None:
        noverlap_eff = nperseg_eff // 2
    else:
        noverlap_eff = min(int(noverlap), nperseg_eff - 1)

    f, psd = ssig.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        detrend="constant",
        scaling="density",
    )
    return f, psd

def band_metrics(f, Pxx, frange):
    m = (f >= frange[0]) & (f <= frange[1])
    if np.sum(m) < 2:
        return np.nan, np.nan
    f_band = f[m]
    p_band = Pxx[m]
    fpk = float(f_band[np.argmax(p_band)])
    var_band = float(np.trapezoid(p_band, f_band))
    Aeq_band = np.sqrt(2.0) * np.sqrt(max(var_band, 0.0))
    return fpk, Aeq_band

def worst_case_amplitude(t, x, t_start=800, t_end=1000):
    m = (t >= t_start) & (t <= t_end)
    if np.sum(m) < 2:
        return np.nan
    xw = x[m]
    return 0.5 * (np.max(xw) - np.min(xw))

def nanmean_std(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    if a.size == 1:
        return float(a[0]), np.nan
    return float(np.mean(a)), float(np.std(a, ddof=1))

def stack_psd_mean(spec_list):
    """
    Fast mean stack, assumes same frequency grid for all seeds.
    Falls back to interpolation if needed.
    """
    if len(spec_list) == 0:
        return None, None

    f_ref = spec_list[0][0]
    aligned = all(np.array_equal(f_ref, f) for f, _ in spec_list[1:])

    if aligned:
        P = np.vstack([p for _, p in spec_list])
    else:
        P = np.vstack([np.interp(f_ref, f, p) for f, p in spec_list])

    return f_ref, np.mean(P, axis=0)

# ----------------------------
# CT and CT' helper
# ----------------------------
def pooled_nanmean_std(series_list):
    """
    series_list: list of 1D arrays (one per seed)
    returns pooled mean/std over all finite points from all seeds+time
    """
    if len(series_list) == 0:
        return np.nan, np.nan, 0

    y = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series_list if s is not None and len(s) > 0])
    y = y[np.isfinite(y)]

    if y.size == 0:
        return np.nan, np.nan, 0
    if y.size == 1:
        return float(y[0]), np.nan, 1
    return float(np.mean(y)), float(np.std(y, ddof=1)), int(y.size)

# ----------------------------
# Plot helper
# ----------------------------
def plot_condition_2x2(
    t_list, x_list, v_list, Tp,
    x_wel_specs, v_wel_specs,              # <-- per-seed Welch spectra
    f_x_wel, P_x_wel_mean,
    f_v_wel, P_v_wel_mean,
    title, out_png,
    x_fft_specs=None, v_fft_specs=None,    # <-- per-seed FFT spectra (optional)
    f_x_fft=None, P_x_fft_mean=None,
    f_v_fft=None, P_v_fft_mean=None,
    plot_seed_psd=True
):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_t_x, ax_t_v = axs[0, 0], axs[0, 1]
    ax_p_x, ax_p_v = axs[1, 0], axs[1, 1]

    # top row traces
    for i, (t, x) in enumerate(zip(t_list, x_list)):
        ax_t_x.plot(t, x, lw=1.5, label=f"Seed {i+1}")
    for i, (t, v) in enumerate(zip(t_list, v_list)):
        ax_t_v.plot(t, v, lw=1.5, label=f"Seed {i+1}")

    ax_t_x.set_title("Displacement (seed traces)")
    ax_t_x.set_xlabel("Time [s]")
    ax_t_x.set_ylabel("x [m]")
    ax_t_x.grid(alpha=0.3)

    ax_t_v.set_title("Velocity (seed traces)")
    ax_t_v.set_xlabel("Time [s]")
    ax_t_v.set_ylabel("v [m/s]")
    ax_t_v.grid(alpha=0.3)

    # -------- bottom row PSDs: per-seed + mean --------
    if plot_seed_psd:
        # Welch seeds
        for i, (f, p) in enumerate(x_wel_specs):
            ax_p_x.loglog(f[1:], p[1:], color="C1", lw=1.0, alpha=0.5,
                          label="Welch seeds" if i == 0 else None)
        for i, (f, p) in enumerate(v_wel_specs):
            ax_p_v.loglog(f[1:], p[1:], color="C1", lw=1.0, alpha=0.5,
                          label="Welch seeds" if i == 0 else None)

        # FFT seeds (optional)
        if x_fft_specs is not None:
            for i, (f, p) in enumerate(x_fft_specs):
                ax_p_x.loglog(f[1:], p[1:], color="C0", lw=0.8, alpha=0.20,
                              label="FFT seeds" if i == 0 else None)
        if v_fft_specs is not None:
            for i, (f, p) in enumerate(v_fft_specs):
                ax_p_v.loglog(f[1:], p[1:], color="C0", lw=0.8, alpha=0.20,
                              label="FFT seeds" if i == 0 else None)

    # Mean lines on top
    if f_x_fft is not None and P_x_fft_mean is not None:
        ax_p_x.loglog(f_x_fft[1:], P_x_fft_mean[1:], "C0-", lw=2.5, label="FFT mean")
    ax_p_x.loglog(f_x_wel[1:], P_x_wel_mean[1:], "C1--", lw=2.5, label="Welch mean")
    ax_p_x.axvline(1.0 / Tp, linestyle="dashed", color="k")
    ax_p_x.set_title("Displacement PSD")
    ax_p_x.set_xlabel("Frequency [Hz]")
    ax_p_x.set_ylabel("PSD [m²/Hz]")
    ax_p_x.grid(alpha=0.3)
    ax_p_x.legend()

    if f_v_fft is not None and P_v_fft_mean is not None:
        ax_p_v.loglog(f_v_fft[1:], P_v_fft_mean[1:], "C0-", lw=2.2, label="FFT mean")
    ax_p_v.loglog(f_v_wel[1:], P_v_wel_mean[1:], "C1--", lw=2.2, label="Welch mean")
    ax_p_v.axvline(1.0 / Tp, linestyle="dashed", color="k")
    ax_p_v.set_title("Velocity PSD")
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
    cases,
    fetch_data,
    LF_frange=(1/300, 1/30),
    WF_frange=(1/30, 1/1),
    vel_method="savgol",
    do_fft_analysis=False,     # <<< key flag
    make_plots=True,           # <<< speed flag
    welch_nperseg=8193,        # <<< lower than 8192 for speed
    welch_noverlap=None,
    t_analyze_start=600,
    t_trace_start=950,
    out_csv="condition_summary_stats_all_cases.csv",
    fig_dir="fig_ensemble",
    # wind_speeds=(10.5, 5.0, 25.0),
    min_hs=1.0,
    min_tp=4.0,
):
    os.makedirs(fig_dir, exist_ok=True)
    summary_rows = []

    # Fast local references
    HUB = fetch_data.HUB
    load_outfile = fetch_data.load_outfile
    data_path = fetch_data.DATA_PATH

    # upfront filtering
    # cases = cases[cases["HWindSpeed"].isin(wind_speeds)]
    cases = cases[(cases["WaveHs"] > min_hs) & (cases["WaveTp"] > min_tp)]

    for (U, Hs, Tp), g in cases.groupby(["HWindSpeed", "WaveHs", "WaveTp"], sort=True):
        if g.empty:
            continue

        print(f"Selected case group: U={U}, Hs={Hs}, Tp={Tp}")

        # per-seed metric storage (lists; no inner DataFrame)
        metrics = {
            "x_wel_f_lf": [], "x_wel_A_lf": [],
            "v_wel_f_lf": [], "v_wel_A_lf": [],
            "x_wel_f_wf": [], "x_wel_A_wf": [],
            "v_wel_f_wf": [], "v_wel_A_wf": [],
            "Av_from_Ax_wf_wel": [],
            "x_wc_amp": [], "v_wc_amp": [],
        }

        if do_fft_analysis:
            metrics.update({
                "x_fft_f_lf": [], "x_fft_A_lf": [],
                "v_fft_f_lf": [], "v_fft_A_lf": [],
            })

        # only allocate plotting storage if needed
        if make_plots:
            t_list, x_list, v_list = [], [], []
            x_wel_specs, v_wel_specs = [], []
            if do_fft_analysis:
                x_fft_specs, v_fft_specs = [], []

        for row in g.itertuples(index=False):
            case_name = row.case_name
            file = os.path.join(data_path, "outfiles", f"{case_name}.out")
            data = load_outfile(file)

            t, x = build_displacement(data, HUB)

            # crop analysis window early to reduce work
            m = t > t_analyze_start
            t = t[m]
            x = x[m]

            if len(t) < 10:
                continue

            v = calc_velocity(t, x, method=vel_method)

            # Welch (always)
            fx_wel, Px_wel = welch_psd_density(t, x, nperseg=welch_nperseg, noverlap=welch_noverlap)
            fv_wel, Pv_wel = welch_psd_density(t, v, nperseg=welch_nperseg, noverlap=welch_noverlap)

            # Welch LF + WF metrics
            x_wel_f_lf, x_wel_A_lf = band_metrics(fx_wel, Px_wel, LF_frange)
            v_wel_f_lf, v_wel_A_lf = band_metrics(fv_wel, Pv_wel, LF_frange)
            x_wel_f_wf, x_wel_A_wf = band_metrics(fx_wel, Px_wel, WF_frange)
            v_wel_f_wf, v_wel_A_wf = band_metrics(fv_wel, Pv_wel, WF_frange)

            metrics["x_wel_f_lf"].append(x_wel_f_lf)
            metrics["x_wel_A_lf"].append(x_wel_A_lf)
            metrics["v_wel_f_lf"].append(v_wel_f_lf)
            metrics["v_wel_A_lf"].append(v_wel_A_lf)
            metrics["x_wel_f_wf"].append(x_wel_f_wf)
            metrics["x_wel_A_wf"].append(x_wel_A_wf)
            metrics["v_wel_f_wf"].append(v_wel_f_wf)
            metrics["v_wel_A_wf"].append(v_wel_A_wf)
            metrics["Av_from_Ax_wf_wel"].append(x_wel_A_wf * 2 * np.pi * x_wel_f_wf)

            # worst case
            metrics["x_wc_amp"].append(worst_case_amplitude(t, x, 800, 1000))
            metrics["v_wc_amp"].append(worst_case_amplitude(t, v, 800, 1000))

            # optional FFT LF
            if do_fft_analysis:
                fx_fft, Px_fft = fft_psd_density(t, x)
                fv_fft, Pv_fft = fft_psd_density(t, v)

                x_fft_f_lf, x_fft_A_lf = band_metrics(fx_fft, Px_fft, LF_frange)
                v_fft_f_lf, v_fft_A_lf = band_metrics(fv_fft, Pv_fft, LF_frange)

                metrics["x_fft_f_lf"].append(x_fft_f_lf)
                metrics["x_fft_A_lf"].append(x_fft_A_lf)
                metrics["v_fft_f_lf"].append(v_fft_f_lf)
                metrics["v_fft_A_lf"].append(v_fft_A_lf)

            # optional plotting storage
            if make_plots:
                mp = t >= t_trace_start
                t_list.append(t[mp])
                x_list.append(x[mp])
                v_list.append(v[mp])

                x_wel_specs.append((fx_wel, Px_wel))
                v_wel_specs.append((fv_wel, Pv_wel))
                if do_fft_analysis:
                    x_fft_specs.append((fx_fft, Px_fft))
                    v_fft_specs.append((fv_fft, Pv_fft))

        n_used = len(metrics["x_wel_f_wf"])
        if n_used == 0:
            continue

        row_out = {
            "HWindSpeed": U,
            "WaveHs": Hs,
            "WaveTp": Tp,
            "n_seeds": n_used,
        }

        # aggregate Welch stats
        welch_map = {
            "x_wel_ens_f_lf": "x_wel_f_lf",
            "x_wel_ens_A_lf": "x_wel_A_lf",
            "v_wel_ens_f_lf": "v_wel_f_lf",
            "v_wel_ens_A_lf": "v_wel_A_lf",
            "x_wel_ens_f_wf": "x_wel_f_wf",
            "x_wel_ens_A_wf": "x_wel_A_wf",
            "v_wel_ens_f_wf": "v_wel_f_wf",
            "v_wel_ens_A_wf": "v_wel_A_wf",
            "Av_from_Ax_wf_wel_ens": "Av_from_Ax_wf_wel",
        }
        for out_col, src in welch_map.items():
            mu, sd = nanmean_std(metrics[src])
            row_out[out_col] = mu
            row_out[out_col + "_std"] = sd

        # aggregate worst-case
        row_out["x_wc_amp_mean"], row_out["x_wc_amp_std"] = nanmean_std(metrics["x_wc_amp"])
        row_out["v_wc_amp_mean"], row_out["v_wc_amp_std"] = nanmean_std(metrics["v_wc_amp"])

        # aggregate FFT LF stats or fill NaN
        fft_cols = [
            "x_fft_ens_f_lf", "x_fft_ens_A_lf",
            "v_fft_ens_f_lf", "v_fft_ens_A_lf",
        ]
        if do_fft_analysis:
            fft_map = {
                "x_fft_ens_f_lf": "x_fft_f_lf",
                "x_fft_ens_A_lf": "x_fft_A_lf",
                "v_fft_ens_f_lf": "v_fft_f_lf",
                "v_fft_ens_A_lf": "v_fft_A_lf",
            }
            for out_col, src in fft_map.items():
                mu, sd = nanmean_std(metrics[src])
                row_out[out_col] = mu
                row_out[out_col + "_std"] = sd
        else:
            for c in fft_cols:
                row_out[c] = np.nan
                row_out[c + "_std"] = np.nan

        summary_rows.append(row_out)

        # optional plots
        if make_plots:
            fx_wel_e, Px_wel_mean = stack_psd_mean(x_wel_specs)
            fv_wel_e, Pv_wel_mean = stack_psd_mean(v_wel_specs)

            fx_fft_e = Px_fft_mean = fv_fft_e = Pv_fft_mean = None
            if do_fft_analysis:
                fx_fft_e, Px_fft_mean = stack_psd_mean(x_fft_specs)
                fv_fft_e, Pv_fft_mean = stack_psd_mean(v_fft_specs)

            title = f"HWindSpeed={U}, WaveHs={Hs}, WaveTp={Tp}, n={n_used}"
            out_png = os.path.join(fig_dir, f"U{U}_Hs{Hs}_Tp{Tp}.png")

            plot_condition_2x2(
                t_list, x_list, v_list, Tp,
                x_wel_specs, v_wel_specs,
                fx_wel_e, Px_wel_mean,
                fv_wel_e, Pv_wel_mean,
                title, out_png,
                x_fft_specs=x_fft_specs if do_fft_analysis else None,
                v_fft_specs=v_fft_specs if do_fft_analysis else None,
                f_x_fft=fx_fft_e, P_x_fft_mean=Px_fft_mean,
                f_v_fft=fv_fft_e, P_v_fft_mean=Pv_fft_mean,
                plot_seed_psd=True,
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_csv, index=False)
    return summary_df


if __name__ == "__main__":
    cases = fetch_data.get_cases()

    summary_df = run_analysis(
        cases=cases,
        fetch_data=fetch_data,
        LF_frange=(1/300, 1/30),
        WF_frange=(1/30, 1/1),
        vel_method="savgol",        # "gradient" is faster if you want
        do_fft_analysis=False,      # <<< Welch-only focus
        make_plots=True,            # set False for max speed
        welch_nperseg=4096*2,
        welch_noverlap=None,
        t_analyze_start=600,
        t_trace_start=950,
        out_csv="condition_summary_stats_all_cases.csv",
        fig_dir="fig_ensemble",
    )

    print(summary_df.head())