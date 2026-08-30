import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ssig
import fetch_data
from UnifiedMomentumModel import Momentum as UMM
import warnings
import itertools

# ----------------------------
# Signal helpers
# ----------------------------
def build_displacement(data, hub):
    t = np.asarray(data["Time"], dtype=float)
    pitch_deg = np.asarray(data["PtfmPitch"], dtype=float)
    surge = np.asarray(data["PtfmSurge"], dtype=float)
    x = surge + hub * np.sin(np.deg2rad(pitch_deg))
    return t, x


def calc_velocity(t, x, method="savgol", window=201, poly=3):
    dt = float(np.mean(np.diff(t)))

    if method == "gradient":
        return np.gradient(x, dt, edge_order=2)

    elif method == "savgol":
        w = min(window, len(x) - (1 - len(x) % 2))  # odd <= len(x)
        if w < 5:
            warnings.warn(
                f"Window too small for savgol with w='{w}'. Defaulting to 'gradient'.",
                UserWarning,
                stacklevel=2
            )
            return np.gradient(x, dt, edge_order=2)

        if w % 2 == 0:
            w -= 1

        p = min(poly, w - 2)
        return ssig.savgol_filter(
            x, window_length=w, polyorder=p, deriv=1, delta=dt, mode="interp"
        )

    else:
        warnings.warn(
            f"Unknown method '{method}'. Defaulting to 'gradient'.",
            UserWarning,
            stacklevel=2
        )
        return np.gradient(x, dt, edge_order=2)

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
    n = a.size
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return float(a[0]), np.nan, 1
    return float(np.mean(a)), float(np.std(a, ddof=1)), n

def pooled_nanmean_std(series_list):
    """
    series_list: list of 1D arrays (one per seed)
    returns pooled mean/std over all finite points from all seeds+time
    """
    if len(series_list) == 0:
        return np.nan, np.nan, 0

    y = np.concatenate([
        np.asarray(s, dtype=float).ravel()
        for s in series_list
        if s is not None and len(s) > 0
    ])
    return nanmean_std(y)

def stack_psd_mean(spec_list):
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
# CT helpers
# ----------------------------
def compute_ct(Fxh, rho, A, Uref, yaw, tilt):
    den = 0.5 * rho * A * (Uref * np.cos(yaw) * np.cos(tilt))**2
    if Uref <= 0:
        return np.full_like(Fxh, np.nan, dtype=float)
    return Fxh / den

def compute_ct_prime(model_Ct, Ct, yaw, tilt):
    Ct = np.asarray(Ct, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    tilt = np.asarray(tilt, dtype=float)
    sol = model_Ct(Ct, yaw = yaw, tilt=tilt)
    return sol.Ctprime

# ----------------------------
# Plot helpers
# ----------------------------
def plot_condition_2x2(
    t_list, x_list, v_list, Tp,
    x_specs, v_specs,
    f_x, P_x_mean,
    f_v, P_v_mean,
    title, out_png,
    plot_seed_psd=True
):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_t_x, ax_t_v = axs[0, 0], axs[0, 1]
    ax_p_x, ax_p_v = axs[1, 0], axs[1, 1]

    # top row traces
    for i, (t, x) in enumerate(zip(t_list, x_list)):
        ax_t_x.plot(t, x, lw=1.2, label=f"Seed {i+1}")
    for i, (t, v) in enumerate(zip(t_list, v_list)):
        ax_t_v.plot(t, v, lw=1.2, label=f"Seed {i+1}")

    ax_t_x.set_title("Displacement (seed traces)")
    ax_t_x.set_xlabel("Time [s]")
    ax_t_x.set_ylabel("x [m]")
    ax_t_x.grid(alpha=0.3)

    ax_t_v.set_title("Velocity (seed traces)")
    ax_t_v.set_xlabel("Time [s]")
    ax_t_v.set_ylabel("v [m/s]")
    ax_t_v.grid(alpha=0.3)

    # bottom row PSDs
    if plot_seed_psd:
        nseed = len(x_specs)
        cmap = plt.get_cmap("tab20", max(nseed, 1))
        for i, ((fx, px), (fv, pv)) in enumerate(zip(x_specs, v_specs)):
            c = cmap(i)
            ax_p_x.loglog(fx[1:], px[1:], color=c, lw=0.9, alpha=0.65)
            ax_p_v.loglog(fv[1:], pv[1:], color=c, lw=0.9, alpha=0.65)


    # means
    ax_p_x.loglog(f_x[1:], P_x_mean[1:], "k-", lw=2.4, label="Welch mean")
    ax_p_x.axvline(1.0 / Tp, linestyle="dashed", color="k")
    ax_p_x.set_title("Displacement PSD")
    ax_p_x.set_xlabel("Frequency [Hz]")
    ax_p_x.set_ylabel("PSD [m²/Hz]")
    ax_p_x.grid(alpha=0.3)
    ax_p_x.legend()

    ax_p_v.loglog(f_v[1:], P_v_mean[1:], "k-", lw=2.4, label="Welch mean")
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


def plot_ct_timeseries(
    t_list, ct_list, ctp_list, ct_ref_list, title, out_png
):
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax_ct, ax_ctp, ax_ctref = axs
    cmap = plt.get_cmap("tab20", max(len(t_list), 1))

    for i, (t, ct, ctp, ctref) in enumerate(zip(t_list, ct_list, ctp_list, ct_ref_list)):
        c = cmap(i)
        ax_ct.plot(t, ct, color=c, lw=1.1, alpha=0.85, label=f"Seed {i+1}")
        ax_ctp.plot(t, ctp, color=c, lw=1.1, alpha=0.85)
        ax_ctref.plot(t, ctref, color=c, lw=1.1, alpha=0.85)

    ax_ct.set_title("CT(t)")
    ax_ct.set_ylabel("CT [-]")
    ax_ct.grid(alpha=0.3)
    ax_ct.legend(ncol=2, fontsize=9)

    ax_ctp.set_title("CT'(t)")
    ax_ctp.set_ylabel("CT' [-]")
    ax_ctp.grid(alpha=0.3)

    ax_ctref.set_title("RtFldCt(t)")
    ax_ctref.set_ylabel("RtFldCt [-]")
    ax_ctref.set_xlabel("Time [s]")
    ax_ctref.grid(alpha=0.3)

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
    min_hs=1.0,
    min_tp=4.0,
    detrend = True,
    WF_frange=(1/30, 1/1),
    vel_method="savgol",
    sg_window=201, sg_poly=3,
    welch_nperseg=8192, welch_overlap_frac=0.5,
    do_fft_analysis=False,          # controls LF FFT analysis only
    do_ct_calcs = True,
    t_analyze_start=600,
    t_trace_start=950,
    out_csv="condition_summary_stats_all_cases.csv",
    ct_out_csv="ct_condition_summary_stats_all_cases.csv",
    make_psd_plots = False,
    make_ct_plots = False,
    psd_fig_dir="fig_ensemble",
    ct_fig_dir="fig_ct_timeseries",
):
    rho=1.225
    D=240.0
    model_Ct = UMM.ThrustBasedUnified()

    summary_df = pd.DataFrame()
    ct_summary_df = pd.DataFrame()

    if make_psd_plots:
        os.makedirs(psd_fig_dir, exist_ok=True)
    if make_ct_plots:
        os.makedirs(ct_fig_dir, exist_ok=True)

    summary_rows = []
    ct_rows = []

    HUB = fetch_data.HUB
    load_outfile = fetch_data.load_outfile
    data_path = fetch_data.DATA_PATH
    A = np.pi * (D ** 2) / 4.0

    cases = cases[(cases["WaveHs"] > min_hs) & (cases["WaveTp"] > min_tp)]
    cases = cases[(cases["HWindSpeed"] == 3) | (cases["HWindSpeed"] == 10.5) | (cases["HWindSpeed"] == 25)]

    for (U, Hs, Tp), g in cases.groupby(["HWindSpeed", "WaveHs", "WaveTp"], sort=True):
        if g.empty:
            continue

        print(f"Selected case group: U={U}, Hs={Hs}, Tp={Tp}")

        # per-seed scalar metrics
        metrics = {
            "x_f": [], "x_A": [],
            "v_f": [], "v_A": [],
            "v_A_from_x_A": [],
            "x_A_wc": [], "v_A_wc": [],
        }

        # pooled CT arrays (across seeds/time)
        ct_all = []
        ctp_all = []
        ctref_all = []

        # plotting storage
        if make_psd_plots:
            t_list, x_list, v_list = [], [], []
            x_specs, v_specs = [], []

        if make_ct_plots:
            t_ct_list, ct_list_plot, ctp_list_plot, ctref_list_plot = [], [], [], []

        for row in g.itertuples(index=False):
            case_name = row.case_name
            file = os.path.join(data_path, "outfiles", f"{case_name}.out")
            data = load_outfile(file)

            # displacement and velocity
            t_all, x_all = build_displacement(data, HUB)
            if detrend:
                x_all = ssig.detrend(x_all, type='linear')

            v_all = calc_velocity(t_all, x_all, method=vel_method, window=sg_window, poly=sg_poly)

            m = t_all > t_analyze_start
            t = t_all[m]
            x = x_all[m]
            v = v_all[m]

            # power spectral densities
            if do_fft_analysis: # FFT
                fx, Px = fft_psd_density(t, x)
                fv, Pv = fft_psd_density(t, v)
            else: # Welch
                nperseg_eff = min(int(welch_nperseg), len(x))
                noverlap_eff = int(welch_overlap_frac * nperseg_eff)
                noverlap_eff = min(noverlap_eff, nperseg_eff - 1)
                fx, Px = welch_psd_density(t, x, nperseg=welch_nperseg, noverlap=noverlap_eff)
                fv, Pv = welch_psd_density(t, v, nperseg=welch_nperseg, noverlap=noverlap_eff)

            x_f, x_A = band_metrics(fx, Px, WF_frange)
            v_f, v_A = band_metrics(fv, Pv, WF_frange)

            metrics["x_f"].append(x_f)
            metrics["x_A"].append(x_A)
            metrics["v_f"].append(v_f)
            metrics["v_A"].append(v_A)
            metrics["v_A_from_x_A"].append(x_A * 2 * np.pi * x_f)
            
            # worst-case amplitudes
            metrics["x_A_wc"].append(worst_case_amplitude(t, x, 800, 1000))
            metrics["v_A_wc"].append(worst_case_amplitude(t, v, 800, 1000))
            

            # ---- CT/CT' time series ----
            if do_ct_calcs:
                for fld in ("RtFldCt", "RtFldFxh", "RtVAvgxh"):
                    if fld not in data:
                        raise KeyError(f"Required field '{fld}' not found in outfile data.")

                Uref=float(U)
                Fxh = data["RtFldFxh"].to_numpy(dtype=float)[m]
                Ct_ref = data["RtFldCt"].to_numpy(dtype=float)[m]

                yaw_deg = data["PtfmYaw"].to_numpy(dtype=float)[m]
                tilt_deg = data["PtfmPitch"].to_numpy(dtype=float)[m]
                yaw_rad = np.deg2rad(yaw_deg)
                tilt_rad = np.deg2rad(tilt_deg)

                Ct = compute_ct(Fxh, rho=rho, A=A, Uref=Uref, yaw = yaw_rad, tilt = tilt_rad)
                Ct_p = compute_ct_prime(model_Ct, Ct_ref, yaw_rad, tilt_rad)

                ct_all.append(Ct)
                ctp_all.append(Ct_p)
                ctref_all.append(Ct_ref)

            # plotting storage
            if make_psd_plots:
                mp = t >= t_trace_start
                t_list.append(t[mp])
                x_list.append(x[mp])
                v_list.append(v[mp])

                x_specs.append((fx, Px))
                v_specs.append((fv, Pv))

            if do_ct_calcs and make_ct_plots:
                mp = t >= t_trace_start
                t_ct_list.append(t[mp])
                ct_list_plot.append(Ct[mp])
                ctp_list_plot.append(Ct_p[mp])
                ctref_list_plot.append(Ct_ref[mp])

        n_used = len(metrics["x_f"])
        if n_used == 0:
            continue

        # ---- spectral/amp summary ----
        row_out = {
            "HWindSpeed": U,
            "WaveHs": Hs,
            "WaveTp": Tp,
            "n_seeds": n_used,
        }

        # WF Welch aggregates
        psd_map = {
            "x_f_ens": "x_f",
            "x_A_ens": "x_A",
            "v_f_ens": "v_f",
            "v_A_ens": "v_A",
            "v_A_from_x_A_ens": "v_A_from_x_A",
        }
        for out_col, src in psd_map.items():
            mu, sd, _ = nanmean_std(metrics[src])
            row_out[out_col + "_mean"] = mu
            row_out[out_col + "_std"] = sd

        row_out["x_A_wc_ens_mean"], row_out["x_A_wc_ens_std"], _  = nanmean_std(metrics["x_A_wc"])
        row_out["v_A_wc_ens_mean"], row_out["v_A_wc_ens_std"], _ = nanmean_std(metrics["v_A_wc"])

        summary_rows.append(row_out)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_csv, index=False)

        # ---- CT pooled stats across seeds+time ----
        if do_ct_calcs:
            ct_mean, ct_std, ct_n = pooled_nanmean_std(ct_all)
            ctp_mean, ctp_std, ctp_n = pooled_nanmean_std(ctp_all)
            ctref_mean, ctref_std, ctref_n = pooled_nanmean_std(ctref_all)

            ct_rows.append({
                "HWindSpeed": U,
                "WaveHs": Hs,
                "WaveTp": Tp,
                "n_seeds": n_used,
                "CT_ens_mean": ct_mean,
                "CT_ens_std": ct_std,
                "CT_ens_n": ct_n,
                "CTp_ens_mean": ctp_mean,
                "CTp_ens_std": ctp_std,
                "CTp_ens_n": ctp_n,
                "RtFldCt_ens_mean": ctref_mean,
                "RtFldCt_ens_std": ctref_std,
                "RtFldCt_ens_n": ctref_n,
            })

            ct_summary_df = pd.DataFrame(ct_rows)
            ct_summary_df.to_csv(ct_out_csv, index=False)

        # ---- PSD plots ----
        if make_psd_plots:
            fx_e, Px_mean = stack_psd_mean(x_specs)
            fv_e, Pv_mean = stack_psd_mean(v_specs)

            title = f"HWindSpeed={U}, WaveHs={Hs}, WaveTp={Tp}, n={n_used}"
            out_png = os.path.join(psd_fig_dir, f"U{U}_Hs{Hs}_Tp{Tp}.png")

            plot_condition_2x2(
                t_list, x_list, v_list, Tp,
                x_specs, v_specs,
                fx_e, Px_mean,
                fv_e, Pv_mean,
                title, out_png,
                plot_seed_psd=True,
            )

        # ---- CT plots ----
        if do_ct_calcs and make_ct_plots:
            ct_title = f"CT time series | U={U}, Hs={Hs}, Tp={Tp}, n={n_used}"
            ct_png = os.path.join(ct_fig_dir, f"CT_U{U}_Hs{Hs}_Tp{Tp}.png")
            plot_ct_timeseries(
                t_ct_list, ct_list_plot, ctp_list_plot, ctref_list_plot,
                ct_title, ct_png
            )

    return summary_df, ct_summary_df

def pct_change(new, base):
    if not np.isfinite(new) or not np.isfinite(base) or base == 0:
        return np.nan
    return 100.0 * (new - base) / base

def assess_sensitivity(all_df, baseline_cfg):
    """
    baseline_cfg keys:
      detrend_mode, vel_method, sg_window, sg_poly, welch_nperseg, welch_overlap_frac
    """
    key_cols = ["HWindSpeed", "WaveHs", "WaveTp"]

    cfg_cols = [
        "detrend_mode", "vel_method", "sg_window", "sg_poly",
        "welch_nperseg", "welch_overlap_frac"
    ]

    metric_cols = [
        "x_f_ens_mean", "x_A_ens_mean",
        "v_f_ens_mean", "v_A_ens_mean",
        "v_A_from_x_A_ens_mean",
        "x_A_wc_ens_mean", "v_A_wc_ens_mean"
    ]

    # baseline subset
    m_base = np.ones(len(all_df), dtype=bool)
    for k, v in baseline_cfg.items():
        if pd.isna(v):
            m_base &= all_df[k].isna()
        else:
            m_base &= (all_df[k] == v)

    base_df = all_df.loc[m_base, key_cols + metric_cols].copy()
    base_df = base_df.rename(columns={c: c + "_base" for c in metric_cols})

    merged = all_df.merge(base_df, on=key_cols, how="left")

    for c in metric_cols:
        merged[c + "_pct_change"] = [
            pct_change(n, b) for n, b in zip(merged[c].to_numpy(), merged[c + "_base"].to_numpy())
        ]

    # aggregate sensitivity by config (median abs % change across conditions)
    abschg_cols = [c + "_pct_change" for c in metric_cols]
    tmp = merged.copy()
    for c in abschg_cols:
        tmp[c] = np.abs(tmp[c])

    agg = tmp.groupby(cfg_cols, dropna=False)[abschg_cols].median().reset_index()

    # simple scalar rank score
    agg["sensitivity_score"] = agg[abschg_cols].mean(axis=1, skipna=True)
    agg = agg.sort_values("sensitivity_score")

    return merged, agg


if __name__ == "__main__":
    cases = fetch_data.get_cases()

    summary_df, ct_summary_df = run_analysis(
        cases=cases,
        fetch_data=fetch_data,
        min_hs=1.0,
        min_tp=4.0,
        detrend = False,
        WF_frange=(1/30, 1/1),
        vel_method="savgol",
        sg_window=201, sg_poly=3,
        do_fft_analysis=False, do_ct_calcs = True,
        welch_nperseg=4096 * 2, welch_overlap_frac=0.5,
        t_analyze_start=600,
        t_trace_start=950,
        out_csv="condition_summary_stats_all_cases.csv",
        ct_out_csv="ct_condition_summary_stats_all_cases.csv",
        make_psd_plots = False,
        make_ct_plots = False,
        psd_fig_dir="fig_ensemble",
        ct_fig_dir="fig_ct_timeseries",
    )

    print(summary_df.head())
    print(ct_summary_df.head())
