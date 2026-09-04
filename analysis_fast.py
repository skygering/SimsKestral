import os
import numpy as np
import pandas as pd
import scipy.signal as ssig
import warnings
from tqdm import tqdm
from UnifiedMomentumModel import Momentum as UMM
from collections import defaultdict
import fetch_data
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from typing import Optional
from time import sleep
from ct_fast import (
    CTPrimeCache, compute_ct, compute_ct_prime,
    eval_ctprime_cached_array,
    analyze_ct_ctprime_for_seed,
    reduce_ct_phase_ensemble, reduce_ct_seed_ensemble,
    save_phase_debug_plots
)
import pickle
import time

# =========================================
# U-BATCH ANALYSIS VERSION
# Key ideas:
# 1) preload all cases for ONE U at a time
# 2) run all configs against that preload
# 3) keep memory bounded to one-U slice
# 4) write outputs once in driver
# =========================================

@dataclass(frozen=True)
class AnalysisCfg:
    detrend: bool = False
    vel_method: str = "savgol"      # "savgol" or "gradient"
    sg_window: Optional[int] = 301
    sg_poly: Optional[int] = 4
    welch_nperseg: int = 3000
    welch_overlap_frac: float = 0.5
    do_fft_analysis: bool = False
    tstart: int = 600
    f1: float = 1/30
    f2: float = 1.0
    psd_plot_mode: str = "mean"     # "mean" or "seed"
    include_vel_in_plot: bool = False
    ct_cache_decimals: int = 5
    phase_bins: int = 10
    min_cycles_for_phase: int = 8
    peakprominence_scale: float = 0.25
    peakmin_separation_frac: float = 0.5
    cycle_period_tol_frac: float = 0.35
    save_phase_debug: bool = False

    def __post_init__(self):
        vm = self.vel_method.lower()
        pm = self.psd_plot_mode.lower()

        object.__setattr__(self, "vel_method", vm)
        object.__setattr__(self, "psd_plot_mode", pm)

        if vm not in {"savgol", "gradient"}:
            raise ValueError("vel_method must be 'savgol' or 'gradient'")
        if pm not in {"mean", "seed"}:
            raise ValueError("psd_plot_mode must be 'mean' or 'seed'")

        if self.welch_nperseg < 8:
            raise ValueError("welch_nperseg must be >= 8")
        if not (0 <= self.welch_overlap_frac < 1):
            raise ValueError("welch_overlap_frac must be in [0, 1)")

        if vm == "savgol":
            if self.sg_window is None or self.sg_poly is None:
                raise ValueError("sg_window and sg_poly required for savgol")
            if self.sg_window < 5:
                raise ValueError("sg_window should be >= 5")
            if self.sg_poly < 1:
                raise ValueError("sg_poly should be >= 1")


# ----------------------------
# Signal helpers
# ----------------------------
def build_displacement(data, hub):
    t = np.asarray(data["Time"], dtype=float)
    pitch_deg = np.asarray(data["PtfmPitch"], dtype=float)
    # pitch = 0
    surge = np.asarray(data["PtfmSurge"], dtype=float)
    # surge = 0
    x = surge + hub * np.sin(np.deg2rad(pitch_deg))
    return t, x


def calc_velocity(t, x, method="savgol", window=201, poly=3):
    dt = float(np.mean(np.diff(t)))

    if method == "gradient":
        return np.gradient(x, dt, edge_order=2)

    if method == "savgol":
        w = min(int(window), len(x) - (1 - len(x) % 2))
        if w < 5:
            warnings.warn(f"Window too small for savgol (w={w}); using gradient.")
            return np.gradient(x, dt, edge_order=2)
        if w % 2 == 0:
            w -= 1
        p = min(int(poly), w - 2)
        if p < 1:
            warnings.warn(f"Invalid savgol poly after clipping (p={p}); using gradient.")
            return np.gradient(x, dt, edge_order=2)

        return ssig.savgol_filter(
            x, window_length=w, polyorder=p, deriv=1, delta=dt, mode="interp"
        )

    warnings.warn(f"Unknown method '{method}'; using gradient.")
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


def welch_psd_density(t, x, nperseg=8192, overlap_frac=0.5):
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt

    nperseg_eff = min(int(nperseg), len(x))
    if nperseg_eff < 8:
        return np.array([np.nan]), np.array([np.nan])

    noverlap_eff = int(float(overlap_frac) * nperseg_eff)
    noverlap_eff = min(noverlap_eff, nperseg_eff - 1)

    # NOTE: no detrend here (per your preference)
    f, psd = ssig.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        scaling="density",
        detrend = "constant"
    )
    return f, psd


def band_metrics(f, Pxx, frange):
    m = (f >= frange[0]) & (f <= frange[1])
    if np.sum(m) < 2:
        return np.nan, np.nan
    fb = f[m]
    pb = Pxx[m]
    fpk = float(fb[np.argmax(pb)])
    var_band = float(np.trapezoid(pb, fb))
    Aeq = np.sqrt(2.0) * np.sqrt(max(var_band, 0.0))
    return fpk, Aeq


# def worst_case_amplitude(t, x, t_start=800, t_end=1000):
#     m = (t >= t_start) & (t <= t_end)
#     if np.sum(m) < 2:
#         return np.nan
#     xx = x[m]
#     return (np.max(xx) - np.min(xx)) / 2

def worst_case_amplitude(t, x, t_start=800, t_end=1000):
    m = (t >= t_start) & (t <= t_end)
    if np.sum(m) < 2:
        return np.nan
    xx = x[m]
    mean_x = np.mean(xx)
    return np.max(np.abs(xx - mean_x))


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
    if len(series_list) == 0:
        return np.nan, np.nan, 0
    y = np.concatenate(
        [np.asarray(s, dtype=float).ravel() for s in series_list if s is not None and len(s) > 0]
    )
    return nanmean_std(y)

# ----------------------------
# Preload once per U
# ----------------------------
def preload_cases_for_u(
    cases_u,
    fetch_data,
    t_analyze_start=600,
    do_ct_calcs=False,
):
    """
    Load each outfile once for this U-slice.
    Store raw (post-time-window) arrays; detrending is applied per-config later.
    """
    HUB = fetch_data.HUB
    load_outfile = fetch_data.load_outfile
    data_path = fetch_data.DATA_PATH

    cache = {}
    for row in tqdm(cases_u.itertuples(index=False), total=len(cases_u), desc="Preloading U-slice"):
        case_name = row.case_name
        file = os.path.join(data_path, "outfiles", f"{case_name}.out")
        data = load_outfile(file)

        t_all, x_all = build_displacement(data, HUB)
        m = t_all > t_analyze_start

        item = {
            "t": t_all[m],
            "x_raw": x_all[m],
        }

        if do_ct_calcs:
            item["Fxh"] = data["RtFldFxh"].to_numpy(dtype=float)[m]
            # item["Ct_ref"] = data["RtFldCt"].to_numpy(dtype=float)[m]
            item["yaw"] = np.deg2rad(data["PtfmYaw"].to_numpy(dtype=float)[m])
            item["tilt"] = np.deg2rad(data["PtfmPitch"].to_numpy(dtype=float)[m])

        cache[case_name] = item

    return cache

# ----------------------------
# Run one config on one U-slice cache
# ----------------------------
def run_one_config_on_u_cache(
    cases_u,
    u_cache,
    cfg,
    do_ct_calcs=False,
    collect_traces=False,
):
    rho = 1.225
    D = 240.0
    A = np.pi * (D ** 2) / 4.0

    model_Ct = UMM.ThrustBasedUnified() if do_ct_calcs else None
    ctp_cache = (
        CTPrimeCache(model_Ct, ndigits=getattr(cfg, "ct_cache_decimals", 5))
        if do_ct_calcs else None
    )

    summary_rows = []
    ct_rows = []
    ct_seed_rows = []
    ct_phase_rows = []

    trace_payload = defaultdict(lambda: {"psd": [], "savgol_vel": []})

    for (U, Hs, Tp), g in cases_u.groupby(["HWindSpeed", "WaveHs", "WaveTp"], sort=True):
        metrics = {
            "x_f": [], "x_A": [],
            "v_f": [], "v_A": [],
            "v_A_from_x_A": [],
            "x_A_wc": [], "v_A_wc": [],
        }

        ct_all, ctp_all = [], []

        for row in g.itertuples(index=False):
            case_name = row.case_name
            item = u_cache[case_name]
            t = item["t"]
            x = item["x_raw"]

            if cfg.detrend:
                x = ssig.detrend(x, type="linear")

            v = calc_velocity(
                t, x,
                method=cfg.vel_method,
                window=cfg.sg_window if cfg.sg_window is not None else 201,
                poly=cfg.sg_poly if cfg.sg_poly is not None else 3,
            )

            # edge trim after velocity calc
            m2 = t > cfg.tstart
            x, v, t = x[m2], v[m2], t[m2]

            if cfg.do_fft_analysis:
                fx, Px = fft_psd_density(t, x)
                fv, Pv = fft_psd_density(t, v)
            else:
                fx, Px = welch_psd_density(
                    t, x,
                    nperseg=cfg.welch_nperseg,
                    overlap_frac=cfg.welch_overlap_frac,
                )
                fv, Pv = welch_psd_density(
                    t, v,
                    nperseg=cfg.welch_nperseg,
                    overlap_frac=cfg.welch_overlap_frac,
                )

            # band-limited metrics (for amplitudes/frequencies)
            x_f, x_A = band_metrics(fx, Px, (cfg.f1, cfg.f2))
            v_f, v_A = band_metrics(fv, Pv, (cfg.f1, cfg.f2))

            metrics["x_f"].append(x_f)
            metrics["x_A"].append(x_A)
            metrics["v_f"].append(v_f)
            metrics["v_A"].append(v_A)
            metrics["v_A_from_x_A"].append(
                x_A * 2 * np.pi * x_f if np.isfinite(x_A) and np.isfinite(x_f) else np.nan
            )
            metrics["x_A_wc"].append(worst_case_amplitude(t, x, 800, 1000))
            metrics["v_A_wc"].append(worst_case_amplitude(t, v, 800, 1000))

            if do_ct_calcs:
                Fxh = item["Fxh"]
                yaw = item["yaw"]
                tilt = item["tilt"]

                # Your CT
                Ct = compute_ct(Fxh, rho=rho, A=A, Uref=float(U), yaw=yaw, tilt=tilt)

                # CT' via cache (rounded keys)
                Ct_p = eval_ctprime_cached_array(
                    ctp_cache, Ct, yaw, tilt, compute_ct_prime
                )

                ct_all.append(Ct)
                ctp_all.append(Ct_p)

                # optional aux series for phase debug plotting:
                # use displacement x (already trimmed/aligned) to verify cycle/phase ID
                aux = {"v": v}

                seed_row, phase_rows, phase_debug = analyze_ct_ctprime_for_seed(
                    case_name=case_name,
                    t=t,
                    v=v,       # phase basis = velocity
                    ct=Ct,
                    ctp=Ct_p,
                    f_est=v_f,
                    U=U, Hs=Hs, Tp=Tp,
                    n_bins=getattr(cfg, "phase_bins", 10),
                    # min_cycles_for_phase=getattr(cfg, "min_cycles_for_phase", 8),
                    # prominence_scale=getattr(cfg, "peakprominence_scale", 0.25),
                    # min_sep_frac=getattr(cfg, "peakmin_separation_frac", 0.5),
                    # period_tol_frac=getattr(cfg, "cycle_period_tol_frac", 0.35),
                    aux_series=aux,
                    return_phase_debug=getattr(cfg, "save_phase_debug", False),
                )

                ct_seed_rows.append(seed_row)
                ct_phase_rows.extend(phase_rows)

                if getattr(cfg, "save_phase_debug", False) and phase_debug is not None:
                    sea_key = (float(U), float(Hs), float(Tp))
                    trace_payload[sea_key].setdefault("phase_debug", [])
                    trace_payload[sea_key]["phase_debug"].append((case_name, phase_debug))

            if collect_traces:
                sea_key = (float(U), float(Hs), float(Tp))
                trace_payload[sea_key].setdefault("psd", [])
                trace_payload[sea_key].setdefault("savgol_vel", [])

                good_psd = np.isfinite(fx) & np.isfinite(Px)
                if np.sum(good_psd) >= 2:
                    trace_payload[sea_key]["psd"].append(
                        (case_name, np.asarray(fx)[good_psd], np.asarray(Px)[good_psd])
                    )

                good_v = np.isfinite(t) & np.isfinite(v)
                if np.sum(good_v) >= 2:
                    trace_payload[sea_key]["savgol_vel"].append(
                        (case_name, np.asarray(t)[good_v], np.asarray(v)[good_v])
                    )

        n_used = len(metrics["x_f"])
        if n_used == 0:
            continue

        # motion summary
        row_out = {"HWindSpeed": U, "WaveHs": Hs, "WaveTp": Tp, "n_seeds": n_used}
        for out_col, src in {
            "x_f_ens": "x_f",
            "x_A_ens": "x_A",
            "v_f_ens": "v_f",
            "v_A_ens": "v_A",
            "v_A_from_x_A_ens": "v_A_from_x_A",
            "x_A_wc_ens": "x_A_wc",
            "v_A_wc_ens": "v_A_wc",
        }.items():
            mu, sd, _ = nanmean_std(metrics[src])
            row_out[out_col + "_mean"] = mu
            row_out[out_col + "_std"] = sd
        summary_rows.append(row_out)

        # existing pooled CT summary
        if do_ct_calcs:
            ct_mean, ct_std, ct_n = pooled_nanmean_std(ct_all)
            ctp_mean, ctp_std, ctp_n = pooled_nanmean_std(ctp_all)

            ct_rows.append({
                "HWindSpeed": U, "WaveHs": Hs, "WaveTp": Tp, "n_seeds": n_used,
                "CT_ens_mean": ct_mean, "CT_ens_std": ct_std, "CT_ens_n": ct_n,
                "CTp_ens_mean": ctp_mean, "CTp_ens_std": ctp_std, "CTp_ens_n": ctp_n,
            })

    summary_df = pd.DataFrame(summary_rows)
    ct_df = pd.DataFrame(ct_rows)
    ct_seed_df = pd.DataFrame(ct_seed_rows)
    ct_phase_df = pd.DataFrame(ct_phase_rows)

    print(f"CT' cache size={len(ctp_cache._cache)} hits={ctp_cache.hits} misses={ctp_cache.misses}")

    return summary_df, ct_df, dict(trace_payload), ct_seed_df, ct_phase_df

def merge_trace_payloads(payload_list):
    merged = defaultdict(lambda: {"psd": [], "savgol_vel": [], "phase_debug": []})
    for p in payload_list:
        for k, d in p.items():
            merged[k]["psd"].extend(d.get("psd", []))
            merged[k]["savgol_vel"].extend(d.get("savgol_vel", []))
            merged[k]["phase_debug"].extend(d.get("phase_debug", []))
    return dict(merged)

def process_one_u(U, cases_u, default_cfg, do_ct_calcs, collect_traces = False):
    print(f"[Worker] U={U}, n_cases={len(cases_u)}")

    u_cache = preload_cases_for_u(
        cases_u=cases_u,
        fetch_data=fetch_data,
        t_analyze_start=default_cfg.tstart,
        do_ct_calcs=do_ct_calcs,
    )

    sdf, cdf, traces, ct_seed_df, ct_phase_df  = run_one_config_on_u_cache(
        cases_u=cases_u,
        u_cache=u_cache,
        cfg=default_cfg,
        do_ct_calcs=do_ct_calcs,
        collect_traces=collect_traces,
    )

    ct_seed_ens_df = reduce_ct_seed_ensemble(ct_seed_df)
    ct_phase_ens_df = reduce_ct_phase_ensemble(ct_phase_df)

    del u_cache
    return U, sdf, cdf, traces, ct_seed_ens_df, ct_phase_ens_df


def plot_psd_by_seastate(trace_payload, outdir, mode="mean", xlim=(1e-2, 10.0), include_vel_in_plot=False):
    os.makedirs(outdir, exist_ok=True)

    for sea_key, d in trace_payload.items():
        psd_curves = d.get("psd", [])
        if len(psd_curves) == 0:
            continue

        U, Hs, Tp = sea_key

        if include_vel_in_plot:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
            ax0 = None

        # ---------- top panel: velocity (optional) ----------
        if include_vel_in_plot:
            vel_curves = d.get("savgol_vel", [])
            clean_vel = []
            for case_name, t, v in vel_curves:
                t = np.asarray(t, float); v = np.asarray(v, float)
                m = np.isfinite(t) & np.isfinite(v)
                if np.sum(m) >= 2:
                    clean_vel.append((case_name, t[m], v[m]))

            if mode == "seed":
                for case_name, t, v in clean_vel:
                    ax0.plot(t, v, lw=1.0, alpha=0.8, label=case_name)

            elif mode == "mean" and len(clean_vel) > 0:
                tmin = max(np.min(t) for _, t, _ in clean_vel)
                tmax = min(np.max(t) for _, t, _ in clean_vel)
                if tmax > tmin:
                    tg = np.linspace(tmin, tmax, 1500)
                    V = []
                    for _, t, v in clean_vel:
                        idx = np.argsort(t)
                        tu, iu = np.unique(t[idx], return_index=True)
                        vu = v[idx][iu]
                        if len(tu) >= 2:
                            V.append(np.interp(tg, tu, vu))
                    if len(V) > 0:
                        V = np.vstack(V)
                        vm = np.nanmean(V, axis=0)
                        vs = np.nanstd(V, axis=0, ddof=1) if V.shape[0] > 1 else np.zeros_like(vm)
                        ax0.plot(tg, vm, lw=2, label=f"Mean vel (n={V.shape[0]})")
                        ax0.fill_between(tg, vm - vs, vm + vs, alpha=0.25, label="±1 std")

            ax0.set_ylabel("Velocity [m/s]")
            ax0.set_title(f"U={U}, Hs={Hs}, Tp={Tp}")
            ax0.grid(True, alpha=0.3)
            if mode == "seed" and len(clean_vel) <= 12 and len(clean_vel) > 0:
                ax0.legend(fontsize=7, ncol=2)
            elif mode == "mean":
                ax0.legend()

        # ---------- bottom/only panel: PSD ----------
        clean_psd = []
        for case_name, f, p in psd_curves:
            f = np.asarray(f, float); p = np.asarray(p, float)
            m = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p > 0)
            if np.sum(m) >= 2:
                clean_psd.append((case_name, f[m], p[m]))
        if not clean_psd:
            plt.close(fig)
            continue

        if mode == "seed":
            fmax_seen = 0.0
            for case_name, f, p in clean_psd:
                fmax_seen = max(fmax_seen, float(np.max(f)))
                ax1.plot(f, p, lw=1.0, alpha=0.8, label=case_name)
            n_used = len(clean_psd)

        elif mode == "mean":
            f0 = clean_psd[0][1]
            P = []
            for _, f, p in clean_psd:
                if len(f) == len(f0) and np.allclose(f, f0, atol=1e-12, rtol=0):
                    P.append(p)
            if len(P) == 0:
                plt.close(fig)
                continue
            P = np.vstack(P)
            pm = np.nanmean(P, axis=0)
            ps = np.nanstd(P, axis=0, ddof=1) if P.shape[0] > 1 else np.zeros_like(pm)
            ax1.plot(f0, pm, lw=2, label=f"Mean PSD (n={P.shape[0]})")
            ax1.fill_between(f0, np.maximum(pm - ps, 1e-16), pm + ps, alpha=0.25, label="±1 std")
            fmax_seen = float(np.max(f0))
            n_used = P.shape[0]
        else:
            plt.close(fig)
            raise ValueError("mode must be 'mean' or 'seed'")

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("PSD [m²/Hz]")
        ax1.grid(True, which="both", alpha=0.3)

        xmin, xmax_req = xlim
        ax1.set_xlim(xmin, min(xmax_req, fmax_seen) if fmax_seen > xmin else xmax_req)

        if mode == "seed" and n_used <= 12:
            ax1.legend(fontsize=7, ncol=2)
        if mode == "mean":
            ax1.legend()

        if include_vel_in_plot:
            fig.suptitle(f"Velocity + PSD ({mode})", y=0.98)
        else:
            ax1.set_title(f"PSD ({mode}) U={U}, Hs={Hs}, Tp={Tp}, n={n_used}")

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"PSD_{mode}_U{U:.2f}_Hs{Hs:.2f}_Tp{Tp:.2f}.png"), dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    outdir = "psd_values_fast"
    os.makedirs(outdir, exist_ok=True)

    n_jobs = 8

    start = time.time()

    # You can set this True for CT-enabled run
    do_ct_calcs = True
    collect_traces = False

    default_cfg = AnalysisCfg(
        detrend=False,
        vel_method="savgol",
        sg_window=301,
        sg_poly=4,
        welch_nperseg=3000,
        welch_overlap_frac=0.5,
        do_fft_analysis=False,
        tstart=600,
        f1=1/30,
        f2=1,
        psd_plot_mode="mean",
        include_vel_in_plot=False,
        ct_cache_decimals = 4,
        save_phase_debug = False
    )

    # cfg2 = replace(default_cfg, sg_window=401, sg_poly=5)
    # cfg3 = replace(default_cfg, vel_method="gradient", sg_window=None, sg_poly=None)


    cases = fetch_data.get_cases()
    cases_f = cases[
        (cases["WaveHs"] > 1.0) &
        (cases["WaveTp"] > 4.0) #&
        # (cases["HWindSpeed"] == 12.0)
    ]

    if cases_f.empty:
        raise RuntimeError("No cases after filtering.")

    summary_cases_list = []
    ct_cases_list = []
    all_trace_payloads = []
    ct_seed_list = []
    ct_phase_list = []

    U_values = sorted(cases_f["HWindSpeed"].unique())
    tasks = [(U, cases_f[cases_f["HWindSpeed"] == U].copy()) for U in U_values]

    # conservative worker count for memory-heavy workloads
    # n_jobs = min(len(tasks), max(1, (os.cpu_count() or 2) - 1))
    print(f"Running {len(tasks)} U-slices with n_jobs={n_jobs}")

    # loky = robust process-based backend (default)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_one_u)(U, cases_u, default_cfg, do_ct_calcs, collect_traces)
        for U, cases_u in tasks
    )

    # unpack
    for U, sdf, cdf, trace_payload, ct_seed_df, ct_phase_df in results:
        print(f"Done U={U}: summary_rows={len(sdf)}, ct_rows={len(cdf)}")
        summary_cases_list.append(sdf)
        if do_ct_calcs and (cdf is not None) and (not cdf.empty):
            ct_cases_list.append(cdf)
            ct_seed_list.append(ct_seed_df)
            ct_phase_list.append(ct_phase_df)
        if collect_traces:
            all_trace_payloads.append(trace_payload)

    # ========================================================
    # Aggregate results and save as CSV for plotting
    # ========================================================
    summary_df = pd.concat(summary_cases_list, ignore_index=True)
    if do_ct_calcs:
        ct_df = pd.concat(ct_cases_list, ignore_index=True)
        ct_seed_df = pd.concat(ct_seed_list, ignore_index=True)
        ct_phase_df = pd.concat(ct_phase_list, ignore_index=True)
    
    # Save summary CSV for frequency/amplitude plots
    summary_df.to_csv(
        os.path.join(outdir, "condition_summary_stats_all_cases.csv"),
        index=False
    )
    print(f"Saved: {os.path.join(outdir, 'condition_summary_stats_all_cases.csv')}")
    print(f"  Shape: {summary_df.shape}")
    print(f"  Columns: {list(summary_df.columns)}")

    # Merge traces across workers
    if collect_traces:
        trace_merged = merge_trace_payloads(all_trace_payloads)

        if do_ct_calcs and default_cfg.save_phase_debug:
            phase_debug_file = os.path.join(outdir, "phase_debug_traces.pkl")
            with open(phase_debug_file, "wb") as f:
                pickle.dump(trace_merged, f)
            print(f"Saved phase debug traces: {phase_debug_file}")
            
    # --------
    # CT dataframe mapping and save
    # --------
    if do_ct_calcs:
        ct_df.to_csv(
            os.path.join(outdir, "ct_condition_summary_stats_all_cases.csv"),
            index=False
        )
        print(f"Saved: {os.path.join(outdir, 'ct_all_summary.csv')}")
        print(f"  Shape: {ct_df.shape}")
        print(f"  Columns: {list(ct_df.columns)}")

        ct_seed_df.to_csv(
            os.path.join(outdir, "ct_seed_summary.csv"),
            index=False
        )
        ct_phase_df.to_csv(
            os.path.join(outdir, "ct_phase_summary.csv"),
            index=False
        )
    
    # --------
    # Optional: Verify data for plotting
    # --------
    print(f"\n=== Data Summary ===")
    print(f"Wind speeds: {sorted(summary_df['HWindSpeed'].unique())}")
    print(f"Wave heights: {sorted(summary_df['WaveHs'].unique())}")
    print(f"Wave periods: {sorted(summary_df['WaveTp'].unique())}")
    print(f"Total rows: {len(summary_df)}")

    if collect_traces:

        # Plot one averaged PSD per sea-state
        plot_mode = default_cfg.psd_plot_mode  # "mean" or "seed"
        save_path = os.path.join(outdir, f"psd_{plot_mode}_by_seastate")
        plot_psd_by_seastate(
            trace_merged,
            outdir=os.path.join(outdir, f"psd_{plot_mode}_by_seastate"),
            mode=plot_mode,
            xlim=(1e-2, 10.0),
            include_vel_in_plot = default_cfg.include_vel_in_plot

        )
        print(f"Saved PSD plots to: " + save_path)

        if do_ct_calcs and default_cfg.save_phase_debug:
            save_phase_debug_plots(
                trace_merged,
                outdir=os.path.join(outdir, "phase_debug_plots"),
                max_cases=72
            )
    end = time.time()
    print(end - start)



