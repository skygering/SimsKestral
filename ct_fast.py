import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

EPS = 1e-12

# ----------------------------
# CT helpers
# ----------------------------
def compute_ct(Fxh, rho, A, Uref, yaw, tilt):
    den = 0.5 * rho * A * (Uref * np.cos(yaw) * np.cos(tilt)) ** 2
    if Uref <= 0:
        return np.full_like(Fxh, np.nan, dtype=float)
    return Fxh / den

def compute_ct_prime(model_Ct, Ct, yaw, tilt):
    Ct = np.asarray(Ct, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    tilt = np.asarray(tilt, dtype=float)
    sol = model_Ct(Ct, yaw=yaw, tilt=tilt)
    return sol.Ctprime

_MISS = object()
class CTPrimeCache:
    __slots__ = ("model", "ndigits", "scale", "_cache", "hits", "misses")

    def __init__(self, model_ctp, ndigits=5):
        self.model = model_ctp
        self.ndigits = ndigits
        self.scale = 10.0 ** ndigits
        self._cache = {}   # key: (qct, qyaw, qtilt) -> ctprime
        self.hits = 0
        self.misses = 0


def eval_ctprime_cached_array(cache, ct_arr, yaw_arr, tilt_arr, fn_compute_ctp):
    ct = np.asarray(ct_arr, dtype=np.float64)
    yaw = np.asarray(yaw_arr, dtype=np.float64)
    tilt = np.asarray(tilt_arr, dtype=np.float64)

    if not (ct.shape == yaw.shape == tilt.shape):
        raise ValueError("ct_arr, yaw_arr, tilt_arr must have the same shape")

    out = np.full(ct.shape, np.nan, dtype=np.float64)

    mask = np.isfinite(ct) & np.isfinite(yaw) & np.isfinite(tilt)
    if not np.any(mask):
        return out

    c = ct[mask]
    y = yaw[mask]
    t = tilt[mask]

    # Quantize to ndigits once (integer keys)
    s = cache.scale
    keys = np.empty((c.size, 3), dtype=np.int64)
    keys[:, 0] = np.rint(c * s).astype(np.int64)
    keys[:, 1] = np.rint(y * s).astype(np.int64)
    keys[:, 2] = np.rint(t * s).astype(np.int64)

    # Unique rounded keys + mapping back
    uniq, first_idx, inv = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    counts = np.bincount(inv, minlength=uniq.shape[0])

    uvals = np.empty(uniq.shape[0], dtype=np.float64)
    miss_u_idx = []

    get = cache._cache.get

    # Cache lookup only once per unique key
    for j in range(uniq.shape[0]):
        k = (int(uniq[j, 0]), int(uniq[j, 1]), int(uniq[j, 2]))
        v = get(k, _MISS)
        if v is _MISS:
            miss_u_idx.append(j)
        else:
            uvals[j] = v
            cache.hits += int(counts[j])

    # Compute all misses in one vectorized call
    if miss_u_idx:
        miss_u_idx = np.asarray(miss_u_idx, dtype=np.int64)
        src = first_idx[miss_u_idx]

        miss_vals = np.asarray(
            fn_compute_ctp(cache.model, c[src], y[src], t[src]),
            dtype=np.float64
        ).reshape(-1)

        if miss_vals.size == 1 and miss_u_idx.size > 1:
            miss_vals = np.full(miss_u_idx.size, miss_vals.item(), dtype=np.float64)
        if miss_vals.size != miss_u_idx.size:
            raise ValueError("fn_compute_ctp returned unexpected shape for misses")

        for j, v in zip(miss_u_idx, miss_vals):
            k = (int(uniq[j, 0]), int(uniq[j, 1]), int(uniq[j, 2]))
            fv = float(v)
            uvals[j] = fv
            cache._cache[k] = fv

        cache.misses += int(miss_u_idx.size)
        cache.hits += int((counts[miss_u_idx] - 1).sum())  # duplicates within call

    out[mask] = uvals[inv]
    return out

def safe_cv(x):
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(m) or abs(m) < EPS:
        return np.nan
    return s / abs(m)

def percent_fluctuation_series(x):
    x = np.asarray(x, float)
    m = np.nanmean(x)
    if not np.isfinite(m) or abs(m) < EPS:
        return np.full_like(x, np.nan)
    return 100.0 * (x - m) / abs(m)

def seed_variability_metrics(x, prefix):
    x = np.asarray(x, float)
    p = percent_fluctuation_series(x)
    return {
        f"{prefix}_mean": np.nanmean(x),
        f"{prefix}_std": np.nanstd(x),
        f"{prefix}_cv": safe_cv(x),
        f"{prefix}_pct_rms": np.sqrt(np.nanmean(p**2)),
        f"{prefix}_pct_p95_p5": np.nanpercentile(p, 95) - np.nanpercentile(p, 5),
    }

def detect_peak_cycles_from_v(
    t, v, f_est,
    prominence_scale=0.6,
    min_sep_frac=0.6,
    period_tol_frac=0.4,
    peak_mid_tol_frac=0.4,
    peak_prom_frac_of_peak=0.6,
):
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    good = np.isfinite(t) & np.isfinite(v)
    t, v = t[good], v[good]
    if len(t) < 5 or not np.isfinite(f_est) or f_est <= 0:
        return [], np.array([], dtype=int), np.nan

    dt = np.nanmedian(np.diff(t))
    fs = 1.0 / dt
    Texp = 1.0 / f_est

    min_sep = max(1, int(min_sep_frac * fs * Texp))
    prom = prominence_scale * np.nanstd(v)

    # global peak and trough candidates
    peak_idx, _ = find_peaks(v, distance=min_sep, prominence=prom)
    trough_idx, _ = find_peaks(-v, distance=max(1, min_sep // 2),
                            prominence=prom * peak_prom_frac_of_peak)

    if len(peak_idx) < 2:
        return [], peak_idx, Texp

    cycles = []
    for i0, i1 in zip(peak_idx[:-1], peak_idx[1:]):
        Tc = t[i1] - t[i0]
        if not np.isfinite(Tc) or Tc <= 0:
            continue

        rel_err = abs(Tc - Texp) / Texp
        if rel_err > period_tol_frac:
            continue

        # trough strictly between peaks
        inside = trough_idx[(trough_idx > i0) & (trough_idx < i1)]
        if len(inside) != 1:
            continue

        tr = inside[0]
        t_mid = 0.5 * (t[i0] + t[i1])
        half = 0.5 * Tc
        if half <= 0:
            continue

        mid_rel = abs(t[tr] - t_mid) / half
        if mid_rel > peak_mid_tol_frac:
            continue

        cycles.append((i0, i1))

    return cycles, peak_idx, Texp

def cycle_to_phase01(t_seg):
    # maps segment endpoints to [0,1]
    t0, t1 = t_seg[0], t_seg[-1]
    if t1 <= t0:
        return None
    return (t_seg - t0) / (t1 - t0)

def resample_cycle_to_bins(phase01, y, n_bins=10):
    centers = (np.arange(n_bins) + 0.5) / n_bins
    yb = np.interp(centers, phase01, y, left=np.nan, right=np.nan)
    return centers, yb

def phase_profile_from_cycles(t, y, cycles, n_bins=10):
    prof = []
    centers = (np.arange(n_bins) + 0.5) / n_bins
    for i0, i1 in cycles:
        if i1 - i0 < 3:
            continue
        tt = t[i0:i1+1]
        yy = y[i0:i1+1]
        good = np.isfinite(tt) & np.isfinite(yy)
        tt, yy = tt[good], yy[good]
        if len(tt) < 3:
            continue
        ph = cycle_to_phase01(tt)
        if ph is None:
            continue
        _, yb = resample_cycle_to_bins(ph, yy, n_bins=n_bins)
        prof.append(yb)
    if len(prof) == 0:
        return centers, np.full(n_bins, np.nan), np.full(n_bins, np.nan), 0
    arr = np.asarray(prof, float)
    return centers, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), arr.shape[0]

def analyze_ct_ctprime_for_seed(
    *,
    case_name,
    t,
    v,
    ct,
    ctp,
    f_est,
    U,
    Hs,
    Tp,
    n_bins=10,
    min_cycles_for_phase=8,
    prominence_scale=0.25,
    min_sep_frac=0.5,
    period_tol_frac=0.35,
    aux_series=None,      
    return_phase_debug=False,        # flag
):
    """
    Returns
    -------
    seed_row : dict
    phase_rows : list[dict]
    phase_debug : dict or None
    """
    ct = np.asarray(ct, dtype=float)
    ctp = np.asarray(ctp, dtype=float)
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    aux_series = aux_series or {}

    # joint finite mask so all plotted/analyzed series are aligned
    good = np.isfinite(t) & np.isfinite(v) & np.isfinite(ct) & np.isfinite(ctp)
    for _, arr in aux_series.items():
        a = np.asarray(arr, dtype=float)
        good &= np.isfinite(a)

    t0, v0, ct0, ctp0 = t[good], v[good], ct[good], ctp[good]
    aux0 = {k: np.asarray(arr, dtype=float)[good] for k, arr in aux_series.items()}

    seed_row = {
        "case_name": case_name,
        "HWindSpeed": U,
        "WaveHs": Hs,
        "WaveTp": Tp,
        "n_samples": int(np.sum(good)),
        "f_est": f_est,
        "n_cycles_found": 0,
        "n_cycles_kept": 0,
        "phase_ok": False,
        "phase_bins": n_bins,
    }

    phase_debug = None

    if len(t0) < 5:
        seed_row.update(seed_variability_metrics(np.array([np.nan]), "CT"))
        seed_row.update(seed_variability_metrics(np.array([np.nan]), "CTp"))
        seed_row["cv_ratio_CTp_over_CT"] = np.nan
        seed_row["cv_diff_CT_minus_CTp"] = np.nan
        return seed_row, [], phase_debug

    # variability
    m_ct = seed_variability_metrics(ct0, "CT")
    m_ctp = seed_variability_metrics(ctp0, "CTp")
    ct_cv = m_ct["CT_cv"]
    ctp_cv = m_ctp["CTp_cv"]

    seed_row.update(m_ct)
    seed_row.update(m_ctp)
    seed_row["cv_ratio_CTp_over_CT"] = (ctp_cv / ct_cv) if (np.isfinite(ct_cv) and abs(ct_cv) > EPS) else np.nan
    seed_row["cv_diff_CT_minus_CTp"] = (ct_cv - ctp_cv) if (np.isfinite(ct_cv) and np.isfinite(ctp_cv)) else np.nan

    # cycles
    cycles, peak_idx, Texp = detect_peak_cycles_from_v(
        t0, v0, f_est=f_est,
        # prominence_scale=prominence_scale,
        # min_sep_frac=min_sep_frac,
        # period_tol_frac=period_tol_frac,
    )
    seed_row["n_cycles_found"] = max(0, len(peak_idx) - 1)
    seed_row["n_cycles_kept"] = len(cycles)
    seed_row["T_expected"] = Texp

    # build debug payload regardless of pass/fail if requested
    if return_phase_debug:
        cycle_periods = []
        for i0, i1 in zip(peak_idx[:-1], peak_idx[1:]):
            cycle_periods.append(t0[i1] - t0[i0])

        y_map = {"CT": ct0, "CTp": ctp0, **aux0}
        series_dbg = {}
        for name, yy in y_map.items():
            centers, meanp, stdp, ncy = phase_profile_from_cycles(t0, yy, cycles, n_bins=n_bins)
            series_dbg[name] = {
                "phase01": centers,
                "mean": meanp,
                "std": stdp,
                "n_cycles": ncy,
            }

        phase_debug = {
            "case_name": case_name,
            "HWindSpeed": U, "WaveHs": Hs, "WaveTp": Tp,
            "f_est": f_est,
            "T_expected": Texp,
            "t": t0,
            "v": v0,
            "peak_idx": np.asarray(peak_idx, dtype=int),
            "peak_t": t0[peak_idx] if len(peak_idx) else np.array([]),
            "candidate_cycle_periods": np.asarray(cycle_periods, dtype=float),
            "accepted_cycles_idx_pairs": list(cycles),
            "series": series_dbg,
        }

    if len(cycles) < int(min_cycles_for_phase):
        seed_row["phase_ok"] = False
        return seed_row, [], phase_debug

    seed_row["phase_ok"] = True

    centers, ct_phase_mean, ct_phase_std, ncy_ct = phase_profile_from_cycles(t0, ct0, cycles, n_bins=n_bins)
    _, ctp_phase_mean, ctp_phase_std, ncy_ctp = phase_profile_from_cycles(t0, ctp0, cycles, n_bins=n_bins)

    ct_mu = np.nanmean(ct_phase_mean)
    ctp_mu = np.nanmean(ctp_phase_mean)

    seed_row["CT_phase_moddepth"] = ((np.nanmax(ct_phase_mean) - np.nanmin(ct_phase_mean)) / abs(ct_mu)
                                     if np.isfinite(ct_mu) and abs(ct_mu) > EPS else np.nan)
    seed_row["CTp_phase_moddepth"] = ((np.nanmax(ctp_phase_mean) - np.nanmin(ctp_phase_mean)) / abs(ctp_mu)
                                      if np.isfinite(ctp_mu) and abs(ctp_mu) > EPS else np.nan)
    seed_row["phase_moddepth_ratio_CTp_over_CT"] = (
        seed_row["CTp_phase_moddepth"] / seed_row["CT_phase_moddepth"]
        if np.isfinite(seed_row["CT_phase_moddepth"]) and abs(seed_row["CT_phase_moddepth"]) > EPS
        else np.nan
    )

    phase_rows = []
    for i, ph in enumerate(centers):
        phase_rows.append({
            "case_name": case_name,
            "HWindSpeed": U,
            "WaveHs": Hs,
            "WaveTp": Tp,
            "phase01": float(ph),
            "phase_bin": int(i),
            "n_cycles_used": int(min(ncy_ct, ncy_ctp)),
            "CT_phase_mean": float(ct_phase_mean[i]) if np.isfinite(ct_phase_mean[i]) else np.nan,
            "CT_phase_std": float(ct_phase_std[i]) if np.isfinite(ct_phase_std[i]) else np.nan,
            "CTp_phase_mean": float(ctp_phase_mean[i]) if np.isfinite(ctp_phase_mean[i]) else np.nan,
            "CTp_phase_std": float(ctp_phase_std[i]) if np.isfinite(ctp_phase_std[i]) else np.nan,
        })

    return seed_row, phase_rows, phase_debug

def reduce_ct_seed_ensemble(ct_seed_df):
    """
    Ensemble summary over per-seed CT/CT' variability metrics.
    One row per (HWindSpeed, WaveHs, WaveTp).
    """
    if ct_seed_df is None or len(ct_seed_df) == 0:
        return pd.DataFrame()

    grp = ["HWindSpeed", "WaveHs", "WaveTp"]

    cols_mean = [
        "CT_cv", "CTp_cv",
        "cv_ratio_CTp_over_CT", "cv_diff_CT_minus_CTp",
        "CT_pct_rms", "CTp_pct_rms",
        "CT_pct_p95_p5", "CTp_pct_p95_p5",
        "CT_phase_moddepth", "CTp_phase_moddepth",
        "phase_moddepth_ratio_CTp_over_CT",
    ]
    cols_mean = [c for c in cols_mean if c in ct_seed_df.columns]

    agg = {c: ["mean", "std"] for c in cols_mean}
    agg.update({
        "case_name": "nunique",
        "phase_ok": "sum",
    })

    out = ct_seed_df.groupby(grp).agg(agg)
    out.columns = ["_".join([a for a in col if a]).rstrip("_") for col in out.columns.to_flat_index()]
    out = out.reset_index().rename(columns={
        "case_name_nunique": "n_seeds",
        "phase_ok_sum": "n_seeds_phase_ok",
    })
    return out

def reduce_ct_phase_ensemble(ct_phase_df):
    """
    Reduce per-seed phase rows into ensemble phase stats per sea state.

    Input expected columns:
      - case_name
      - HWindSpeed, WaveHs, WaveTp
      - phase01, phase_bin
      - CT_phase_mean, CT_phase_std
      - CTp_phase_mean, CTp_phase_std
      - n_cycles_used

    Returns
    -------
    phase_ens_df : pd.DataFrame
      One row per (HWindSpeed, WaveHs, WaveTp, phase01, phase_bin), with:
        - number of seeds contributing
        - ensemble mean/std for CT and CT'
        - optional constancy indicators at each phase bin
    """
    if ct_phase_df is None or len(ct_phase_df) == 0:
        return pd.DataFrame(columns=[
            "HWindSpeed", "WaveHs", "WaveTp", "phase01", "phase_bin",
            "n_seeds_phase",
            "CT_phase_ens_mean", "CT_phase_ens_std",
            "CTp_phase_ens_mean", "CTp_phase_ens_std",
            "CTp_minus_CT_phase_ens_mean", "CTp_minus_CT_phase_ens_std",
            "n_cycles_used_mean", "n_cycles_used_min", "n_cycles_used_max",
        ])

    req = [
        "HWindSpeed", "WaveHs", "WaveTp", "phase01", "phase_bin",
        "case_name", "CT_phase_mean", "CTp_phase_mean", "n_cycles_used"
    ]
    missing = [c for c in req if c not in ct_phase_df.columns]
    if missing:
        raise ValueError(f"ct_phase_df missing required columns: {missing}")

    df = ct_phase_df.copy()

    # paired difference per seed/bin (helps compare shape differences directly)
    df["CTp_minus_CT_phase"] = df["CTp_phase_mean"] - df["CT_phase_mean"]

    grp_cols = ["HWindSpeed", "WaveHs", "WaveTp", "phase01", "phase_bin"]

    def _nanstd(x):
        x = np.asarray(x, dtype=float)
        return np.nanstd(x)

    out = (
        df.groupby(grp_cols, dropna=False)
          .agg(
              n_seeds_phase=("case_name", lambda x: x.nunique()),
              CT_phase_ens_mean=("CT_phase_mean", "mean"),
              CT_phase_ens_std=("CT_phase_mean", _nanstd),
              CTp_phase_ens_mean=("CTp_phase_mean", "mean"),
              CTp_phase_ens_std=("CTp_phase_mean", _nanstd),
              CTp_minus_CT_phase_ens_mean=("CTp_minus_CT_phase", "mean"),
              CTp_minus_CT_phase_ens_std=("CTp_minus_CT_phase", _nanstd),
              n_cycles_used_mean=("n_cycles_used", "mean"),
              n_cycles_used_min=("n_cycles_used", "min"),
              n_cycles_used_max=("n_cycles_used", "max"),
          )
          .reset_index()
          .sort_values(["HWindSpeed", "WaveHs", "WaveTp", "phase_bin"])
    )

    return out

def make_phase_debug_payload(t, v, cycles, peak_idx, Texp, y_map, n_bins=10):
    out = {
        "T_expected": Texp,
        "peak_idx": np.asarray(peak_idx),
        "peak_t": np.asarray(t)[peak_idx] if len(peak_idx) else np.array([]),
        "cycles": cycles,  # accepted pairs
        "series": {}
    }
    for name, y in y_map.items():
        centers, meanp, stdp, ncy = phase_profile_from_cycles(t, y, cycles, n_bins=n_bins)
        out["series"][name] = {
            "phase01": centers,
            "mean": meanp,
            "std": stdp,
            "n_cycles": ncy,
        }
    return out

def plot_phase_debug_case(
    phase_debug, U_infty,
    series_to_plot=("CT", "CTp", "v"),
    show=True
):
    if phase_debug is None:
        raise ValueError("phase_debug is None")

    t = phase_debug["t"]
    v = phase_debug["v"]
    peak_idx = phase_debug["peak_idx"]
    cycles = phase_debug["accepted_cycles_idx_pairs"]
    series = phase_debug["series"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    # A) velocity + troughs
    ax = axes[0]
    ax.plot(t, v, lw=1.2, label="v")
    if len(peak_idx):
        ax.plot(t[peak_idx], v[peak_idx], "rv", ms=5, label="peaks")
    ax.set_title(f"{phase_debug.get('case_name','case')} | Velocity + detected peaks")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("v")
    ax.legend()

    # B) accepted cycle spans
    ax = axes[1]
    ax.plot(t, v, color="0.6", lw=1.0, label="v (background)")
    cmap = plt.get_cmap("tab10")
    for j, (i0, i1) in enumerate(cycles):
        c = cmap(j % 10)
        ax.axvspan(t[i0], t[i1], color=c, alpha=0.22, label=f"cycle {j}" if j < 10 else None)
        tm = 0.5 * (t[i0] + t[i1])
        ax.axvline(tm, color=c, lw=0.8, alpha=0.6)
    ax.set_title(f"Accepted cycles: {len(cycles)} | Texp={phase_debug.get('T_expected', np.nan):.3f}s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("v")
    if len(cycles) <= 10:
        ax.legend(ncol=2, fontsize=8)

    # C) phase profiles (left axis)
    ax = axes[2]
    for name in series_to_plot:
        if name not in series:
            continue
        ph = np.asarray(series[name]["phase01"])
        mu = np.asarray(series[name]["mean"])
        sd = np.asarray(series[name]["std"])
        ax.plot(ph, mu, lw=2, label=f"{name} mean")
        ax.fill_between(ph, mu - sd, mu + sd, alpha=0.2, label=f"{name} ±1σ")

    # ---- right axis: velocity phase mean ± std ----
    # ---- right axis: non-dimensional velocity phase mean ± std ----
    axr = ax.twinx()
    if "v" in series:
        phv = np.asarray(series["v"]["phase01"], float)
        vmu = np.asarray(series["v"]["mean"], float)
        vsd = np.asarray(series["v"]["std"], float)

        if not np.isfinite(U_infty) or abs(U_infty) < 1e-12:
            raise ValueError(f"Invalid U_infty={U_infty} for non-dimensionalization.")

        urel_mu = 1.0 - (vmu / U_infty)
        urel_sd = np.abs(vsd / U_infty)

        axr.plot(phv, urel_mu, color="black", lw=2.2, ls="--", label=r"$\overline{U}_{rel}$ (right)")
        axr.fill_between(phv, urel_mu - urel_sd, urel_mu + urel_sd,
                         color="gray", alpha=0.20, label=r"$U_{rel}\pm1\sigma$ (right)")
        axr.set_ylabel(r"$U_{rel}=1-v/U_\infty$ [-]", color="black")
        axr.tick_params(axis="y", labelcolor="black")

    # v peak-phase scatter on right axis (also non-dimensionalized)
    peak_phases, peak_urel_vals, cols = [], [], []
    cmap = plt.get_cmap("tab10")
    for j, (i0, i1) in enumerate(cycles):
        if i1 <= i0 + 2:
            continue
        vseg = np.asarray(v[i0:i1+1], float)
        if np.all(~np.isfinite(vseg)):
            continue
        k = int(np.nanargmin(vseg))
        ph_tr = k / (len(vseg) - 1)
        vtr = float(np.nanmin(vseg))
        peak_phases.append(ph_tr)   # rename variable if you want
        peak_urel_vals.append(1.0 - vtr / U_infty)
        cols.append(cmap(j % 10))

    if len(peak_phases):
        axr.scatter(peak_phases, peak_urel_vals, c=cols, s=28, edgecolors="k",
                    linewidths=0.3, label=r"$U_{rel}$ peak per cycle")
        ax.axvline(0.5, color="k", ls="--", lw=1.0, alpha=0.6)

    # ax.set_xlim(0, 1)
    ax.set_title("Phase profiles (0..1)")
    ax.set_xlabel("Phase (0..1)")
    ax.set_ylabel("CT / CT' (left axis)")

    # combine legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=2, fontsize=8, loc="best")

    ax.set_title("Phase profiles (0..1)")
    ax.set_xlabel("Phase (0..1)")
    ax.set_ylabel("Value")
    # ax.set_xlim(0, 1)
    ax.legend(ncol=2)

    if show:
        plt.show()

    return fig, axes

def save_phase_debug_plots(trace_payload, outdir, max_cases=30, series_to_plot=("CT", "CTp", "x")):
    os.makedirs(outdir, exist_ok=True)
    n_saved = 0

    for sea_key, d in trace_payload.items():
        dbg_list = d.get("phase_debug", [])
        if not dbg_list:
            continue

        U, Hs, Tp = sea_key
        for case_name, phase_debug in dbg_list:
            fig, _ = plot_phase_debug_case(
                phase_debug, U,
                series_to_plot=series_to_plot,
                show=False
            )
            fname = f"phase_debug_U{U:.2f}_Hs{Hs:.2f}_Tp{Tp:.2f}_{case_name}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
            plt.close(fig)

            n_saved += 1
            if n_saved >= max_cases:
                print(f"Saved {n_saved} phase debug plots to {outdir}")
                return

    print(f"Saved {n_saved} phase debug plots to {outdir}")