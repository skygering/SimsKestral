import os
import itertools
import numpy as np
import pandas as pd
import fetch_data
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_fast import (
    normalize_cfg,
    cfg_key,
    preload_cases_for_u,
    run_one_config_on_u_cache,
    merge_trace_payloads
)

# =========================================================
# U-BATCH SENSITIVITY DRIVER
# - loops by U
# - preload each U once
# - run all configs on that preload
# - avoids repeated .out reads across configs
# =========================================================


def add_cfg_cols(df, cfg, stage):
    d = df.copy()
    d["stage"] = stage
    d["detrend"] = cfg["detrend"]
    d["vel_method"] = cfg["vel_method"]
    d["sg_window"] = cfg["sg_window"]
    d["sg_poly"] = cfg["sg_poly"]
    d["welch_nperseg"] = cfg["welch_nperseg"]
    d["welch_overlap_frac"] = cfg["welch_overlap_frac"]
    d["do_fft_analysis"] = cfg.get("do_fft_analysis", False)
    return d


def compare_vs_default(
    all_df,
    default_cfg,
    threshold_amp_pct=5.0,      # percent, e.g. 5 means 5%
    threshold_freq_abs=0.001,   # absolute Hz
    key_cols=("HWindSpeed", "WaveHs", "WaveTp"),
    metric_cols=(
        "v_A_ens_mean",
        "x_f_ens_mean",
        "v_f_ens_mean",
    ),
    cfg_cols=("stage", "detrend", "vel_method", "sg_window", "sg_poly", "welch_nperseg", "welch_overlap_frac", "do_fft_analysis"),
):
    import numpy as np
    import pandas as pd

    df = all_df.copy()

    # Default mask
    m = np.ones(len(df), dtype=bool)
    for c, v in default_cfg.items():
        if pd.isna(v):
            m &= df[c].isna()
        else:
            m &= (df[c] == v)

    base = df.loc[m, list(key_cols) + list(metric_cols)].copy()
    if base.empty:
        raise ValueError("Default config rows not found in all_df.")

    # Ensure one default row per sea-state key
    base = (
        base.groupby(list(key_cols), as_index=False)
        .first()
    )

    base = base.rename(columns={c: f"{c}_default" for c in metric_cols})
    merged = df.merge(base, on=list(key_cols), how="left", validate="many_to_one")

    # classify metrics
    amp_metrics = [c for c in metric_cols if (c.startswith("x_A") or c.startswith("v_A"))]
    freq_metrics = [c for c in metric_cols if (c.startswith("x_f") or c.startswith("v_f"))]

    for c in metric_cols:
        x = merged[c].to_numpy(float)
        b = merged[f"{c}_default"].to_numpy(float)

        # absolute diff (always useful to store)
        abs_diff = np.full_like(x, np.nan, dtype=float)
        ok_abs = np.isfinite(x) & np.isfinite(b)
        abs_diff[ok_abs] = x[ok_abs] - b[ok_abs]
        merged[f"{c}_abs_diff"] = abs_diff

        if c in amp_metrics:
            # percent diff for amplitude metrics
            pct_diff = np.full_like(x, np.nan, dtype=float)
            ok_pct = np.isfinite(x) & np.isfinite(b) & (b != 0.0)
            pct_diff[ok_pct] = 100.0 * (x[ok_pct] - b[ok_pct]) / b[ok_pct]
            merged[f"{c}_pct_diff"] = pct_diff
            merged[f"{c}_flag"] = np.abs(pct_diff) > float(threshold_amp_pct)

        elif c in freq_metrics:
            # absolute diff for frequency metrics
            merged[f"{c}_flag"] = np.abs(abs_diff) > float(threshold_freq_abs)

        else:
            # fallback: absolute
            merged[f"{c}_flag"] = np.abs(abs_diff) > float(threshold_freq_abs)

    flag_cols = [f"{c}_flag" for c in metric_cols]
    merged["any_flag"] = merged[flag_cols].any(axis=1)

    grp = merged.groupby(list(cfg_cols), dropna=False)
    summary = grp["any_flag"].agg(n_sea_states="count", n_flagged="sum").reset_index()
    summary["pct_flagged"] = 100.0 * summary["n_flagged"] / summary["n_sea_states"]

    # compact metrics in summary
    for c in metric_cols:
        if c in amp_metrics:
            s = grp[f"{c}_pct_diff"]
            med = s.apply(lambda z: np.nanmedian(np.abs(z.to_numpy(float)))).reset_index(
                name=f"{c}_med_abs_pct_diff"
            )
            mx = s.apply(lambda z: np.nanmax(np.abs(z.to_numpy(float)))).reset_index(
                name=f"{c}_max_abs_pct_diff"
            )
        else:
            s = grp[f"{c}_abs_diff"]
            med = s.apply(lambda z: np.nanmedian(np.abs(z.to_numpy(float)))).reset_index(
                name=f"{c}_med_abs_abs_diff"
            )
            mx = s.apply(lambda z: np.nanmax(np.abs(z.to_numpy(float)))).reset_index(
                name=f"{c}_max_abs_abs_diff"
            )

        summary = summary.merge(med, on=list(cfg_cols), how="left")
        summary = summary.merge(mx, on=list(cfg_cols), how="left")

    flagged = merged.loc[merged["any_flag"]].copy()
    return merged, summary, flagged


def build_staged_configs(default_cfg):
    stage_cfgs = {
        "A_welch": [],
        "B_detrend": [],
        "C_velocity": [],
        "D_savgol": [],
        "E_interaction": [],
    }

    # A: Welch-only
    for nperseg, ov in itertools.product([2500, 3000, 3500], [0.45, 0.5, 0.55]):
        c = dict(default_cfg)
        c["welch_nperseg"] = nperseg
        c["welch_overlap_frac"] = ov
        stage_cfgs["A_welch"].append(normalize_cfg(c))

    # B: detrend-only
    c = dict(default_cfg)
    c["detrend"] = True
    stage_cfgs["B_detrend"].append(normalize_cfg(c))

    # C: velocity-only
    c = dict(default_cfg)
    c["vel_method"] = "gradient"
    c["sg_window"] = np.nan
    c["sg_poly"] = np.nan
    stage_cfgs["C_velocity"].append(normalize_cfg(c))

    # D: SavGol-only
    for w, p in itertools.product([201, 301, 401], [3, 4, 5]):
        c = dict(default_cfg)
        c["vel_method"] = "savgol"
        c["sg_window"] = w
        c["sg_poly"] = p
        stage_cfgs["D_savgol"].append(normalize_cfg(c))

    # # E: interaction
    # inter = [
    #     {"detrend": True,  "vel_method": "savgol",   "sg_window": 301, "sg_poly": 4, "welch_nperseg": 4096,  "welch_overlap_frac": 0.75},
    #     {"detrend": False, "vel_method": "gradient", "sg_window": np.nan, "sg_poly": np.nan, "welch_nperseg": 16384, "welch_overlap_frac": 0.25},
    #     {"detrend": True,  "vel_method": "gradient", "sg_window": np.nan, "sg_poly": np.nan, "welch_nperseg": 8192,  "welch_overlap_frac": 0.5},
    #     {"detrend": False, "vel_method": "savgol",   "sg_window": 401, "sg_poly": 4, "welch_nperseg": 4096,  "welch_overlap_frac": 0.25},
    # ]
    # for x in inter:
    #     c = dict(default_cfg)
    #     c.update(x)
    #     stage_cfgs["E_interaction"].append(normalize_cfg(c))

    return stage_cfgs

def plot_failed_from_traces(
    flagged_df,
    stage_cfgs,
    default_cfg,
    trace_by_cfg,
    outdir,
    WF_frange=(1/30, 1/1),
    t_plot_start=600.0,
):
    plot_root = os.path.join(outdir, "failed_case_plots")
    os.makedirs(plot_root, exist_ok=True)

    target_stages = [s for s in ["A_welch", "D_savgol"] if s in stage_cfgs]
    ff = flagged_df[flagged_df["stage"].isin(target_stages)].copy()
    if ff.empty:
        print("No failed rows for A_welch/D_savgol.")
        return

    def mean_curve(curves, n=1000):
        if len(curves) == 0:
            return None, None
        xmin = max(np.min(c[0]) for c in curves)
        xmax = min(np.max(c[0]) for c in curves)
        if xmax <= xmin:
            return None, None
        xg = np.linspace(xmin, xmax, n)
        ys = []
        for c in curves:
            if len(c) == 3:
                _, x, y = c
            else:
                x, y = c
            idx = np.argsort(x)
            x = x[idx]; y = y[idx]
            ux, ui = np.unique(x, return_index=True)
            uy = y[ui]
            if len(ux) < 2:
                continue
            ys.append(np.interp(xg, ux, uy))
        if len(ys) == 0:
            return None, None
        return xg, np.nanmean(np.vstack(ys), axis=0)

    for stage in target_stages:
        stg_fail = ff[ff["stage"] == stage][["HWindSpeed", "WaveHs", "WaveTp"]].drop_duplicates()
        if stg_fail.empty:
            continue

        cfg_list = [default_cfg] + stage_cfgs[stage]
        stage_dir = os.path.join(plot_root, stage)
        os.makedirs(stage_dir, exist_ok=True)

        for _, ssrow in stg_fail.iterrows():
            U, Hs, Tp = float(ssrow["HWindSpeed"]), float(ssrow["WaveHs"]), float(ssrow["WaveTp"])
            sea_key = (U, Hs, Tp)

            fig, ax = plt.subplots(figsize=(10, 6))

            for i, cfg in enumerate(cfg_list):
                k = cfg_key(cfg)
                payload = trace_by_cfg.get(k, {})
                if sea_key not in payload:
                    continue

                color = plt.cm.tab10(i % 10)
                ls = ["-", "--", "-.", ":"][(i // 10) % 4]

                if stage == "A_welch":
                    xg, yg = mean_curve(payload[sea_key]["psd"])
                    if xg is None:
                        continue
                    band = (xg >= WF_frange[0]) & (xg <= WF_frange[1])
                    ax.plot(
                        xg[band], yg[band], color=color, linestyle=ls, lw=1.8,
                        label=f"nperseg={cfg['welch_nperseg']}, ov={cfg['welch_overlap_frac']}, det={cfg['detrend']}"
                    )
                else:
                    xg, yg = mean_curve(payload[sea_key]["savgol_vel"])
                    if xg is None:
                        continue
                    m = xg >= t_plot_start
                    ax.plot(
                        xg[m], yg[m], color=color, linestyle=ls, lw=1.8,
                        label=f"sg=({cfg['sg_window']},{cfg['sg_poly']}), det={cfg['detrend']}"
                    )

            ax.set_title(f"{stage} FAILED | U={U}, Hs={Hs}, Tp={Tp}")
            if stage == "A_welch":
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("PSD [m$^2$/Hz]")
                ax.set_yscale("log")
            else:
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Velocity [m/s]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, f"{stage}_U{U}_Hs{Hs}_Tp{Tp}.png"), dpi=150)
            plt.close(fig)

    print(f"Saved failed plots to: {plot_root}")


if __name__ == "__main__":
    outdir = "psd_sensitivity_fast"
    os.makedirs(outdir, exist_ok=True)

    threshold_freq_abs = 0.001  # Hz
    threshold_amp_pct = 2.0 # %
    metric_cols = (
        "v_A_ens_mean",
        "x_f_ens_mean",
        "v_f_ens_mean",
    )

    # You can set this True for CT-enabled run
    do_ct_calcs = False

    default_cfg = normalize_cfg({
        "detrend": False,
        "vel_method": "savgol",
        "sg_window": 301,
        "sg_poly": 4,
        "welch_nperseg": 3000,
        "welch_overlap_frac": 0.5,
        "do_fft_analysis": False,
    })

    stage_cfgs = build_staged_configs(default_cfg)

    # Build one deduplicated config list (default + all stages)
    all_cfg_records = [("default_reference", default_cfg)]
    for stage, cfg_list in stage_cfgs.items():
        for cfg in cfg_list:
            all_cfg_records.append((stage, cfg))

    # dedupe by cfg key
    unique_cfg_map = {}
    for stage, cfg in all_cfg_records:
        k = cfg_key(cfg)
        if k not in unique_cfg_map:
            unique_cfg_map[k] = cfg

    trace_cfg_keys = set()
    trace_cfg_keys.add(cfg_key(default_cfg))
    for cfg in stage_cfgs.get("A_welch", []):
        trace_cfg_keys.add(cfg_key(cfg))
    for cfg in stage_cfgs.get("D_savgol", []):
        trace_cfg_keys.add(cfg_key(cfg))

    cases = fetch_data.get_cases()
    cases_f = cases[(cases["WaveHs"] > 1.0) & (cases["WaveTp"] > 4.0)]
    cases_f = cases_f[cases_f["HWindSpeed"].isin([3, 10.5, 25])]

    if cases_f.empty:
        raise RuntimeError("No cases after filtering.")

    # Store results by config key, but computed U-by-U and appended
    summary_parts_by_cfg = {k: [] for k in unique_cfg_map.keys()}
    ct_parts_by_cfg = {k: [] for k in unique_cfg_map.keys()}
    trace_parts_by_cfg = {k: [] for k in unique_cfg_map.keys()}  # NEW

    # --------
    # U loop: preload once, run all unique configs
    # --------
    for U in sorted(cases_f["HWindSpeed"].unique()):
        cases_u = cases_f[cases_f["HWindSpeed"] == U].copy()
        print(f"\n=== U={U}: {len(cases_u)} cases ===")

        u_cache = preload_cases_for_u(
            cases_u=cases_u,
            fetch_data=fetch_data,
            t_analyze_start=600,
            do_ct_calcs=do_ct_calcs,
        )

        for k, cfg in unique_cfg_map.items():
            need_traces = k in trace_cfg_keys
            if need_traces:
                sdf, cdf, tr = run_one_config_on_u_cache(
                    cases_u=cases_u,
                    u_cache=u_cache,
                    cfg=cfg,
                    WF_frange=(1 / 30, 1 / 1),
                    do_ct_calcs=do_ct_calcs,
                    collect_traces=True,
                )
                trace_parts_by_cfg[k].append(tr)
            else:
                sdf, cdf, _ = run_one_config_on_u_cache(
                    cases_u=cases_u,
                    u_cache=u_cache,
                    cfg=cfg,
                    WF_frange=(1 / 30, 1 / 1),
                    do_ct_calcs=do_ct_calcs,
                    collect_traces=False,
                )

            summary_parts_by_cfg[k].append(sdf)
            if do_ct_calcs:
                ct_parts_by_cfg[k].append(cdf)

        # free memory from this U slice
        del u_cache

    # Materialize full result per config
    summary_by_cfg = {}
    ct_by_cfg = {}
    trace_by_cfg = {}
    for k, cfg in unique_cfg_map.items():
        summary_by_cfg[k] = pd.concat(summary_parts_by_cfg[k], ignore_index=True)
        if do_ct_calcs:
            ct_by_cfg[k] = pd.concat(ct_parts_by_cfg[k], ignore_index=True)
        need_traces = k in trace_cfg_keys
        if need_traces:
            trace_by_cfg[k] = merge_trace_payloads(trace_parts_by_cfg[k]) if len(trace_parts_by_cfg[k]) else {}

    # Expand into stage-labeled runs (reusing computed config results)
    all_runs = []
    for stage, cfg in all_cfg_records:
        k = cfg_key(cfg)
        all_runs.append(add_cfg_cols(summary_by_cfg[k], cfg, stage))

    all_df = pd.concat(all_runs, ignore_index=True).drop_duplicates()
    all_df.to_csv(os.path.join(outdir, "all_runs_compact.csv"), index=False)

    # Per-stage outputs
    for stage, cfg_list in stage_cfgs.items():
        stage_runs = [add_cfg_cols(summary_by_cfg[cfg_key(default_cfg)], default_cfg, stage)]
        for cfg in cfg_list:
            stage_runs.append(add_cfg_cols(summary_by_cfg[cfg_key(cfg)], cfg, stage))

        stage_df = pd.concat(stage_runs, ignore_index=True)

        merged, summary, flagged = compare_vs_default(
            stage_df,
            default_cfg=default_cfg,
            threshold_amp_pct=threshold_amp_pct,
            threshold_freq_abs=threshold_freq_abs,
            metric_cols=metric_cols,
        )
        merged.to_csv(os.path.join(outdir, f"{stage}_merged.csv"), index=False)
        summary.to_csv(os.path.join(outdir, f"{stage}_summary.csv"), index=False)
        flagged.to_csv(os.path.join(outdir, f"{stage}_flagged_only.csv"), index=False)

        print(f"{stage}: flagged rows = {int(merged['any_flag'].sum())}/{len(merged)} "
              f"({100.0*merged['any_flag'].mean():.2f}%)")

    # Combined comparisons
    all_merged, all_summary, all_flagged = compare_vs_default(
        all_df,
        default_cfg=default_cfg,
        threshold_amp_pct=threshold_amp_pct,
        threshold_freq_abs=threshold_freq_abs,
        metric_cols=metric_cols,
    )

    plot_failed_from_traces(
        flagged_df=all_flagged,
        stage_cfgs=stage_cfgs,
        default_cfg=default_cfg,
        trace_by_cfg=trace_by_cfg,
        outdir=outdir,
        WF_frange=(1/30, 1/1),
        t_plot_start=600.0,
    )

    all_summary.to_csv(os.path.join(outdir, "all_configs_summary.csv"), index=False)
    all_flagged.to_csv(os.path.join(outdir, "all_flagged_only.csv"), index=False)

    stage_flag_summary = (
        all_merged.groupby("stage", dropna=False)["any_flag"]
        .agg(n_rows="count", n_flagged="sum")
        .reset_index()
    )
    stage_flag_summary["pct_flagged"] = 100.0 * stage_flag_summary["n_flagged"] / stage_flag_summary["n_rows"]
    stage_flag_summary.to_csv(os.path.join(outdir, "stage_flag_summary.csv"), index=False)

    print("\n========== FINAL ==========")
    print(stage_flag_summary.to_string(index=False))
    print(f"\nSaved outputs to: {outdir}")



# ----------------------------
# Load data
# ----------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("psd_sensitivity_fast/A_welch_merged.csv")

# pick frequency diff column
if "x_f_ens_mean_abs_diff" in df.columns:
    freq_diff_col = "x_f_ens_mean_abs_diff"
elif "x_f_ens_mean_diff" in df.columns:
    freq_diff_col = "x_f_ens_mean_diff"
else:
    raise ValueError("Could not find frequency diff column.")

# clean + derived metrics
df = df[np.isfinite(df["welch_nperseg"]) & np.isfinite(df["welch_overlap_frac"]) & np.isfinite(df[freq_diff_col])].copy()
df["welch_nperseg"] = df["welch_nperseg"].astype(int)

fs = 100.0  # dt=0.01 s
default_bin = fs / 3000
df["delta_f_bin"] = np.round(fs / df["welch_nperseg"], decimals = 3)
df["diff_hz"] = df[freq_diff_col]
df["pct_bin"] = 100.0 * df["diff_hz"] / df["delta_f_bin"]

# categorical overlap label for hue
df["ov_label"] = df["welch_overlap_frac"].map({0.45: "0.45", 0.50: "0.50", 0.55: "0.55"})
df = df[df["ov_label"].notna()].copy()

df["x_label"] = (
    r"$N_S$ =" + df["welch_nperseg"].astype(str) + "\n" +
    r"($\Delta f=$" + df["delta_f_bin"].astype(str) + " Hz)"
)

# Get unique labels sorted by delta_f_bin (smallest first)
sorted_labels = df.drop_duplicates("x_label").sort_values("welch_nperseg")["x_label"].tolist()

# Convert to categorical with proper ordering
df["x_label"] = pd.Categorical(df["x_label"], categories=sorted_labels, ordered=True)
# Convert ov_label to categorical sorted by overlap value

# ----------------------------
# Plot (1x2 boxplots with hue)
# ----------------------------
sns.set_theme(style="whitegrid")
palette = {"0.45": "#1f77b4", "0.50": "#2ca02c", "0.55": "#d62728"}

fig, ax = plt.subplots(1, 1)

# Left: absolute difference [Hz]
sns.violinplot(
    data=df,
    x="x_label",
    y="diff_hz",
    hue="ov_label",
    palette=palette,
    ax=ax,
    inner="point", density_norm="count",
    hue_order=["0.45", "0.50", "0.55"],
)
ax.set_title("Frequency shift with Welch Parameters compared to \n$N_S = 3000$ and 50% Overlap")
ax.set_xlabel(r"$N_S$")
ax.set_ylabel(r"$\Delta f$ [Hz]")

# # single shared legend
handles, labels = ax.get_legend_handles_labels()

ax.get_legend().remove()
fig.legend(
    handles, 
    [f"{l}" for l in labels], 
    loc="upper center", 
    bbox_to_anchor=(1.1, 0.7),  # Push down below the plot
    frameon=False,
    title="Overlap Fraction"
)

plt.tight_layout()
plt.savefig(os.path.join(outdir, f"welch_freq_changes.png"), dpi=300, bbox_inches='tight')
