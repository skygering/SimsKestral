import os
import itertools
import numpy as np
import pandas as pd
import fetch_data

# IMPORTANT: replace with your module
from analyze_2 import run_analysis


# -----------------------------
# Utilities
# -----------------------------
def add_config_cols(df, detrend, vel_method, sg_window, sg_poly, nperseg, ov, stage_label):
    out = df.copy()
    out["detrend"] = bool(detrend)
    out["vel_method"] = vel_method
    out["sg_window"] = sg_window if vel_method == "savgol" else np.nan
    out["sg_poly"] = sg_poly if vel_method == "savgol" else np.nan
    out["welch_nperseg"] = int(nperseg)
    out["welch_overlap_frac"] = float(ov)
    out["stage"] = stage_label
    return out


def run_one(cases, detrend, vel_method, sg_window, sg_poly, nperseg, ov, stage_label):
    summary_df, _ = run_analysis(
        cases=cases,
        fetch_data=fetch_data,
        min_hs=1.0,
        min_tp=4.0,
        detrend=detrend,
        WF_frange=(1/30, 1/1),
        vel_method=vel_method,
        sg_window=int(sg_window) if np.isfinite(sg_window) else 201,
        sg_poly=int(sg_poly) if np.isfinite(sg_poly) else 3,
        do_fft_analysis=False,
        do_ct_calcs=False,
        welch_nperseg=int(nperseg),
        welch_overlap_frac=float(ov),
        t_analyze_start=600,
    )
    return add_config_cols(summary_df, detrend, vel_method, sg_window, sg_poly, nperseg, ov, stage_label)

def flag_vs_default(
    all_df,
    default_cfg,
    threshold_pct=5.0,
    key_cols=("HWindSpeed", "WaveHs", "WaveTp"),
    metric_cols=(
        "x_A_ens_mean",
        "v_A_ens_mean",
        "v_A_from_x_A_ens_mean",
        "x_f_ens_mean",
        "v_f_ens_mean",
    ),
    cfg_cols=("stage", "detrend", "vel_method", "sg_window", "sg_poly", "welch_nperseg", "welch_overlap_frac"),
):
    """
    Compare every row in all_df against a fixed default configuration
    for the same sea state (key_cols), and flag rows where abs(%diff)
    exceeds threshold_pct for any metric.

    Returns:
      merged_df   : row-level table with default values, %diffs, metric flags, any_flag
      summary_df  : grouped by cfg_cols with flagged counts and % flagged
      flagged_df  : subset of merged_df where any_flag == True
    """
    df = all_df.copy()

    # -----------------------------
    # 1) Extract default rows
    # -----------------------------
    # Build boolean mask for rows matching default config.
    default_mask = np.ones(len(df), dtype=bool)
    for col, default_val in default_cfg.items():
        if pd.isna(default_val):
            default_mask &= df[col].isna()
        else:
            default_mask &= (df[col] == default_val)

    default_rows = df.loc[default_mask, list(key_cols) + list(metric_cols)].copy()
    if default_rows.empty:
        raise ValueError(
            "No rows found for default config. "
            f"default_cfg={default_cfg}"
        )

    # Rename default metric columns so we can merge them onto all rows.
    rename_map = {m: f"{m}__default" for m in metric_cols}
    default_rows = default_rows.rename(columns=rename_map)

    # -----------------------------
    # 2) Merge default metrics onto every row by sea state
    # -----------------------------
    merged = df.merge(
        default_rows,
        on=list(key_cols),
        how="left",
        validate="many_to_one",
    )

    # -----------------------------
    # 3) Compute % difference + per-metric flags
    # -----------------------------
    # %diff = 100 * (value - default) / default
    for metric in metric_cols:
        value = merged[metric].to_numpy(dtype=float)
        base = merged[f"{metric}__default"].to_numpy(dtype=float)

        pctdiff = np.full_like(value, np.nan, dtype=float)
        valid = np.isfinite(value) & np.isfinite(base) & (base != 0.0)
        pctdiff[valid] = 100.0 * (value[valid] - base[valid]) / base[valid]

        merged[f"{metric}__pctdiff"] = pctdiff
        merged[f"{metric}__flag"] = np.abs(pctdiff) > float(threshold_pct)

    # Any-metric flag for each row
    metric_flag_cols = [f"{m}__flag" for m in metric_cols]
    merged["any_flag"] = merged[metric_flag_cols].any(axis=1)

    # -----------------------------
    # 4) Summarize by configuration
    # -----------------------------
    grouped = merged.groupby(list(cfg_cols), dropna=False)

    summary = grouped["any_flag"].agg(
        n_sea_states="count",
        n_flagged="sum",
    ).reset_index()
    summary["pct_flagged"] = 100.0 * summary["n_flagged"] / summary["n_sea_states"]

    # Add median and max absolute %diff per metric
    for metric in metric_cols:
        pct_col = f"{metric}__pctdiff"

        med_abs = grouped[pct_col].apply(
            lambda s: np.nanmedian(np.abs(s.to_numpy(dtype=float)))
        ).reset_index(name=f"{metric}_med_abs_pctdiff")

        max_abs = grouped[pct_col].apply(
            lambda s: np.nanmax(np.abs(s.to_numpy(dtype=float)))
        ).reset_index(name=f"{metric}_max_abs_pctdiff")

        summary = summary.merge(med_abs, on=list(cfg_cols), how="left")
        summary = summary.merge(max_abs, on=list(cfg_cols), how="left")

    # -----------------------------
    # 5) Return flagged subset too
    # -----------------------------
    flagged_only = merged.loc[merged["any_flag"]].copy()

    return merged, summary, flagged_only


def stage_report(name, merged, summary, outdir):
    merged.to_csv(os.path.join(outdir, f"{name}_rowwise.csv"), index=False)
    summary.to_csv(os.path.join(outdir, f"{name}_summary.csv"), index=False)
    flagged = merged.loc[merged["any_flag"]]
    flagged.to_csv(os.path.join(outdir, f"{name}_flagged_only.csv"), index=False)

    print(f"\n=== {name} ===")
    print(f"rows: {len(merged)} | flagged rows: {int(merged['any_flag'].sum())} | pct flagged: {100*merged['any_flag'].mean():.2f}%")
    print(summary.sort_values("pct_flagged", ascending=False).head(10).to_string(index=False))


# -----------------------------
# Main: staged reduced workflow
# -----------------------------
if __name__ == "__main__":
    outdir = "psd_sensitivity_staged_vs_default"
    os.makedirs(outdir, exist_ok=True)

    threshold_pct = 5.0  # user input
    metric_cols = (
        "x_A_ens_mean",
        "v_A_ens_mean",
        "v_A_from_x_A_ens_mean",
        "x_f_ens_mean",
        "v_f_ens_mean",
    )

    # Fixed default (reference only)
    default_cfg = {
        "detrend": False,
        "vel_method": "savgol",
        "sg_window": 201,
        "sg_poly": 3,
        "welch_nperseg": 8192,
        "welch_overlap_frac": 0.5,
    }

    cases = fetch_data.get_cases()
    all_stage_runs = []

    # Always run default once (needed as reference row in merged tables)
    default_df = run_one(
        cases=cases,
        detrend=default_cfg["detrend"],
        vel_method=default_cfg["vel_method"],
        sg_window=default_cfg["sg_window"],
        sg_poly=default_cfg["sg_poly"],
        nperseg=default_cfg["welch_nperseg"],
        ov=default_cfg["welch_overlap_frac"],
        stage_label="default_reference",
    )
    all_stage_runs.append(default_df)

    # ---------------- Stage A: Welch-only changes ----------------
    # Keep detrend/vel/sg at default
    welch_npersegs = [4096, 16384]          # changes around default 8192
    welch_overlap_fracs = [0.25, 0.75]      # changes around default 0.5

    stageA = []
    # Include default point in this stage for readability
    stageA.append(default_df.assign(stage="A_welch"))
    for nperseg, ov in itertools.product(welch_npersegs, welch_overlap_fracs):
        print(f"[Stage A] nperseg={nperseg}, ov={ov}")
        df = run_one(
            cases=cases,
            detrend=default_cfg["detrend"],
            vel_method=default_cfg["vel_method"],
            sg_window=default_cfg["sg_window"],
            sg_poly=default_cfg["sg_poly"],
            nperseg=nperseg,
            ov=ov,
            stage_label="A_welch",
        )
        stageA.append(df)
        all_stage_runs.append(df)

    stageA_df = pd.concat(stageA, ignore_index=True)
    A_merged, A_summary, A_flagged = flag_vs_default(
        stageA_df, default_cfg, threshold_pct=threshold_pct, metric_cols=metric_cols
    )
    stage_report("stageA_welch_only", A_merged, A_summary, outdir)

    # ---------------- Stage B: detrend-only change ----------------
    stageB = [default_df.assign(stage="B_detrend")]
    print("[Stage B] detrend=True")
    dfB = run_one(
        cases=cases,
        detrend=True,
        vel_method=default_cfg["vel_method"],
        sg_window=default_cfg["sg_window"],
        sg_poly=default_cfg["sg_poly"],
        nperseg=default_cfg["welch_nperseg"],
        ov=default_cfg["welch_overlap_frac"],
        stage_label="B_detrend",
    )
    stageB.append(dfB)
    all_stage_runs.append(dfB)

    stageB_df = pd.concat(stageB, ignore_index=True)
    B_merged, B_summary, B_flagged = flag_vs_default(
        stageB_df, default_cfg, threshold_pct=threshold_pct, metric_cols=metric_cols
    )
    stage_report("stageB_detrend_only", B_merged, B_summary, outdir)

    # ---------------- Stage C: velocity-method-only change ----------------
    stageC = [default_df.assign(stage="C_velocity")]
    print("[Stage C] vel_method=gradient")
    dfC = run_one(
        cases=cases,
        detrend=default_cfg["detrend"],
        vel_method="gradient",
        sg_window=201,  # ignored for gradient
        sg_poly=3,      # ignored for gradient
        nperseg=default_cfg["welch_nperseg"],
        ov=default_cfg["welch_overlap_frac"],
        stage_label="C_velocity",
    )
    stageC.append(dfC)
    all_stage_runs.append(dfC)

    stageC_df = pd.concat(stageC, ignore_index=True)
    C_merged, C_summary, C_flagged = flag_vs_default(
        stageC_df, default_cfg, threshold_pct=threshold_pct, metric_cols=metric_cols
    )
    stage_report("stageC_velocity_only", C_merged, C_summary, outdir)

    # ---------------- Stage D: SavGol-only changes ----------------
    # Keep detrend/welch at default and vel=savgol; vary SG around default
    stageD = [default_df.assign(stage="D_savgol")]
    sg_windows = [101, 201, 401]   # around default 201
    sg_polys = [3, 4]         # around default 3

    for w, p in itertools.product(sg_windows, sg_polys):
        if p >= w:
            continue
        print(f"[Stage D] sg_window={w}, sg_poly={p}")
        df = run_one(
            cases=cases,
            detrend=default_cfg["detrend"],
            vel_method="savgol",
            sg_window=w,
            sg_poly=p,
            nperseg=default_cfg["welch_nperseg"],
            ov=default_cfg["welch_overlap_frac"],
            stage_label="D_savgol",
        )
        stageD.append(df)
        all_stage_runs.append(df)

    stageD_df = pd.concat(stageD, ignore_index=True)
    D_merged, D_summary, D_flagged = flag_vs_default(
        stageD_df, default_cfg, threshold_pct=threshold_pct, metric_cols=metric_cols
    )
    stage_report("stageD_savgol_only", D_merged, D_summary, outdir)

    # ---------------- Stage E: small interaction spot-check ----------------
    # minimal combos to check interactions without full factorial
    stageE = [default_df.assign(stage="E_interaction")]
    interaction_runs = [
        # Welch + detrend
        {"detrend": True,  "vel_method": "savgol",   "sg_window": 201, "sg_poly": 3, "nperseg": 4096,  "ov": 0.75},
        # Welch + velocity
        {"detrend": False, "vel_method": "gradient", "sg_window": 201, "sg_poly": 3, "nperseg": 16384, "ov": 0.25},
        # detrend + velocity
        {"detrend": True,  "vel_method": "gradient", "sg_window": 201, "sg_poly": 3, "nperseg": 8192,  "ov": 0.5},
        # Welch + SG
        {"detrend": False, "vel_method": "savgol",   "sg_window": 401, "sg_poly": 4, "nperseg": 4096,  "ov": 0.25},
    ]

    for rr in interaction_runs:
        print(f"[Stage E] {rr}")
        df = run_one(
            cases=cases,
            detrend=rr["detrend"],
            vel_method=rr["vel_method"],
            sg_window=rr["sg_window"],
            sg_poly=rr["sg_poly"],
            nperseg=rr["nperseg"],
            ov=rr["ov"],
            stage_label="E_interaction",
        )
        stageE.append(df)
        all_stage_runs.append(df)

    stageE_df = pd.concat(stageE, ignore_index=True)
    E_merged, E_summary, E_flagged = flag_vs_default(
        stageE_df, default_cfg, threshold_pct=threshold_pct, metric_cols=metric_cols
    )
    stage_report("stageE_interaction_spotcheck", E_merged, E_summary, outdir)

    # ---------------- Combined summary ----------------
    all_df = pd.concat(all_stage_runs, ignore_index=True).drop_duplicates()
    all_df.to_csv(os.path.join(outdir, "all_stages_raw.csv"), index=False)

    all_merged, all_summary, all_flagged = flag_vs_default(
        all_df, default_cfg, threshold_pct=threshold_pct, metric_cols=metric_cols
    )
    all_merged.to_csv(os.path.join(outdir, "all_stages_vs_default_rowwise.csv"), index=False)
    all_summary.to_csv(os.path.join(outdir, "all_stages_config_summary.csv"), index=False)
    all_flagged.to_csv(os.path.join(outdir, "all_stages_flagged_only.csv"), index=False)

    # Stage-level "what changed causes flags" summary
    stage_flag_summary = (
        all_merged.groupby("stage", dropna=False)["any_flag"]
        .agg(n_rows="count", n_flagged="sum")
        .reset_index()
    )
    stage_flag_summary["pct_flagged"] = 100.0 * stage_flag_summary["n_flagged"] / stage_flag_summary["n_rows"]
    stage_flag_summary.to_csv(os.path.join(outdir, "stage_flag_summary.csv"), index=False)

    print("\n================ FINAL ================")
    print(f"Meaningful-difference threshold: {threshold_pct:.2f}%")
    print(stage_flag_summary.to_string(index=False))
    print("\nIf a stage has flagged rows, that type of method change can cause meaningful differences.")
    print(f"\nSaved all outputs to: {outdir}")