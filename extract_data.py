import analysis_fast as af
import numpy as np
import os
from UnifiedMomentumModel import Momentum as UMM
import fetch_data
import pandas as pd

def export_single_case_from_cases_df(
    cases_df,
    cfg,
    fetch_data,
    case_idx=0,
    t_analyze_start=600,
    out_path = ".",
    out_csv_ts="single_case_timeseries.csv",
    out_csv_meta="single_case_metadata.csv",
    WF_frange=(1/30, 1/1),
    non_dim = True
):
    row = cases_df.iloc[case_idx]
    case_name = row["case_name"]
    Uref = float(row["HWindSpeed"])

    rho = 1.225
    D = 240.0
    A = np.pi * (D ** 2) / 4.0

    file = os.path.join(fetch_data.DATA_PATH, "outfiles", f"{case_name}.out")
    data = fetch_data.load_outfile(file)

    t_all, x_all = af.build_displacement(data, fetch_data.HUB)
    m = t_all > t_analyze_start
    t = t_all[m]
    x = x_all[m]

    v = af.calc_velocity(
        t, x,
        method=cfg["vel_method"],
        window=int(cfg["sg_window"]),
        poly=int(cfg["sg_poly"]),
    )

    fv, Pv = af.welch_psd_density(
        t, v,
        nperseg=int(cfg["welch_nperseg"]),
        overlap_frac=float(cfg["welch_overlap_frac"]),
    )
    v_f, v_A = af.band_metrics(fv, Pv, WF_frange)

    if non_dim:
        print(v_f)
        v_f = v_f * D / Uref
        v_A = v_A / Uref

    model_Ct = UMM.ThrustBasedUnified()

    Fxh = data["RtFldFxh"].to_numpy(dtype=float)[m]
    Ct_ref = data["RtFldCt"].to_numpy(dtype=float)[m]
    yaw = np.deg2rad(data["PtfmYaw"].to_numpy(dtype=float)[m])
    tilt = np.deg2rad(data["PtfmPitch"].to_numpy(dtype=float)[m])

    Ct = af.compute_ct(Fxh, rho=rho, A=A, Uref=Uref, yaw=yaw, tilt=tilt)
    Ct_p = af.compute_ct_prime(model_Ct, Ct_ref, yaw, tilt)

    mean_Ct = float(np.nanmean(Ct))
    mean_Ct_p = float(np.nanmean(Ct_p))

    if non_dim:
        x = x / D
        v = v / Uref
        t = t * Uref / D

    n = min(len(t), len(x), len(v), len(Ct), len(Ct_p))

    # 1) timeseries CSV
    df_ts = pd.DataFrame({
        "case_name": case_name,
        "HWindSpeed": Uref,
        "WaveHs": float(row["WaveHs"]),
        "WaveTp": float(row["WaveTp"]),
        "time": t[:n],
        "displacement": x[:n],
        "velocity": v[:n],
        "CT_prime": Ct_p[:n],
        "CT": Ct[:n],
    })
    out_csv_ts = os.path.join(out_path,out_csv_ts)
    df_ts.to_csv(out_csv_ts, index=False)

    # 2) metadata CSV (single row)
    df_meta = pd.DataFrame([{
        "case_name": case_name,
        "HWindSpeed": Uref,
        "WaveHs": float(row["WaveHs"]),
        "WaveTp": float(row["WaveTp"]),
        "t_analyze_start": float(t_analyze_start),
        "WF_frange_low": float(WF_frange[0]),
        "WF_frange_high": float(WF_frange[1]),
        "vel_method": cfg["vel_method"],
        "sg_window": int(cfg["sg_window"]),
        "sg_poly": int(cfg["sg_poly"]),
        "welch_nperseg": int(cfg["welch_nperseg"]),
        "welch_overlap_frac": float(cfg["welch_overlap_frac"]),
        "CT_mean": mean_Ct,
        "CT_prime_mean": mean_Ct_p,
        "v_A": float(v_A) if np.isfinite(v_A) else np.nan,
        "v_f": float(v_f) if np.isfinite(v_f) else np.nan,
        "n_samples": int(n),
        "non_dim": non_dim
    }])
    out_csv_meta = os.path.join(out_path,out_csv_meta)
    df_meta.to_csv(out_csv_meta, index=False)

    print(f"Saved timeseries: {out_csv_ts} ({n} rows)")
    print(f"Saved metadata:   {out_csv_meta} (1 row)")
    return df_ts, df_meta


if __name__ == "__main__":
    n_jobs = 6
    WF_frange=(1 / 30, 1 / 1),

    # You can set this True for CT-enabled run
    do_ct_calcs = False
    collect_traces = True

    default_cfg = af.normalize_cfg({
        "detrend": True,
        "vel_method": "savgol",
        "sg_window": 301,
        "sg_poly": 4,
        "welch_nperseg": 3000,
        "welch_overlap_frac": 0.5,
        "do_fft_analysis": False,
    })

    out_path = "HS5_TP8_WS10"
    os.makedirs(out_path, exist_ok=True)

    cases = af.fetch_data.get_cases()
    cases_f = cases[
        (cases["WaveHs"] == 5.0) &
        (cases["WaveTp"] == 8.0) &
        (cases["HWindSpeed"] == 10.0)
    ]
    t_analyze_start = 400

    for seed in range(6):
        # choose one case from filtered df (first one here)
        df_one = export_single_case_from_cases_df(
            cases_f,
            default_cfg,
            fetch_data,
            case_idx=seed,
            t_analyze_start=t_analyze_start,
            out_path = out_path,
            out_csv_ts=f"seed{seed}_timeseries.csv",
            out_csv_meta=f"seed{seed}_metadata.csv",
        )