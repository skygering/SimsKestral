# Big Picture:

# Need a Python routine that breaks the signal into two pieces:

# (1) Fit/remove decaying low-frequency component 
# (2) Extract wave freuqency peak and amplitude from the residual
# (3) Return statistics

import fetch_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

cases = fetch_data.get_cases()

HWindSpeed_list = np.unique(cases["HWindSpeed"])
WaveHs_list = np.unique(cases["WaveHs"])
WaveTp_list = np.unique(cases["WaveTp"])

LF_frange = (1/300, 1/30)
WF_frange = (1/24, 1/1)

n = len(HWindSpeed_list)
all_seed_rows = []
for HWindSpeed in HWindSpeed_list[[10]]:
    # filter wind speed
    wind_cases = cases[cases["HWindSpeed"] == HWindSpeed]
    for WaveHs in WaveHs_list:
        # filter signifigant wave height
        hs_cases = wind_cases[wind_cases["WaveHs"] == WaveHs]
        for WaveTp in WaveTp_list:
            # filter signifigant wave frequency
            tp_cases = hs_cases[hs_cases["WaveTp"] == WaveTp]

            print(f"Selected cases: HWindSpeed = {HWindSpeed}, WaveHs = {WaveHs}, WaveTp = {WaveTp}")

            fig, axes = plt.subplots(nrows = 2)
            ax1, ax2 = axes
            for i, row in tp_cases.iterrows():
                seed = row["WaveSeed1"]
                print(seed)
                case_name = row["case_name"]

                # get data
                file = (fetch_data.DATA_PATH + "outfiles/" + case_name + ".out")
                data = fetch_data.load_outfile(file)
                # t, x_surge = data["Time"], data["PtfmSurge"]
                t, x_loc = data["Time"], np.sin(np.deg2rad(data["PtfmPitch"])) * fetch_data.HUB + data["PtfmSurge"]
                x_surge = scipy.signal.detrend(x_loc, type="linear")
                # get fft values
                f_fft, psd_fft = fetch_data.fft_psd_density(t, x_surge)

                # get welch values
                f_welch, psd_welch = fetch_data.welch_psd_density(t, x_surge)

                # plot timeseries and fft data
                ax1.plot(t, x_surge, label=f"Seed={seed}")
                ax2.loglog(f_fft[1:], psd_fft[1:])  # skip f=0

                # LF/WF metrics from FFT
                f_lf_fft, std_lf_fft, A_lf_fft = fetch_data.band_metrics(f_fft, psd_fft, LF_frange)
                f_wf_fft, std_wf_fft, A_wf_fft = fetch_data.band_metrics(f_fft, psd_fft, WF_frange)

                # LF/WF metrics from Welch
                f_lf_welch, std_lf_welch, A_lf_welch = fetch_data.band_metrics(f_welch, psd_welch, LF_frange)
                f_wf_welch, std_wf_welch, A_wf_welch = fetch_data.band_metrics(f_welch, psd_welch, WF_frange)

                # worst-case half range
                Ax_worst = 0.5 * (np.max(x_surge) - np.min(x_surge))

                all_seed_rows.append({
                    "HWindSpeed": HWindSpeed,
                    "WaveHs": WaveHs,
                    "WaveTp": WaveTp,
                    "WaveSeed1": seed,
                    "case_name": case_name,

                    "Ax_worst": Ax_worst,

                    "f_lf_fft": f_lf_fft,
                    "f_wf_fft": f_wf_fft,
                    "std_lf_fft": std_lf_fft,
                    "std_wf_fft": std_wf_fft,
                    "A_lf_fft": A_lf_fft,
                    "A_wf_fft": A_wf_fft,

                    "f_lf_welch": f_lf_welch,
                    "f_wf_welch": f_wf_welch,
                    "std_lf_welch": std_lf_welch,
                    "std_wf_welch": std_wf_welch,
                    "A_lf_welch": A_lf_welch,
                    "A_wf_welch": A_wf_welch,
                })
            # plot values across all seeds
            fetch_data.plot_style(fig, ax1, ax2, HWindSpeed, WaveHs, WaveTp)
 
# Build one seed-level table for all runs
seed_df = pd.DataFrame(all_seed_rows)
seed_df.to_csv("seed_level_stats_all_cases.csv", index=False)

# Build one condition-level summary for all runs
summary_df = (
    seed_df.groupby(["HWindSpeed", "WaveHs", "WaveTp"], as_index=False)
    .agg(
        n_seeds=("WaveSeed1", "count"),

        f_lf_fft_mean=("f_lf_fft", "mean"), f_lf_fft_std=("f_lf_fft", "std"),
        f_wf_fft_mean=("f_wf_fft", "mean"), f_wf_fft_std=("f_wf_fft", "std"),
        A_lf_fft_mean=("A_lf_fft", "mean"), A_lf_fft_std=("A_lf_fft", "std"),
        A_wf_fft_mean=("A_wf_fft", "mean"), A_wf_fft_std=("A_wf_fft", "std"),

        f_lf_welch_mean=("f_lf_welch", "mean"), f_lf_welch_std=("f_lf_welch", "std"),
        f_wf_welch_mean=("f_wf_welch", "mean"), f_wf_welch_std=("f_wf_welch", "std"),
        A_lf_welch_mean=("A_lf_welch", "mean"), A_lf_welch_std=("A_lf_welch", "std"),
        A_wf_welch_mean=("A_wf_welch", "mean"), A_wf_welch_std=("A_wf_welch", "std"),

        Ax_worst_mean=("Ax_worst", "mean"), Ax_worst_std=("Ax_worst", "std"),
        Ax_worst_max=("Ax_worst", "max"),
    )
)

summary_df.to_csv("condition_summary_stats_all_cases.csv", index=False)