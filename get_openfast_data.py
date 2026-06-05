# # module load python/3.11.4
# # poetry install

import re
import os
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.signal import savgol_filter

# data_path = "/projects/floatingweis/dzalkind/datasets/0_full_sweep_mediterranean/"

# def clean_line(line):
#     line = re.sub(r"\(\s*([^,]+),\s*([^)]+)\)", r"\2", line)
#     return line


# def sinusoid_amplitude_fft(x, dt):
#     x = np.asarray(x)
#     x = x - np.mean(x)

#     n = len(x)
#     # take fft
#     fft_vals = np.fft.rfft(x)
#     freqs = np.fft.rfftfreq(n, d=dt)
#     psd = np.abs(fft_vals)**2
#     # ignore DC
#     psd[0] = 0
#     # dominant frequency index
#     k = np.argmax(psd)
#     # amplitude from FFT coefficient
#     # scaling for real sinusoid
#     A = 2 * np.abs(fft_vals[k]) / n

#     return {
#         "amplitude": A,
#         "dom_freq": freqs[k],
#         "dom_period": 1 / freqs[k]
#     }

# import numpy as np
# from scipy.signal import welch


def motion_stats(x, dt, peak_band_fraction=0.2, i = "0"):
    """
    Statistics of a stochastic motion signal.

    Parameters
    ----------
    x : array_like
        Time series.
    dt : float
        Time step [s].
    peak_band_fraction : float
        Fractional width around dominant frequency used
        for band-limited RMS.

    Returns
    -------
    dict
    """

    x = np.asarray(x)
    x = x - np.mean(x)

    fs = 1.0 / dt
    print(dt)

    f, Sxx = welch(
        x, fs=fs,
        window='hann', nperseg=4096, noverlap=2048,
    )


    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(10,4),
        constrained_layout=True
    )

    # Time series
    t = np.arange(len(x))*dt

    ax1.plot(t, x)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Surge [m]")
    ax1.set_title("Surge Time Series")
    ax1.grid(True)

    # PSD
    ax2.loglog(f, Sxx)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("PSD [m$^2$/Hz]")
    ax2.set_title("Power Spectral Density")
    ax2.grid(True)

    fig.savefig(
        os.path.join("fig", f"surge_psd_{i}.png"),
        bbox_inches="tight"
    )

    plt.close(fig)

    # ignore DC
    peak_idx = np.argmax(Sxx[1:]) + 1

    f_peak = f[peak_idx]
    print(f_peak)

    # total variance
    variance = np.trapezoid(Sxx, f)

    rms = np.sqrt(variance)

    equivalent_amplitude = np.sqrt(2 * variance)

    # ------------------------
    # band-limited variance
    # ------------------------

    f_low = (1 - peak_band_fraction) * f_peak
    f_high = (1 + peak_band_fraction) * f_peak

    mask = (f >= f_low) & (f <= f_high)

    peak_variance = np.trapezoid(Sxx[mask], f[mask])

    peak_rms = np.sqrt(peak_variance)

    peak_equivalent_amplitude = np.sqrt(2 * peak_variance)

    return {
        "variance": variance,
        "rms": rms,
        "equivalent_amplitude": equivalent_amplitude,
        "dominant_frequency": f_peak,
        "dominant_period": 1 / f_peak,

        "peak_variance": peak_variance,
        "peak_rms": peak_rms,
        "peak_equivalent_amplitude": peak_equivalent_amplitude,
        "peak_energy_fraction": peak_variance / variance,
    }


# print("loading data")
# with open(data_path + "case_matrix_combined.txt") as f:
#     lines = [clean_line(l.strip()) for l in f]
# df = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+")
# print("filtering data")
# df = df[
#     (df["IECTurbc"] == 0.005)
#     & (df["HWindSpeed"] == 10.0)
# ]
# df = (
#     df.sort_values(("WaveSeed1"))
#       .drop_duplicates(
#           subset=["WaveHs", "WaveTp"],
#         )
# )
# files = df["case_name"]
# waveHs = df["WaveHs"]
# waveTp = df["WaveTp"]

# needed_cols = ["Time", "PtfmSurge", "PtfmPitch", "RotThrust", "GenTq", "RotSpeed", "GenPwr",
#     "RotTorq", "RotSpeed", "RtVAvgxh", "RtFldCt", "RtFldCp", "RtFldFxh", "RtFldFyh", "RtFldFzh",
# ]

# for i in np.arange(len(files)):
#     file = data_path + 'outfiles/' + files.iloc[i] + '.out'
#     Hs = waveHs.iloc[i]
#     Tp = waveTp.iloc[i]
#     print(Hs)
#     print(Tp)
#     # --- read metadata ---
#     with open(file, "r") as f:
#         for _ in range(6):
#             next(f)
#         columns = f.readline().split()
#         units = f.readline().split()
#     # --- read numeric data ---
#     df = pd.read_csv(
#         file,
#         skiprows=8,
#         sep = r"\s+",
#         header=None,
#         names=columns,
#         engine="python",
#     )
#     # --- attach units as metadata ---
#     df.attrs["units"] = dict(zip(columns, units))

#     rho = 1.225   # kg/m^3
#     R = 120 # m
#     A = np.pi * R**2
#     Ct_prime = df["RtFldFxh"] / (0.5 * rho * A * df["RtVAvgxh"]**2)

#     dt = df["Time"][1] - df["Time"][0]
#     print("Surge")
#     print(motion_stats(df["PtfmSurge"], dt))
#     print("Pitch")
#     print(motion_stats(df["PtfmPitch"], dt))
#     print("CT")
#     print(np.mean(df["RtFldCt"]), np.std(df["RtFldCt"]))
#     print("CT'")
#     print(np.mean(Ct_prime), np.std(Ct_prime))
#     print("Yaw")
#     print(np.mean(df["PtfmYaw"]), np.std(df["PtfmYaw"]))

# module load python/3.11.4
# poetry install


# =============================================================================
# SETTINGS
# =============================================================================

data_path = "/projects/floatingweis/dzalkind/datasets/0_full_sweep_mediterranean/"

rho = 1.225
R = 120.0
A = np.pi * R**2


needed_cols = [
    "Time",
    "PtfmSurge",
    "PtfmPitch",
    "PtfmYaw",
    "RtFldCt",
    "RtFldFxh",
    "RtVAvgxh",
]


# =============================================================================
# HELPERS
# =============================================================================

def clean_line(line):
    return re.sub(r"\(\s*([^,]+),\s*([^)]+)\)", r"\2", line)


# def motion_stats(x, dt, peak_band_fraction=0.2):
#     """
#     PSD-based motion statistics.
#     """

#     x = np.asarray(x)
#     x = x - np.mean(x)

#     fs = 1.0 / dt

#     f, Sxx = welch(
#         x,
#         fs=fs,
#         nperseg=min(len(x) // 4, 4096),
#     )

#     peak_idx = np.argmax(Sxx[1:]) + 1
#     f_peak = f[peak_idx]

#     variance = np.trapezoid(Sxx, f)

#     rms = np.sqrt(variance)

#     equivalent_amplitude = np.sqrt(2 * variance)

#     f_low = (1 - peak_band_fraction) * f_peak
#     f_high = (1 + peak_band_fraction) * f_peak

#     mask = (f >= f_low) & (f <= f_high)

#     peak_variance = np.trapezoid(Sxx[mask], f[mask])

#     peak_rms = np.sqrt(peak_variance)

#     peak_equivalent_amplitude = np.sqrt(2 * peak_variance)

#     peak_idx = np.argmax(Sxx[1:]) + 1

#     print(
#         "f_peak =", f[peak_idx],
#         "PSD =", Sxx[peak_idx]
#     )

#     return {
#         "variance": variance,
#         "rms": rms,
#         "equivalent_amplitude": equivalent_amplitude,
#         "dominant_frequency": f_peak,
#         "dominant_period": 1.0 / f_peak,
#         "peak_variance": peak_variance,
#         "peak_rms": peak_rms,
#         "peak_equivalent_amplitude": peak_equivalent_amplitude,
#         "peak_energy_fraction": peak_variance / variance,
#     }


def load_outfile(file, needed_cols):

    with open(file, "r") as f:

        for _ in range(6):
            next(f)

        columns = f.readline().split()
        units = f.readline().split()

    usecols = [
        i for i, c in enumerate(columns)
        if c in needed_cols
    ]

    df = pd.read_csv(
        file,
        skiprows=8,
        sep=r"\s+",
        header=None,
        names=[columns[i] for i in usecols],
        usecols=usecols,
        engine="c",
    )

    df.attrs["units"] = {
        columns[i]: units[i]
        for i in usecols
    }

    return df


def make_heatmap(ax, summary, value, title):

    pivot = summary.pivot(
        index="Hs",
        columns="Tp",
        values=value,
    )

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Tp [s]")
    ax.set_ylabel("Hs [m]")
    ax.set_title(title)

    plt.colorbar(im, ax=ax)


# =============================================================================
# LOAD CASE MATRIX
# =============================================================================

print("Loading case matrix...")

with open(data_path + "case_matrix_combined.txt") as f:
    lines = [clean_line(l.strip()) for l in f]

cases = pd.read_csv(
    StringIO("\n".join(lines)),
    sep=r"\s+",
)

print(f"Total cases: {len(cases)}")


# =============================================================================
# FILTER TO ONE SEED PER (Hs,Tp)
# =============================================================================

cases = cases[
    (cases["IECTurbc"] == 0.005)
    & (cases["HWindSpeed"] == 5.0)
]

cases = (
    cases
    .sort_values("WaveSeed1")
    .drop_duplicates(
        subset=["WaveHs", "WaveTp"]
    )
)

cases = cases.sort_values(
    ["WaveHs", "WaveTp"]
)

print(f"Selected cases: {len(cases)}")
print()


# =============================================================================
# PROCESS CASES
# =============================================================================

results = []

for i, row in cases.iterrows():

    case_name = row["case_name"]

    file = (
        data_path
        + "outfiles/"
        + case_name
        + ".out"
    )

    Hs = row["WaveHs"]
    Tp = row["WaveTp"]

    print(
        f"Processing "
        f"Hs={Hs:4.3f} "
        f"Tp={Tp:4.1f}"
    )

    data = load_outfile(
        file,
        needed_cols,
    )
    data = data[data["Time"] > 400.0]

    dt = (
        data["Time"].iloc[1]
        - data["Time"].iloc[0]
    )

    surge_disp = data["PtfmSurge"]
    surge_vel = np.gradient(surge_disp, dt)

    # x_s = savgol_filter(data["PtfmSurge"], window_length=101, polyorder=3)  # tune window
    # v = savgol_filter(x, window_length=101, polyorder=3, deriv=1, delta=dt)
    # surge_stats = motion_stats(
    #     surge_vel,
    #     dt,
    #     i = str(i),
    # )
    t = data["Time"].to_numpy(dtype=float)
    x = data["PtfmSurge"].to_numpy(dtype=float)

    # Always compute dt from time channel
    dt = np.mean(np.diff(t))

    # Smooth + differentiate (more stable than raw gradient)
    x_s = savgol_filter(x, window_length=101, polyorder=3)  # tune window
    v = savgol_filter(x, window_length=101, polyorder=3, deriv=1, delta=dt)

    surge_stats = motion_stats(
        v,          # only if motion_stats expects velocity
        dt,
        i=str(i),
    )

    # pitch_stats = motion_stats(
    #     data["PtfmPitch"],
    #     dt,
    # )

    # Ct_prime = (
    #     data["RtFldFxh"]
    #     / (0.5 * rho * A * data["RtVAvgxh"]**2)
    # )

    results.append({
        "i": str(i),
        "Hs": Hs,
        "Tp": Tp,

        # surge
        # "surge_amp":
        #     surge_stats["equivalent_amplitude"],

        # "surge_period":
        #     surge_stats["dominant_frequency"],

        # "surge_variance":
        #     surge_stats["variance"],

        # "surge_peak_variance":
        #     surge_stats["peak_variance"],

        # "surge_energy_fraction":
        #     surge_stats["peak_energy_fraction"],

        # # pitch
        # "pitch_offset":
        #     data["PtfmPitch"].mean(),

        # "pitch_amp":
        #     pitch_stats["equivalent_amplitude"],

        # "pitch_period":
        #     pitch_stats["dominant_frequency"],

        # "pitch_energy_fraction":
        #     pitch_stats["peak_energy_fraction"],

        # # yaw
        # "yaw_offset":
        #     data["PtfmYaw"].mean(),

        # "yaw_std":
        #     data["PtfmYaw"].std(),

        # CT
        "Ct_mean":
            data["RtFldCt"].mean(),

        "Ct_std":
            data["RtFldCt"].std(),

        # # CT'
        # "CtPrime_mean":
        #     Ct_prime.mean(),

        # "CtPrime_std":
        #     Ct_prime.std(),
    })


summary = pd.DataFrame(results)

print()
print(summary.head(n=12))


# =============================================================================
# SAVE SUMMARY
# =============================================================================

summary.to_csv(
    "summary_wave_cases.csv",
    index=False,
)


# =============================================================================
# FIGURE 1
# =============================================================================

# fig, axs = plt.subplots(
#     2,
#     2,
#     figsize=(12, 10),
#     constrained_layout=True,
# )

# make_heatmap(
#     axs[0, 0],
#     summary,
#     "surge_amp",
#     "Surge Amplitude [m]",
# )

# make_heatmap(
#     axs[0, 1],
#     summary,
#     "surge_period",
#     "Surge Dominant Period [s]",
# )

# make_heatmap(
#     axs[1, 0],
#     summary,
#     "pitch_amp",
#     "Pitch Amplitude [deg]",
# )

# make_heatmap(
#     axs[1, 1],
#     summary,
#     "pitch_period",
#     "Pitch Dominant Period [s]",
# )

# plt.savefig(
#     "motion_summary.png",
#     dpi=300,
# )

# plt.show()


# =============================================================================
# FIGURE 2
# =============================================================================

# fig, axs = plt.subplots(
#     1,
#     2,
#     figsize=(10, 4),
#     constrained_layout=True,
# )

# make_heatmap(
#     axs[0],
#     summary,
#     "Ct_mean",
#     "Mean Ct",
# )

# make_heatmap(
#     axs[1],
#     summary,
#     "CtPrime_mean",
#     "Mean Ct'",
# )

# plt.savefig(
#     "ct_summary.png",
#     dpi=300,
# )

# plt.show()
