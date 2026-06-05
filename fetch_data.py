import re
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import scipy.signal as ssig

# constants for data processing
NEEDED_COLS = [
    "Time",
    "PtfmSurge",
    "PtfmPitch",
    "PtfmYaw",
    "RtFldCt",
    "RtFldFxh",
    "RtVAvgxh",
]

RHO = 1.225
R = 120.0
HUB = 150.0
A = np.pi * R**2

DATA_PATH = "/projects/floatingweis/dzalkind/datasets/0_full_sweep_mediterranean/"

def clean_line(line):
    return re.sub(r"\(\s*([^,]+),\s*([^)]+)\)", r"\2", line)

def load_outfile(file):
    with open(file, "r") as f:
        for _ in range(6):
            next(f)
        columns = f.readline().split()
        units = f.readline().split()

    usecols = [
        i for i, c in enumerate(columns)
        if c in NEEDED_COLS
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

def get_cases(IECTurbc = 0.005):
    with open(DATA_PATH + "case_matrix_combined.txt") as f:
        lines = [clean_line(l.strip()) for l in f]

    cases = pd.read_csv(
        StringIO("\n".join(lines)),
        sep=r"\s+",
    )
    cases = cases[cases["IECTurbc"] == IECTurbc]
    print(f"Total cases (IECTurbc = {IECTurbc}): {len(cases)}")
    return cases

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
        x, fs=fs, window="hann", nperseg=min(8192, len(x)),
        noverlap=min(8192, len(x))//2, detrend="constant", scaling="density"
    )
    return f, psd

    
def plot_style(fig, ax1, ax2, HWindSpeed, WaveHs, WaveTp):
    ax2.axvline(1.0 / WaveTp, color="k", ls="--", label="1/Tp")
    ax2.text(x = 1.0 / WaveTp + 0.1, y = 0.05, s = "$f = 1 / T_P$", color="k")
    ax2.set_ylabel("Surge PSD")
    ax2.set_xlabel("Frequencies [1/s]")
    # after plotting
    ax1.legend(loc="upper left", bbox_to_anchor=(0.05, 1.4), ncols = 3)
    ax1.set_ylabel("Surge Displacement [m]")
    ax1.set_xlabel("Time [s]")

    fig.suptitle(f"HWindSpeed = {HWindSpeed}, WaveHs = {WaveHs}, WaveTp = {WaveTp}", y = 1.1)
    fig.subplots_adjust(hspace=0.4, right=0.8, top=0.9, bottom=0.1)
    fig.savefig(
        "fig/ws" + str(HWindSpeed) + "_hs" + str(WaveHs) + "_tp" + str(WaveTp) + ".png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2
    )
    plt.close()

def band_metrics(f, Pxx, frange):
    m = (f >= frange[0]) & (f <= frange[1])
    if np.sum(m) < 2:
        return np.nan, np.nan, np.nan
    f_peak = float(f[m][np.argmax(Pxx[m])])
    var_band = float(np.trapezoid(Pxx[m], f[m]))
    std_band = np.sqrt(max(var_band, 0.0))
    Aeq_band = np.sqrt(2.0) * std_band
    return f_peak, std_band, Aeq_band



