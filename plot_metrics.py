import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# ============================================================
# Config
# ============================================================
CSV_PATH = "condition_summary_stats_all_cases.csv"

TP_VALUES = [6, 8, 10, 12]
HS_VALUES = [2, 3, 5]

D = 240.0
NON_DIM = True          # <-- Toggle here
ATOL = 1e-6

sns.set_theme(style="whitegrid", context="talk")


# ============================================================
# Helpers: Data prep
# ============================================================
def filter_by_tp_hs(df, tp_values, hs_values, atol=1e-6):
    """Robustly filter dataframe to requested Tp and Hs values."""
    tp_arr = np.asarray(tp_values, dtype=float)
    hs_arr = np.asarray(hs_values, dtype=float)

    tp_mask = np.isclose(df["WaveTp"].to_numpy()[:, None], tp_arr[None, :], atol=atol).any(axis=1)
    hs_mask = np.isclose(df["WaveHs"].to_numpy()[:, None], hs_arr[None, :], atol=atol).any(axis=1)

    return df.loc[tp_mask & hs_mask].copy()


def build_long_metric_df(df, mapping, value_col_name, std_col_name, group_col_name):
    """
    Build a long-format dataframe from a mapping:
      mapping = {
        "LabelA": ("value_col_A", "std_col_A"),
        "LabelB": ("value_col_B", "std_col_B"),
      }
    """
    base_cols = ["HWindSpeed", "WaveHs", "WaveTp"]
    parts = []

    for label, (val_col, std_col) in mapping.items():
        tmp = df[base_cols + [val_col, std_col]].copy()
        tmp = tmp.rename(columns={val_col: value_col_name, std_col: std_col_name})
        tmp[group_col_name] = label
        parts.append(tmp)

    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=[value_col_name]).copy()
    out[std_col_name] = out[std_col_name].fillna(0.0)
    return out


def prepare_metric_for_plot(
    data,
    value_col,
    std_col,
    metric_type,   # "frequency" or "amplitude"
    nondim=False,
    D=240.0,
    u_col="HWindSpeed",
):
    """
    Returns:
      out_df with y_plot and ystd_plot
      y_col_name, ystd_col_name, ylabel
    """
    out = data.copy()
    out["y_plot"] = out[value_col].astype(float)
    out["ystd_plot"] = out[std_col].astype(float)

    if not nondim:
        if metric_type == "frequency":
            ylabel = "Welch WF Frequency [Hz]"
        elif metric_type == "amplitude":
            ylabel = "Surge velocity amplitude [m/s]"
        else:
            raise ValueError("metric_type must be 'frequency' or 'amplitude'")
        return out, "y_plot", "ystd_plot", ylabel

    # Non-dimensional scaling
    U = out[u_col].astype(float).to_numpy()
    U_safe = np.where(np.abs(U) > 1e-12, U, np.nan)  # avoid divide-by-zero

    if metric_type == "frequency":
        # f_non_dim = f * D / U
        scale = D / U_safe
        ylabel = "Non-dimensional frequency, $fD/U_\infty$ [-]"
    elif metric_type == "amplitude":
        # A_non_dim = A_v / U
        scale = 1.0 / U_safe
        ylabel = "Non-dimensional amplitude, $A_v/U_\infty$ [-]"
    else:
        raise ValueError("metric_type must be 'frequency' or 'amplitude'")

    out["y_plot"] = out["y_plot"].to_numpy() * scale
    out["ystd_plot"] = out["ystd_plot"].to_numpy() * np.abs(scale)  # std scales same way

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_plot"]).copy()
    return out, "y_plot", "ystd_plot", ylabel


def add_centered_offsets(df, x_col, group_col, group_order=None, half_width=0.15, out_col="x_plot"):
    """Offset x-position by category to avoid overlap."""
    out = df.copy()

    if group_order is None:
        group_order = list(out[group_col].dropna().unique())

    if len(group_order) <= 1:
        offsets = {group_order[0]: 0.0} if len(group_order) == 1 else {}
    else:
        vals = np.linspace(-half_width, half_width, len(group_order))
        offsets = {k: v for k, v in zip(group_order, vals)}

    out[out_col] = out[x_col] + out[group_col].map(offsets)
    return out, offsets


def draw_errorbars(ax, data, x_col, y_col, yerr_col, color_fn, lw=1.2, capsize=2.5, alpha=0.9):
    """Draw row-wise y-error bars."""
    if data.empty:
        return

    mask = np.isfinite(data[yerr_col].to_numpy()) & (data[yerr_col].to_numpy() > 0)
    for _, r in data.loc[mask].iterrows():
        ax.errorbar(
            r[x_col], r[y_col],
            yerr=r[yerr_col],
            fmt="none",
            ecolor=color_fn(r),
            elinewidth=lw,
            capsize=capsize,
            alpha=alpha
        )


# ============================================================
# Plot layout A: 3x4 grid (rows=Hs, cols=Tp), x=HWindSpeed
# ============================================================
def plot_grid_hs_tp(
    data,
    tp_values,
    hs_values,
    group_col,
    y_col,
    ystd_col,
    ylabel,
    outfile,
    title,
    markers=None,
    x_col="HWindSpeed",
    x_offset=0.15,
    atol=1e-6,
    add_tp_reference_line=False,   # meaningful only for dimensional frequency
):
    group_order = list(data[group_col].dropna().unique())
    plot_df, _ = add_centered_offsets(
        data,
        x_col=x_col,
        group_col=group_col,
        group_order=group_order,
        half_width=x_offset,
        out_col="x_plot"
    )

    palette = dict(zip(group_order, sns.color_palette("Set2", n_colors=len(group_order))))

    g = sns.FacetGrid(
        plot_df,
        row="WaveHs",
        col="WaveTp",
        row_order=hs_values,
        col_order=tp_values,
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=2.9,
        aspect=1.15,
        despine=False,
    )

    for hs in hs_values:
        for tp in tp_values:
            ax = g.axes_dict[(hs, tp)]
            sub = plot_df[
                np.isclose(plot_df["WaveHs"], hs, atol=atol) &
                np.isclose(plot_df["WaveTp"], tp, atol=atol)
            ].copy()

            if sub.empty:
                ax.set_visible(False)
                continue

            sns.scatterplot(
                data=sub,
                x="x_plot",
                y=y_col,
                hue=group_col,
                style=group_col,
                hue_order=group_order,
                style_order=group_order,
                palette=palette,
                markers=markers,
                s=105,
                alpha=0.85,
                linewidth=0.8,
                legend=False,
                ax=ax
            )

            draw_errorbars(
                ax=ax,
                data=sub,
                x_col="x_plot",
                y_col=y_col,
                yerr_col=ystd_col,
                color_fn=lambda r: palette[r[group_col]],
            )

            # Optional horizontal reference: 1/Tp
            if add_tp_reference_line:
                ax.axhline(1.0 / tp, ls="--", c="crimson", lw=1.1, alpha=0.85)

            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=0.25)

    # remove per-axis labels
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    # one label for whole figure
    fig = g.figure
    fig.supxlabel("HWindSpeed [m/s]")   # or "Tp [s]" in your second layout
    fig.supylabel(ylabel)
    g.set_titles(row_template="$H_S = {row_name}$ [m]", col_template="$T_P = {col_name}$ [s]")

    # custom legend
    legend_handles = []
    for name in group_order:
        mk = markers.get(name, "o") if markers else "o"
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=mk, linestyle="",
                markerfacecolor=palette[name],
                markeredgecolor="black",
                markersize=12,
                label=name
            )
        )

    fig = g.figure
    fig.legend(
        handles=legend_handles,
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        frameon=False,
        ncols = 3
    )
    fig.suptitle(title, y=1.02, fontsize=20)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    df_raw = pd.read_csv(CSV_PATH)
    df = filter_by_tp_hs(df_raw, TP_VALUES, HS_VALUES, atol=ATOL)

    # ---------- Build long frequency dataframe ----------
    freq_mapping = {
        "Displacement": ("x_wel_ens_f_wf", "x_wel_ens_f_wf_std"),
        "Velocity": ("v_wel_ens_f_wf", "v_wel_ens_f_wf_std"),
    }
    freq_df = build_long_metric_df(
        df=df,
        mapping=freq_mapping,
        value_col_name="Frequency",
        std_col_name="FrequencyStd",
        group_col_name="Signal",
    )

    # ---------- Build long amplitude dataframe ----------
    amp_mapping = {
        "From x-Welch ($2 \pi f \cdot A_x$)": ("Av_from_Ax_wf_wel_ens", "Av_from_Ax_wf_wel_ens_std"),
        "From v-Welch": ("v_wel_ens_A_wf", "v_wel_ens_A_wf_std"),
        "Worst-case (800-1000s)": ("v_wc_amp_mean", "v_wc_amp_std"),
    }
    amp_df = build_long_metric_df(
        df=df,
        mapping=amp_mapping,
        value_col_name="Amplitude",
        std_col_name="AmplitudeStd",
        group_col_name="Source",
    )

    # ---------- Optional non-dimensionalization ----------
    freq_plot_df, freq_y, freq_ystd, freq_ylabel = prepare_metric_for_plot(
        data=freq_df,
        value_col="Frequency",
        std_col="FrequencyStd",
        metric_type="frequency",
        nondim=NON_DIM,
        D=D,
        u_col="HWindSpeed",
    )

    amp_plot_df, amp_y, amp_ystd, amp_ylabel = prepare_metric_for_plot(
        data=amp_df,
        value_col="Amplitude",
        std_col="AmplitudeStd",
        metric_type="amplitude",
        nondim=NON_DIM,
        D=D,
        u_col="HWindSpeed",
    )

    suffix = "nondim" if NON_DIM else "dim"

    freq_markers = {"Displacement": "o", "Velocity": "s"}
    amp_markers = {
        "From x-Welch ($2 \pi f \cdot A_x$)": "o",
        "From v-Welch": "s",
        "Worst-case (800-1000s)": "^",
    }

    # ========================================================
    # Layout A: 3x4 grid (Hs x Tp), x = wind speed
    # ========================================================
    plot_grid_hs_tp(
        data=freq_plot_df,
        tp_values=TP_VALUES,
        hs_values=HS_VALUES,
        group_col="Signal",
        y_col=freq_y,
        ystd_col=freq_ystd,
        ylabel=freq_ylabel,
        outfile=f"freq_grid_hs_tp_3x4_{suffix}.png",
        title=f"Frequency vs Wind Speed",
        markers=freq_markers,
        x_col="HWindSpeed",
        x_offset=0.15,
        atol=ATOL,
        add_tp_reference_line=(not NON_DIM),   # 1/Tp line only for dimensional frequency
    )

    plot_grid_hs_tp(
        data=amp_plot_df,
        tp_values=TP_VALUES,
        hs_values=HS_VALUES,
        group_col="Source",
        y_col=amp_y,
        ystd_col=amp_ystd,
        ylabel=amp_ylabel,
        outfile=f"amp_grid_hs_tp_3x4_{suffix}.png",
        title=f"Surge Velocity Amplitude vs Wind Speed",
        markers=amp_markers,
        x_col="HWindSpeed",
        x_offset=0.18,
        atol=ATOL,
        add_tp_reference_line=False,
    )

    