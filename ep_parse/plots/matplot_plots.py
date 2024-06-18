import matplotlib.pyplot as plt
from math import isnan
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

import ep_parse.plots.common as ppc
import ep_parse.utils as u
import logging

log = logging.getLogger(__name__)

plt.style.use("dark_background")


def _plot_meta_features(frame: dict, channels_in_plot: list[str], y_offsets: list[float], axis, opts: dict = {}):
    cs_org_pairs = frame.get("cs_org_pairs")
    if cs_org_pairs is not None and opts.get("show_cs_org"):
        ch_to_offset = dict(zip(channels_in_plot, y_offsets))
        for org_row in cs_org_pairs.values.tolist():
            x_vals = org_row[2:4]  # centroids for the channels
            y_vals = [ch_to_offset[x] for x in org_row[0:2]]
            axis.plot(x_vals, y_vals)

    qrs_intervals = frame.get("qrs_intervals")
    if qrs_intervals is not None and opts.get("qrs_intervals"):
        ydiff = abs(y_offsets[1] - y_offsets[0])
        yz = [y_offsets[0] - ydiff, y_offsets[-1] + ydiff]
        for s, e in qrs_intervals:
            axis.plot([s, s], yz, color="deeppink", alpha=0.75, linewidth=3)
            axis.plot([e, e], yz, color="deeppink", alpha=0.75, linewidth=3)


def _plot_features(features, y_offset: float, axis, plot_height: float = None, opts: dict = {}, channel: str = None):
    """Plot signals and the associated features, potentially within an axis that already contains
    other signals.  Activations are denoted as a horizontal line beneath the signal iff deflections are plotted; otherwise
    they will be a red overlay.

    Args:
        features (dict): feature_name to characteristics (often a dataframe)
        y_offset (float): The value by which the y values need to be shifted
        axis (matplotlib.axes.Axes): The Axes on which to draw
        plot_height (float, optional): Plottable window is assumed to be [y_offset - plot_height, y_offset + plot_height]. Defaults to None.
        opts (dict, optional): additional params. Defaults to empty.
        channel (str): Channel being plotted. Defaults to None
    """

    signal = features["signal"].reset_index(drop=True)
    if y_offset != 0:
        signal = signal + y_offset
    y_margin = plot_height or 0.25
    _inc = y_margin / 3
    deflection_rendering = None if features.get("deflections") is None else opts.get("show_deflections")

    axis.plot(signal, color=ppc.raw_sig_color(channel), zorder=0)

    if deflection_rendering:
        for d in features["deflections"]:
            c = "yellow" if deflection_rendering == "extra_wavelet_only" else ("r" if i % 2 else "c")
            axis.plot(range(d.start_index, d.end_index), signal.iloc[d.start_index : d.end_index], color=c, zorder=1)

    if features.get("wavelets") is not None:
        show_deflections = deflection_rendering == "all"
        for i, row in features["wavelets"].iterrows():
            s = int(row["start_index"])
            e = int(row["end_index"] + 1)
            yz = [y_offset - y_margin + _inc] * (e - s) if show_deflections else signal.iloc[s:e]
            axis.plot(range(s, e), yz, color="y" if show_deflections else "r", zorder=1)
            if opts.get("index_wavelets"):
                axis.annotate(str(i), (int((s + e) / 2), y_offset + _inc - y_margin))
            if opts.get("activation_centroids"):
                axis.plot(row["centroid_idx"], y_offset + y_margin - _inc, marker=".", zorder=2)

    if features.get("annotations") is not None:
        color = "m"
        linewidth = 2
        for _, row in features["annotations"].iterrows():
            s = int(row["start_index"])
            e = int(row["end_index"] + 1)
            if row["type"].startswith("juicy_"):
                color = ppc.disease_color(float(row["type"][-1]))
                linewidth = 4

            axis.plot(
                range(s, e),
                [y_offset + (_inc * 2)] * (e - s),
                color=color,
                linewidth=linewidth,
            )

    if features.get("disease_scores") is not None:
        for (s, e), dscore in features["disease_scores"].items():
            axis.plot(
                range(s, e),
                [y_offset + (_inc * 3)] * (e - s),
                color=ppc.disease_color(dscore),
                linewidth=4,
            )


def _num_str(n: float, round_to: int = 2) -> str:
    return "-" if isnan(n) else round(n, round_to)


def _configure_signals_plot(
    x_index: pd.Index,
    y_axis: list[int] = (-2, 2),
    title: str = None,
    figsize: tuple = None,
    custom_fig_ax: tuple = None,
):
    fig, ax = custom_fig_ax or plt.subplots(nrows=1, ncols=1, figsize=(figsize or (40, 5)))
    if len(x_index) < 10:
        return fig, ax
    ax.set_title(title)
    ax.set(ylim=y_axis, xlim=(0, len(x_index)))
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.xaxis.grid(color="gray", linestyle="dashed", which="both")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    PPS = u.points_per_second(x_index)
    x_ticks = range(0, len(x_index), PPS)
    ax.set_xticks(x_ticks)
    xlabels = [x_index[i].round(freq="s").isoformat()[11:19] for i in x_ticks]
    ax.set_xticklabels(xlabels)
    ax.minorticks_on()

    return fig, ax


def _warn_for_unknown_channels(channels):
    unknown = set(channels).difference(set(ppc.DEFAULT_CHANNEL_ORDER))
    if unknown:
        log.warning(f"Channels were requested that are not in DEFAULT CHANNEL ORDER: {unknown}")


def plot_signal(
    features: dict,
    y_axis: list[int] = (-2, 2),
    title: str = None,
    file_name: str = None,
    figsize: list[int] = (40, 5),
    opts: dict = {},
):
    fig, ax = _configure_signals_plot(
        x_index=features["signal"].index,
        y_axis=y_axis,
        title=title,
        figsize=figsize,
    )
    opts = {**{"index_wavelets": True, "activation_centroids": True}, **opts}

    _plot_features(features, 0, ax, opts=opts)

    plt.show()


def _plot_signals(
    channel_to_features: dict,
    frame: dict,
    y_axis: list[int] = (-2, 2),
    title: str = None,
    file_name: str = None,
    figsize: tuple = None,
    custom_fig_ax: tuple = None,
    opts: dict = {},
):
    channels_in_plot = list(channel_to_features.keys())
    a_signal = channel_to_features[channels_in_plot[0]]["signal"]
    y_offsets = ppc.y_shifts(len(channels_in_plot), y_axis)
    fig, ax = _configure_signals_plot(
        a_signal.index,
        y_axis,
        title,
        figsize=figsize,
        custom_fig_ax=custom_fig_ax,
    )

    ax.set_yticks(y_offsets)
    ax.set_yticklabels(channels_in_plot)

    opts = {**{"index_wavelets": False, "activation_centroids": True}, **opts}

    for i, (ch, features) in enumerate(channel_to_features.items()):
        _plot_features(features, y_offsets[i], ax, opts=opts, channel=ch)

    _plot_meta_features(frame, channels_in_plot, y_offsets, ax, opts=opts)

    plt.show()


def plot_frame(
    frame: dict,
    y_axis: list[int] = (-2, 2),
    title: str = None,
    file_name: str = None,
    figsize: tuple = None,
    opts: dict = {},
):
    # use explicit channel order
    _warn_for_unknown_channels(frame["channel_to_features"].keys())
    c2f = {c: frame["channel_to_features"][c] for c in ppc.DEFAULT_CHANNEL_ORDER if c in frame["channel_to_features"]}

    return _plot_signals(
        c2f,
        frame,
        y_axis=y_axis,
        title=title,
        file_name=file_name,
        figsize=figsize,
        opts=opts,
    )


def plot_signals(
    channel_to_features: dict,
    y_axis: list[int] = (-2, 2),
    title: str = None,
    file_name: str = None,
    figsize: tuple = None,
    opts: dict = {},
):
    frame_features = (
        {}
    )  # actually need to generate cs_org values to keep this functioning.  Probably better to simply call plot_frame instead where cs_org is needed
    _warn_for_unknown_channels(channel_to_features.keys())
    c2f = {c: channel_to_features[c] for c in ppc.DEFAULT_CHANNEL_ORDER if c in channel_to_features}
    return _plot_signals(
        c2f,
        frame_features,
        y_axis=y_axis,
        title=title,
        file_name=file_name,
        figsize=figsize,
        opts=opts,
    )


def plot_raw_signals(
    signals=[],
    channels: list = [],
    signals_df=None,
    title: str = "",
    y_axis=(-1, 1),
    filename: str = None,
    figsize: tuple = None,
    opts: dict = {},
):
    "A convenience function for call plot_signals with only the signals"
    assert (signals and channels) or (
        signals_df is not None
    ), "Must provide signals and names or a DataFrame with the raw values"

    if signals:
        sig_to_features = {nm: {"signal": sig} for nm, sig in zip(channels, signals)}
    else:
        _warn_for_unknown_channels(signals_df.columns)
        sig_to_features = {c: {"signal": signals_df[c]} for c in ppc.DEFAULT_CHANNEL_ORDER if c in signals_df}

    _plot_signals(
        sig_to_features,
        {},
        y_axis=y_axis,
        title=title,
        file_name=filename,
        figsize=figsize,
        opts=opts,
    )
