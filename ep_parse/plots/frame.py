import pandas as pd
import plotly.graph_objects as pgo
import plotly.io as pio
import toolz as tz

import ep_parse.plots.common as ppc
import ep_parse.plots.signal as pvs

pio.renderers.default = "notebook_connected"
pd.options.plotting.backend = "plotly"


def _frame_pps(frame: dict) -> int:
    return frame.get("meta", {}).get("points_per_second", 2000)


def _add_frame_features(frame: dict, fig: pgo.Figure, opts: dict = {}) -> pgo.Figure:

    qrs_intervals = frame.get("qrs_intervals")
    if qrs_intervals is not None and opts.get("show_qrs_intervals"):
        for q in qrs_intervals:
            fig.add_vrect(x0=q.start_index, x1=q.end_index, fillcolor="#3a2133", opacity=0.75, line_width=0)

    qt_intervals = frame.get("qt_intervals")
    if qt_intervals and opts.get("show_qt_intervals"):
        for q in qt_intervals:
            fig.add_vrect(x0=q.start_index, x1=q.end_index, fillcolor="#797979", opacity=0.25, line_width=0)

    return fig


def plot_signals(frame: dict, fig: pgo.Figure = None, fig_opts={}, display_opts: dict = {}) -> pgo.Figure:
    channel_to_features = frame["channel_to_features"]
    if len(channel_to_features) == 0:
        return ppc.pretty_layout(fig, pd.Index([]), fig_opts)

    channels_in_plot = list(channel_to_features.keys())
    if yz := fig_opts.get("y_scale"):
        y_ax = [-yz, yz]
    else:
        y_ax = fig_opts.get("y_axis") or [-2, 2]
    y_offsets = ppc.y_shifts(len(channels_in_plot), y_ax)
    fig = ppc.pretty_layout(fig, frame["index"], fig_opts)

    _opts = {**{"index_wavelets": False, "activation_centroids": True}, **display_opts}

    for i, (channel, features) in enumerate(channel_to_features.items()):
        _features = tz.assoc(features, "qrs_intervals", frame.get("qrs_intervals"))
        # if channel in display_opts.get("disease_channels", {}):
        #     _features["disease_scores"] = (frame.get("disease_scores") or {}).get(channel)

        pvs._plot_features(
            _features,
            fig,
            PPS=_frame_pps(frame),
            channel=channel,
            y_offset=y_offsets[i],
            opts=_opts,
            time_index=frame["index"],
        )

    ppc.with_yaxis_traces(fig, trace_offsets=dict(zip(channels_in_plot, y_offsets)), y_range=y_ax)
    _add_frame_features(frame, fig, channels_in_plot, y_offsets, opts=_opts)

    return fig


def plot_frame(frame: dict, fig_opts: dict = {}, display_opts: dict = {}) -> pgo.Figure:
    """Visualize signals and their characteristics and relationships

    Args:
        frame (dict): keys are channel_to_features, qrs_intervals, etc.
        fig_opts (dict, optional): keyword args for configuring the plotly figure (e.g. y_axis, title, figure_height). Defaults to {}.
        display_opts (dict, optional): keyword args for configuring figure contents (e.g. wavelet_centroids, disease_channel). Defaults to {}.

    Returns:
        pgo.Figure: Plotly figure that represents the frame
    """
    # use explicit channel order
    frame["channel_to_features"] = {
        c: frame["channel_to_features"][c] for c in ppc.DEFAULT_CHANNEL_ORDER if c in frame["channel_to_features"]
    }
    return plot_signals(frame, fig_opts=fig_opts, display_opts=display_opts)


def plot_raw_signals(
    signals=[],
    channels: list = [],
    signals_df=None,
    fig_opts: dict = {},
    filename: str = None,
) -> pgo.Figure:
    "A convenience function for calling plot_signals with only the signals"
    assert (signals and channels) or (
        signals_df is not None
    ), "Must provide signals and names or a DataFrame with the raw values"

    if signals:
        sig_to_features = {nm: {"signal": sig} for nm, sig in zip(channels, signals)}
    else:
        sig_to_features = {col: {"signal": signals_df[col]} for col in signals_df.columns}

    if not sig_to_features:
        return

    index = next(iter(sig_to_features.values()))["signal"].index

    # use explicit channel order
    ordered_ch_features = {c: sig_to_features[c] for c in ppc.DEFAULT_CHANNEL_ORDER if c in sig_to_features}

    return plot_signals(
        {"channel_to_features": ordered_ch_features, "index": index}, fig_opts=fig_opts, filename=filename
    )
