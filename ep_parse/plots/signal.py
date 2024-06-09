import numpy as np
import pandas as pd
import plotly.graph_objects as pgo
import plotly.io as pio

import ep_parse.constants as pc
import ep_parse.features.signal as pfs
import ep_parse.plots.common as ppc
import ep_parse.utils as pu

pio.renderers.default = "notebook_connected"
pd.options.plotting.backend = "plotly"


def _defl_hover(d: pc.Deflection) -> str:
    s = f"{d.start_index} : {d.duration}<br>"
    s += f"{d.max_voltage:.5f} mx, {d.slope_per_volts:.5f}<br>"  # replace these with useful features
    s += f"{d.slope:.6f} slp, {d.voltage_change:.4f} vC"
    return s


def _defl_plot_level(deflections: list[pc.Deflection], channel: str, opts: dict) -> str:
    if lvl := deflections and opts.get("show_deflections"):
        if channel in pc.CHANNEL_GROUPS[pc.ChannelGroup.ECG] and lvl != "detailed":
            return False
        return lvl
    return False


def _signal_over_intervals(signal: pd.Series, intervals: list[pc.Interval]) -> np.ndarray:
    y_vals = signal.to_numpy()
    for s, e in pu.intervals_complement(intervals, len(signal)):
        y_vals[s:e] = np.nan
    return y_vals


def _plot_features(
    features: dict,
    fig: pgo.Figure,
    PPS: int,
    channel: str = None,
    y_offset: float = 0,
    opts: dict = {},
    time_index: pd.Index = None,
) -> pgo.Figure:
    if "signal" not in features:
        return fig
    _signal = features["signal"] if not y_offset else (features["signal"] + y_offset)
    time_index = time_index if time_index is not None else features["signal"].index
    _signal = _signal.reset_index(drop=True)
    _step = 0.033
    y_offset -= 0.2
    deflection_rendering = _defl_plot_level(features.get("deflections"), channel, opts)

    clr_hex = ppc.raw_sig_color(channel)
    h_inf, hovertxt = "text", None
    match opts.get("signal_hoverinfo"):
        case "voltage":
            hovertxt = [f"{v:.3} mV" for v in features["signal"].values]
        case "time":
            hovertxt = [pu.as_time_str(v) for v in time_index]
        case "iloc":
            hovertxt = list(range(len(time_index)))
        case "time_iloc":
            hovertxt = [f"{channel} {i} {pu.as_time_str(t)}" for i, t in enumerate(time_index)]
        case "voltage_iloc":
            hovertxt = [f"{v:.4}<br>{i}" for i, v in enumerate(features["signal"].values)]
        case "empty":
            h_inf = "text"
        case _:
            h_inf = "skip"
    fig.add_trace(
        pgo.Scatter(
            x=_signal.index,
            y=_signal,
            name=channel,
            text=hovertxt,
            hoverinfo=h_inf,
            mode="lines",
            line={"color": clr_hex, "width": 1},
        )
    )

    if deflection_rendering:
        if deflection_rendering == "detailed":
            for i, w in enumerate(features["deflections"]):
                hex_clr = "red" if i % 2 else "cyan"
                sub_sig = _signal.loc[w.start_index : w.end_index]
                fig.add_trace(
                    pgo.Scatter(
                        x=sub_sig.index,
                        y=sub_sig,
                        mode="lines",
                        hoverinfo="text",
                        text=_defl_hover(w),
                        line={"shape": "linear", "color": hex_clr, "width": 1},
                    )
                )
        else:  # faster plotting by using a single trace per signal
            y_vals = _signal_over_intervals(_signal, features["deflections"])
            fig.add_trace(
                pgo.Scatter(
                    x=_signal.index,
                    y=y_vals,
                    mode="lines",
                    hoverinfo="x+y",
                    line={"shape": "linear", "color": "orange", "width": 1},
                )
            )

    if features.get("wavelets") is not None:
        show_deflections = deflection_rendering == "all"
        wave_idx_text = {}

        # Add cycle length traces
        if {"wavelet_cycle_length", "wavelet_iso_ratio"}.intersection(opts.keys()) and "qrs_intervals" in features:
            wv_pairs = pfs.wavelet_dispersal_features(
                features["wavelets"], PPS, features.get("qrs_intervals"), output_as="pairs"
            )
            if wv_pairs["cl"]:
                if opts.get("wavelet_cycle_length"):
                    _dists = [(x[1] - x[0]) / 2 for x in wv_pairs["cl"]]
                    x, y, cl = ppc._as_segments(wv_pairs["cl"], y_offset, _dists)
                    for k, d in zip(wv_pairs["wave_idx"], _dists):
                        wave_idx_text[k] = f"{wave_idx_text.get(k,'')}, CL: {d:.0f}"
                    fig.add_trace(
                        pgo.Scatter(
                            x=x,
                            y=y,
                            mode="lines+text",
                            text=[f"{c:.0f}" if c else None for c in cl],
                            textposition="middle left",
                            textfont={"family": "sans serif", "size": 12, "color": "white"},
                            hoverinfo="skip",
                            marker={"color": "yellow"},
                        )
                    )
                    y_offset -= _step
                if opts.get("wavelet_iso_ratio"):
                    _dists = [(x[1] - x[0]) / (y[1] - y[0]) for x, y in zip(wv_pairs["iso"], wv_pairs["iso+wave"])]
                    x, y, disc = ppc._as_segments(wv_pairs["iso+wave"], y_offset, _dists)
                    for k, d in zip(wv_pairs["wave_idx"], _dists):
                        wave_idx_text[k] = f"{wave_idx_text.get(k,'')}, Disc: {d:.0%}"
                    fig.add_trace(
                        pgo.Scatter(
                            x=x,
                            y=y,
                            mode="lines+text",
                            text=[f"{d:.0%}" if d else None for d in disc],
                            textposition="middle left",
                            textfont={"family": "sans serif", "size": 12, "color": "white"},
                            hoverinfo="skip",
                            marker={"color": "orange"},
                        )
                    )
                    y_offset -= _step

        wavelets = features["wavelets"].to_dict(orient="records")
        if opts.get("wavelet_centroids"):
            fig.add_trace(
                pgo.Scatter(
                    x=[w["centroid_idx"] for w in wavelets],
                    y=[y_offset for _ in wavelets],
                    mode="markers",
                    hoverinfo="skip",
                    marker={"color": "yellow"},
                )
            )

        # Plot the actual wavelets
        y_vals = _signal_over_intervals(_signal, [pc.Interval(w["start_index"], w["end_index"]) for w in wavelets])
        fig.add_trace(
            pgo.Scatter(
                x=_signal.index,
                y=y_vals,
                mode="lines",
                hoverinfo="x+y",
                line={"shape": "linear", "color": ppc.COLOR_HEX["red-salmon"], "width": 1},
            )
        )

        if show_deflections or opts.get("wavelet_centroids"):
            y_offset -= _step

    # dev annotations
    if features.get("annotations") is not None:
        for idx, row in features["annotations"].iterrows():
            hvr_val = row.get("type", idx)
            linewidth = row.get("line_width", 2)
            s = int(row["start_index"])
            e = int(row["end_index"] + 1)
            y = y_offset + (0 if idx % 2 else 0.06)
            clr_hex = row.get("line_color", row.get("color")) or (
                ppc.COLOR_HEX["magenta"] if row.get("semantics", "").startswith("SME") else ppc.COLOR_HEX["cyan"]
            )
            if row.get("semantics", row.get("category")) == "disease":
                hvr_val, linewidth, clr_hex = row["disease_score"], 3, ppc.disease_color(row["disease_score"])

            fig.add_trace(
                pgo.Scatter(
                    x=[s, e],
                    y=[y, y],
                    mode="lines",
                    line={"color": clr_hex, "width": linewidth},
                    text=hvr_val,
                    hoverinfo="text",
                )
            )

    if features.get("sme_annotations") is not None:
        sargs, sme_anns = {}, pu.add_position_indexing(features["sme_annotations"], time_index)
        if opts.get("show_sme_ann_text"):
            txtf = {"family": "sans serif", "size": 10, "color": "white"}
            sargs = {"mode": "lines+text", "textposition": "top right", "textfont": txtf}
        for line_num, ann in sme_anns.iterrows():
            if opts.get("show_sme_ann_text"):
                hvr_val = [f"{ann['channel']} : {ann.get('subcategories') or ''} : L_{line_num}", None]
            else:
                hvr_val = f"{ann['category']} {ann.get('subcategories') or ''}<br>{ann['channel']} , line: {line_num}"
            clr_hex = ann.get("color") or ppc.COLOR_HEX["magenta"]
            fig.add_trace(
                pgo.Scatter(
                    {
                        **{
                            "x": [ann["start_index"], ann["end_index"]],
                            "y": [y_offset, y_offset],
                            "text": hvr_val,
                            "line": {"color": clr_hex, "width": 2},
                            "mode": "lines",
                            "hoverinfo": "text",
                        },
                        **sargs,
                    }
                )
            )
        y_offset -= _step

    return fig


def plot_signal(
    features: dict,
    points_per_second: int,
    channel: str = None,
    fig_opts: dict = {},
    display_opts: dict = {},
) -> pgo.Figure:
    opts = {**{"index_wavelets": True, "activation_centroids": True}, **display_opts}
    fig_opts["title"] = fig_opts.get("title", channel)
    fig = ppc.pretty_layout(None, features["signal"].index, fig_opts)
    _plot_features(features, fig, points_per_second, channel=channel, opts=opts)

    return fig
