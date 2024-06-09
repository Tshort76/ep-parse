import json
import os
import re
import uuid
from collections.abc import Sequence
from enum import Enum
from typing import Union, Callable

import matplotlib
import pandas as pd
import plotly.graph_objects as pgo
import toolz as tz
from PIL import Image

import ep_parse.constants as pc
import ep_parse.utils as pu

PLOTLY_TEMPLATE = "plotly_dark"
# COLOR_SCALE = px.colors.diverging.Spectral[::2] + px.colors.diverging.Spectral[1::2]
COLOR_SCALE = [
    "rgb(158,1,66)",
    "rgb(244,109,67)",
    "rgb(102,194,165)",
    "rgb(94,79,162)",
    "rgb(213,62,79)",
    "rgb(253,174,97)",
    "rgb(255,255,191)",
    "rgb(171,221,164)",
    "rgb(50,136,189)",
    "rgb(254,224,139)",
    "rgb(230,245,152)",
]
V2_PLUS = re.compile(".+_v[0-9]")


DEFAULT_CHANNEL_ORDER = (
    pc.CHANNEL_GROUPS[pc.ChannelGroup.ECG]
    + pc.CHANNEL_GROUPS[pc.ChannelGroup.HRA]
    + pc.CHANNEL_GROUPS[pc.ChannelGroup.CS]
    + pc.CHANNEL_GROUPS[pc.ChannelGroup.ABL]
    + pc.CHANNEL_GROUPS[pc.ChannelGroup.PENTARAY]
    + pc.CHANNEL_GROUPS[pc.ChannelGroup.OCTARAY]
    + pc.CHANNEL_GROUPS[pc.ChannelGroup.HDGRID]
    + pc.CHANNEL_GROUPS[pc.ChannelGroup.LASSO]
    + pc.CHANNEL_GROUPS[pc.ChannelGroup.HALO]
)[::-1]

COLOR_HEX = {
    "white": "#ffffff",
    "gray-light": "#e8e8e8",
    "gray-medium": "#b5b5b5",
    "gray-dark": "#8a8a8a",
    "black": "#000000",
    "yellow-bright": "#ebe534",
    "yellow-gold": "#ebc334",
    "green-lime": "#74f73b",
    "green-bright": "#37ff00",
    "green-dark": "#118f03",
    "cyan": "#34ebd5",
    "magenta": "#d534eb",
    "orange": "#f24f13",
    "red-salmon": "#f76363",
    "red": "#ff0000",
    "red-dark": "#940404",
}


def at_plot_index(channels_in_plot: list[str], idx: int = None) -> str:
    "return channel that is at plot index (0 is bottom trace, so -2 is second from top)"
    ch_order = [k for k in DEFAULT_CHANNEL_ORDER if k in channels_in_plot]
    if idx is None:
        return ch_order
    return ch_order[idx]


def channel_color(channel_group: pc.ChannelGroup) -> str:
    match channel_group:
        case pc.ChannelGroup.CS | pc.ChannelGroup.HRA:
            c = COLOR_HEX["green-bright"]
        case (
            pc.ChannelGroup.PENTARAY
            | pc.ChannelGroup.HDGRID
            | pc.ChannelGroup.OCTARAY
            | pc.ChannelGroup.LASSO
            | pc.ChannelGroup.HALO
        ):
            c = COLOR_HEX["cyan"]
        case pc.ChannelGroup.ABL:
            c = COLOR_HEX["yellow-bright"]
        case pc.ChannelGroup.ECG:
            c = COLOR_HEX["gray-dark"]
        case _:
            c = COLOR_HEX["gray-light"]
    return c


def raw_sig_color(sig_name: str):
    return channel_color(pu.channel_group_of(sig_name))


def y_shifts(num_channels: int, y_axis: tuple):
    if num_channels == 1:
        shifts = [0]
    else:
        _step = (y_axis[1] - y_axis[0]) / (num_channels + 1)
        shifts = [(_step * i) + y_axis[0] for i in range(1, num_channels + 1)]

    return shifts


def with_yaxis_traces(fig: pgo.Figure, trace_offsets: dict[str, float], y_range: tuple[float] = None) -> None:
    y_labels = [
        {
            "xref": "paper",
            "x": 0,
            "y": y,
            "xanchor": "right",
            "yanchor": "middle",
            "text": lbl,
            "showarrow": False,
            "font": {"family": "Arial", "size": 12},
        }
        for lbl, y in trace_offsets.items()
    ]

    fig.update_layout(
        annotations=y_labels,
        yaxis=dict(
            title=None,
            gridwidth=1,
            range=y_range,
            showgrid=True,
            showline=False,
            showticklabels=False,
            tickmode="array",
            tickvals=list(trace_offsets.values()),
            zeroline=False,
        ),
    )


def add_annotation(fig: Union[dict, pgo.Figure], x, ann_type: str = "vline", **kwargs) -> pgo.Figure:
    pfig = pgo.Figure(fig) if isinstance(fig, dict) else fig
    match ann_type:
        case "vline":
            _name = kwargs.get("name", "vertical_line")
            pfig.add_vline(
                x=x,
                line_dash="dash",
                line_color=kwargs.get("color", "orange"),
                name=_name,
                annotation={
                    "text": kwargs.get("label", ""),
                    "name": _name,
                    "xanchor": "right",
                },
            )
        case "arrow":
            pfig.add_annotation(
                x=x,
                y=kwargs["y"],
                arrowcolor="#1fa8f2",
                arrowhead=2,
                arrowwidth=2,
                startarrowhead=6,
                arrowside=kwargs.get("arrowside", "end"),
                opacity=kwargs.get("opacity", 1),
            )
        case "dot":
            pfig.add_annotation(
                x=x,
                y=kwargs["y"],
                showarrow=False,
                text=kwargs.get("label", ""),
                font_color="#ff0000",
                opacity=1,
            )

    return pfig


def _as_segments(boundaries: list[tuple[int, int]], y: float = 0, text: list[str] = []):
    # Interpose None so that segments are plotted (instead of a continuous line)
    x_vals = list(tz.concat(tz.interpose([None], boundaries)))
    if isinstance(y, Sequence):
        y_vals = list(tz.concat(tz.interpose([None], [[a, a] for a in y])))
    else:
        y_vals = list(tz.concat(tz.interpose([None], [[y, y] for _ in boundaries])))
    hhints = list(tz.concat(tz.interpose([None, None], [[txt] for txt in text]))) if text else None
    return x_vals, y_vals, hhints


def pretty_layout(fig: pgo.Figure = None, x_index: pd.Index = None, configs: dict = {}) -> pgo.Figure:
    """Update figure layout (in place) so that it is standardized and aesthetically pleasing

    Args:
        fig (pgo.Figure, optional): The fig to update (create new fig if None)
        x_index (pd.Index, optional): Time index for the data, if a time series. Defaults to None.
        configs (dict, optional): layout configurations, keys: trace_offsets, fig_height, fig_width, width_as_ms, x_axis, y_axis, left_margin, title. Defaults to {}.

    Returns:
        pgo.Figure: The formatted figure
    """
    fig = fig or (pgo.FigureWidget() if configs.get("ui_environment") == "jupyter" else pgo.Figure())
    fig_width, ms_adj = configs.get("fig_width"), {}
    if isinstance(x_index, pd.DatetimeIndex):
        PPS = pu.points_per_second(x_index)
        if configs.get("width_as_ms"):
            fig_width = ms_to_fig_width(configs.get("width_as_ms"), signal_len_ms=pu.points_as_ms(len(x_index), PPS))
        # create 4 ticks per second
        x_ticks = list(range(0, len(x_index), int(PPS / 4)))
        tick_labels = ["" if j % 4 else pu.as_time_str(x_index[i]) for j, i in enumerate(x_ticks)]
        ms_adj = {
            "tickmode": "array",
            "tickvals": x_ticks,
            "ticktext": tick_labels,
        }

    _autosize = fig_width is None and configs.get("width_as_ms") is None

    fig.update_layout(
        autosize=_autosize,
        width=None if _autosize else fig_width,
        height=None if _autosize else configs.get("fig_height"),
        clickmode="event" if configs.get("register_clicks") else None,
        hoverdistance=1,
        showlegend=configs.get("show_legend", False),
        template=PLOTLY_TEMPLATE,
        xaxis={
            "showline": True,
            "showgrid": True,
            "showticklabels": True,
            "range": configs.get("x_axis") if x_index is None else [0, len(x_index)],
            "ticks": "outside",
            **ms_adj,
        },
        yaxis={
            "showgrid": True,
            "gridwidth": 1,
            "zeroline": False,
            "showline": False,
        },
        margin={"l": configs.get("left_margin", 70), "r": 10, "b": 5, "t": 5},
    )

    if configs.get("trace_offsets"):
        with_yaxis_traces(fig, configs.get("trace_offsets"), y_range=configs.get("y_axis"))

    if _title := configs.get("title"):
        fig.update_layout(
            title={
                "text": _title,
                "y": 0.98,
                "x": 0,
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 12, "color": "magenta"},
            },
        )

    if configs.get("y_axis"):
        fig.update_yaxes(range=configs["y_axis"])

    return fig


def ms_to_fig_width(width_as_ms: int, signal_len_ms: int = None, PPS: int = None, signal_len_pts: int = None) -> int:
    l = pu.points_as_ms(signal_len_pts, PPS) if signal_len_pts else signal_len_ms
    return max(400, (l * 2500) / width_as_ms)
