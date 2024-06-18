import json
import re
import plotly.graph_objects as pgo
from datetime import datetime, timedelta
from enum import StrEnum
from collections import OrderedDict

import ep_parse.plots.common as ppc
import ep_parse.constants as pc
import ep_parse.utils as u
import ep_parse.catheters as cath

identity_map = {}
for channel_group in pc.CHANNEL_GROUPS.values():
    identity_map = {**identity_map, **{k: k for k in channel_group}}


def new_version() -> str:
    now = datetime.now().isoformat(timespec="seconds", sep="-")
    return now.replace("-", "").replace(":", "")


class EP_SYSTEM(StrEnum):
    EPMED = "epmed"
    CARDIOLAB = "cardiolab"


class MAPPING_SYSTEM(StrEnum):
    CARTO = "carto"
    ENSITE = "ensite_velocity"


class DataSource(StrEnum):
    ENSITE_VELOCITY = "ensite_velocity"
    CARTO = "carto"
    CARDIOLAB = "cardiolab"
    EPMED = "epmed"
    MD_CATH = "md_catheter_logs"


def channel_name_mapping(source: DataSource) -> dict:
    match source:
        case DataSource.ENSITE_VELOCITY:
            map_file = "ensite_to_standard_channel_mapping.json"
        case DataSource.EPMED:
            map_file = "epmed_to_standard_channel_mapping.json"
        case DataSource.CARDIOLAB:
            map_file = "epmed_to_standard_channel_mapping.json"
        case _:
            raise NotImplementedError(f"No channel mapping exists for {source.name}")

    with open(f"resources/{map_file}", "r") as fp:
        return {**json.load(fp), **identity_map}


def as_RF_tag(
    rf_num: int,
    coords: tuple[float],
    stime: str,
    etime: str,
    radius: float = 2.0,
    sync_time: str = None,
    trace: str = None,
) -> dict:
    synced_at = sync_time or datetime.now().isoformat()
    return {
        "start_time": stime,
        "end_time": etime,
        "coordinates": coords,
        "centroid": coords,
        "label": f"RF {rf_num}",
        "catheter": cath.Catheter.ABL.value,
        "radius": radius,
        "time_synced_at": synced_at,
        "trace": trace,
    }


def standardize_channel_names(signals_df):
    channel_mapping = channel_name_mapping(DataSource.EPMED)
    signals_df.columns = signals_df.columns.to_series().replace(channel_mapping)
    return signals_df[sorted(signals_df.columns)]


def format_channel_name(raw_name: str):
    _name = re.sub(r"\s+", "_", raw_name).replace(",", "-").lower()
    return _name


def std_channel_name(raw_name: str, mapping: dict) -> str:
    return mapping.get(format_channel_name(raw_name))


def channel_name(filename: str, raw: bool = False, mapping: dict = None) -> str:
    raw_name = filename.split(".")[0]
    return raw_name if raw else std_channel_name(raw_name, mapping)


def raw_name_to_std_name(raw_channel_names: list[str]) -> OrderedDict[str, str]:
    "Return raw_name -> standard channel name, ordered according to std UI ordering, with unmapped channels at the bottom"
    name_mapping = channel_name_mapping(DataSource.EPMED)
    std_to_raw = {std_channel_name(ch, name_mapping): ch for ch in raw_channel_names}
    std_to_raw.pop(None, "")  # remove the None -> channel entry
    raw_to_std = OrderedDict()
    # Add unmapped channels
    for raw_name in sorted(raw_channel_names, key=lambda s: s.lower(), reverse=True):
        if raw_name not in std_to_raw.values():
            raw_to_std[raw_name] = None
    # Add mapped channels in standard order (for display)
    for name_ in ppc.DEFAULT_CHANNEL_ORDER:
        if name_ in std_to_raw:
            raw_to_std[std_to_raw[name_]] = name_
    return raw_to_std


def data_coverage_plot(coverage_df, bookmark_df=None) -> pgo.Figure:
    y_ax = [0, 8]
    raw_to_std = raw_name_to_std_name(coverage_df["raw_name"].unique().tolist())
    channel_order = list(raw_to_std.keys())
    offsets = ppc.y_shifts(len(channel_order), y_ax)
    ch_offsets = dict(zip(channel_order, offsets))
    ch_color = {ch: ppc.raw_sig_color(raw_to_std[ch]) for ch in channel_order}
    dur_s = (coverage_df["end_time"].max() - coverage_df["start_time"].min()).total_seconds()

    fig = pgo.Figure()
    for _, row in coverage_df.iterrows():
        y = ch_offsets[row["raw_name"]]
        fig.add_trace(
            pgo.Scatter(
                x=[row["start_time"], row["end_time"]],
                y=[y, y],
                name=row["raw_name"],
                mode="lines+markers",
                hoverinfo="text",
                text=row["filename"],
                line={"shape": "linear", "color": ch_color[row["raw_name"]]},
            )
        )

    if bookmark_df is not None:
        for _, row in bookmark_df.iterrows():
            fig.add_trace(
                pgo.Scatter(
                    x=[row["start_time"], row["end_time"]],
                    y=[0.05, 0.05],
                    name=row["RF"],
                    mode="lines+markers",
                    hoverinfo="text",
                    text=f"RF{row['RF']} ({u.as_time_str(row['start_time'], False)} - {u.as_time_str(row['end_time'], False)})",
                    line={"shape": "linear", "color": "#ff8164"},
                )
            )

    display_offsets = {f"{ch} ({raw_to_std[ch] or ''})": v for ch, v in ch_offsets.items()}

    left_margin = 7 * max(map(len, display_offsets.keys()))
    fig_height = 20 * len(channel_order)
    fig_width = dur_s / 10

    return ppc.pretty_layout(
        fig,
        configs={
            "trace_offsets": display_offsets,
            "left_margin": left_margin,
            "fig_width": fig_width,
            "fig_height": fig_height,
        },
    )


def offset_time_str(start_time: datetime, offset_ms: int) -> str:
    return u.as_time_str(start_time + timedelta(milliseconds=int(offset_ms)))
