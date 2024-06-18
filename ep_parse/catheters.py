from enum import Enum
import numpy as np
import toolz as tz

import ep_parse.constants as pc


class Catheter(str, Enum):
    HD_GRID = "HD Grid"
    PENTARAY_4 = "PentaRay-4-4-4"
    PENTARAY_2_6 = "PentaRay-2-6-2"
    OCTARAY = "Octaray"
    LASSO = "Lasso"
    HALO = "Halo"
    ABL = "ABL"
    TAG = "Tag"
    ABL_MAP = "ABL_map"

    def __str__(self) -> str:
        return str.__str__(self)


CATHETERS = {
    # dimension[0] is distance from A1 to A4 and A1 to D1  (adjacent corners of a square)
    # dimension[1] is distance from A1 to D4 (opposite corners of a square)
    Catheter.HD_GRID.value: {
        "dimensions": [0.6, 0.85],
        "shape": "rectangle",
        "num_columns": 4,
        "num_rows": 4,
        "type": "mapping",
        "channels": pc.CHANNEL_GROUPS[pc.ChannelGroup.HDGRID],
        "electrode_map": {
            "A1-A2": ("A1", "A2"),
            "A2-A3": ("A2", "A3"),
            "A3-A4": ("A3", "A4"),
            "B1-B2": ("B1", "B2"),
            "B2-B3": ("B2", "B3"),
            "B3-B4": ("B3", "B4"),
            "C1-C2": ("C1", "C2"),
            "C2-C3": ("C2", "C3"),
            "C3-C4": ("C3", "C4"),
            "D1-D2": ("D1", "D2"),
            "D2-D3": ("D2", "D3"),
            "D3-D4": ("D3", "D4"),
            "A1-B1": ("A1", "B1"),
            "A2-B2": ("A2", "B2"),
            "A3-B3": ("A3", "B3"),
            "B1-C1": ("B1", "C1"),
            "B2-C2": ("B2", "C2"),
            "B3-C3": ("B3", "C3"),
            "C1-D1": ("C1", "D1"),
            "C2-D2": ("C2", "D2"),
            "C3-D3": ("C3", "D3"),
        },
        "second_spline_channel": "B1-B2",
        "channel_group": pc.ChannelGroup.HDGRID,
    },
    # dimension[0] is distance from pentaray origin to any spine tip
    # dimension[1] is distance (direct line) between adjacent spine tips (2 * sin(36) * dimension[0])
    Catheter.PENTARAY_4.value: {
        "dimensions": [0.85, 0.85],
        "shape": "pentaray",
        "type": "mapping",
        "channels": [
            f"P{row}_{4*i + col}-{4*i + col+1}" for i, row in enumerate(["A", "B", "C", "D", "E"]) for col in [1, 2, 3]
        ],
        "electrode_map": {
            "PA_1-2": ("A1", "A2"),
            "PA_2-3": ("A2", "A3"),
            "PA_3-4": ("A3", "A4"),
            "PB_5-6": ("B1", "B2"),
            "PB_6-7": ("B2", "B3"),
            "PB_7-8": ("B3", "B4"),
            "PC_9-10": ("C1", "C2"),
            "PC_10-11": ("C2", "C3"),
            "PC_11-12": ("C3", "C4"),
            "PD_13-14": ("D1", "D2"),
            "PD_14-15": ("D2", "D3"),
            "PD_15-16": ("D3", "D4"),
            "PE_17-18": ("E1", "E2"),
            "PE_18-19": ("E2", "E3"),
            "PE_19-20": ("E3", "E4"),
        },
        "second_spline_channel": "PB_5-6",
    },
    Catheter.PENTARAY_2_6.value: {
        "dimensions": [0.85, 0.85],
        "shape": "pentaray",
        "type": "mapping",
        "channels": pc.CHANNEL_GROUPS[pc.ChannelGroup.PENTARAY],
        "electrode_map": {
            "PA_1-2": ("A1", "A2"),
            "PA_3-4": ("A3", "A4"),
            "PB_5-6": ("B1", "B2"),
            "PB_7-8": ("B3", "B4"),
            "PC_9-10": ("C1", "C2"),
            "PC_11-12": ("C3", "C4"),
            "PD_13-14": ("D1", "D2"),
            "PD_15-16": ("D3", "D4"),
            "PE_17-18": ("E1", "E2"),
            "PE_19-20": ("E3", "E4"),
        },
        "second_spline_channel": "PB_5-6",
        "channel_group": pc.ChannelGroup.PENTARAY,
    },
    Catheter.OCTARAY.value: {
        "dimensions": None,  # not going to implement
        "shape": "octaray",
        "type": "mapping",
        "channels": pc.CHANNEL_GROUPS[pc.ChannelGroup.OCTARAY],
        "electrode_map": {
            "OA_1-2": ("A1", "A2"),
            "OA_3-4": ("A3", "A4"),
            "OA_5-6": ("A5", "A6"),
            "OB_1-2": ("B1", "B2"),
            "OB_3-4": ("B3", "B4"),
            "OB_5-6": ("B5", "B6"),
            "OC_1-2": ("C1", "C2"),
            "OC_3-4": ("C3", "C4"),
            "OC_5-6": ("C5", "C6"),
            "OD_1-2": ("D1", "D2"),
            "OD_3-4": ("D3", "D4"),
            "OD_5-6": ("D5", "D6"),
            "OE_1-2": ("E1", "E2"),
            "OE_3-4": ("E3", "E4"),
            "OE_5-6": ("E5", "E6"),
            "OF_1-2": ("F1", "F2"),
            "OF_3-4": ("F3", "F4"),
            "OF_5-6": ("F5", "F6"),
            "OG_1-2": ("G1", "G2"),
            "OG_3-4": ("G3", "G4"),
            "OG_5-6": ("G5", "G6"),
            "OH_1-2": ("H1", "H2"),
            "OH_3-4": ("H3", "H4"),
            "OH_5-6": ("H5", "H6"),
        },
        "second_spline_channel": "OB_1-2",
        "channel_group": pc.ChannelGroup.OCTARAY,
    },
    Catheter.LASSO.value: {
        "dimensions": None,  # not going to implement
        "shape": "lasso",
        "type": "mapping",
        "channels": pc.CHANNEL_GROUPS[pc.ChannelGroup.LASSO],
        "electrode_map": {},
        "second_spline_channel": "LASSO_3-4",
        "channel_group": pc.ChannelGroup.LASSO,
    },
    Catheter.HALO.value: {
        "dimensions": None,  # not going to implement
        "shape": "halo",
        "type": "mapping",
        "channels": pc.CHANNEL_GROUPS[pc.ChannelGroup.HALO],
        "electrode_map": {},
        "second_spline_channel": "HALO_3-4",
        "channel_group": pc.ChannelGroup.HALO,
    },
    Catheter.ABL_MAP.value: {
        "dimensions": [0.4],
        "shape": "line",
        "type": "mapping",
        "channels": pc.CHANNEL_GROUPS[pc.ChannelGroup.ABL],
        "electrode_map": {"ABLd": ("A1", "A2"), "ABLp": ("A3", "A4")},
        "second_spline_channel": "ABLp",
        "channel_group": pc.ChannelGroup.ABL,
    },
    Catheter.ABL.value: {"shape": "point", "type": "ablating"},
    Catheter.TAG.value: {"type": "info"},
}


def chGroup_to_catheter(channel_group: pc.ChannelGroup) -> Catheter:
    for c in Catheter:
        if CATHETERS[c].get("channel_group") == channel_group:
            return c


def chGroup_to_catheter_attrs(channel_group: pc.ChannelGroup) -> dict:
    for k, v in CATHETERS.items():
        if v.get("channel_group") == channel_group:
            return tz.assoc(v, "name", k)


def mapping_channels():
    return list(tz.concat([attrs.get("channels", []) for attrs in CATHETERS.values() if attrs["type"] == "mapping"]))


def format_MAP_channels(
    channels: list[str] = None,
    coordinates: list[list[float]] = None,
    channel_to_coordinates: dict[str, list] = None,
    high_fidelity: bool = True,
) -> dict[str, list]:
    channel_to_coordinates = channel_to_coordinates or dict(zip(channels, coordinates))
    return {ch: {"xyz": coords, "high_fidelity": high_fidelity} for ch, coords in channel_to_coordinates.items()}


def is_mapping_cath(catheter: str) -> bool:
    return False if catheter is None else (catheter in tz.valfilter(lambda x: x["type"] == "mapping", CATHETERS).keys())


# This assumes a certain electrode order ...
def pentaray_electrode_indices(channel: str) -> list[int]:
    return list(map(int, channel[3:].split("-")))


def _safe_midpoint(e2coords: dict, e1: str, e2: str) -> tuple[float]:
    if e1 in e2coords and e2 in e2coords:
        return tuple((np.array(e2coords[e1]) + np.array(e2coords[e2])) / 2)


def electrodes_to_channels(catheter: Catheter, elec_coords: dict[str, tuple[float]]) -> dict[str, tuple[float]]:
    emap, channels = [CATHETERS[catheter][k] for k in ("electrode_map", "channels")]
    return {ch: _safe_midpoint(elec_coords, *emap[ch]) for ch in channels if ch in emap}
