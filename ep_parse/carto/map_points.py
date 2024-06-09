import pandas as pd
import os
from datetime import datetime, timedelta
import logging
from collections import namedtuple
from operator import itemgetter
from pprint import pprint

import ep_parse.utils as pu
import ep_parse.catheters as cath
import ep_parse.tag as ptg

log = logging.getLogger(__name__)

MAX_INTERGROUP_DISTANCE = 3.75
Point = namedtuple("Point", "source_file map_id offset coordinates centroid")


FPATTERN_TO_CATHETER = {
    "CS_CONNECTOR_Eleclectrode_Positions_OnAnnotation": None,
    "DEC_CONNECTOR_Eleclectrode_Positions_OnAnnotation": None,
    "NAVISTAR_CONNECTOR_Eleclectrode_Positions_OnAnnotation": cath.Catheter.ABL_MAP,
    "MAGNETIC_20_POLE_A_CONNECTOR_Eleclectrode_Positions_OnAnnotation": cath.Catheter.PENTARAY_2_6,
    "MAGNETIC_20_POLE_B_CONNECTOR_Eleclectrode_Positions_OnAnnotation": None,
    "MCC_DX_CONNECTOR_Eleclectrode_Positions_OnAnnotation": cath.Catheter.OCTARAY,
    "QUAD_A_CONNECTOR_Eleclectrode_Positions_OnAnnotation": None,
}

MAP_ELECTRODE_SEQUENCES = {
    cath.Catheter.ABL_MAP: ("A1", "A2", "A3", "A4"),
    cath.Catheter.PENTARAY_2_6: (
        "A1",
        "A2",
        "A3",
        "A4",
        "B1",
        "B2",
        "B3",
        "B4",
        "C1",
        "C2",
        "C3",
        "C4",
        "D1",
        "D2",
        "D3",
        "D4",
        "E1",
        "E2",
        "E3",
        "E4",
    ),
    cath.Catheter.OCTARAY: (
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    ),
}


def _print_connector_mappings(export_dir: str) -> None:
    pos_file = set()
    for f in os.listdir(export_dir):
        if "Eleclectrode_Positions_OnAnnotation" in f:
            pos_file.add("_".join(f.split("_")[1:-1]))

    print("\nFile to Catheter mappings:")
    pprint({p: FPATTERN_TO_CATHETER.get(p, "!! NEW !!") for p in pos_file})
    print("\n")


def _parse_point_coords(filepath: str, catheter: cath.Catheter) -> list[tuple]:
    df = pd.read_csv(filepath, delim_whitespace=True, skiprows=1)
    match catheter:
        case cath.Catheter.OCTARAY | cath.Catheter.PENTARAY_2_6:
            # the first few coordinates are for the stem ... skip em
            skip = df[df["Electrode#"] == 1].iloc[1].name
            coords = [[r["X"], r["Y"], r["Z"]] for _, r in df.iloc[skip:].iterrows()]
            if len(coords) == len(MAP_ELECTRODE_SEQUENCES[catheter]):
                # cannot tell which spline is absent from file and don't want to mislabel spline,
                # which could lead to a false positive big jump in characteristics across frames
                # since spline X at time T might jump to spline Y values at time T+1
                return coords
        case cath.Catheter.ABL_MAP:
            coords = [[r["X"], r["Y"], r["Z"]] for _, r in df.iterrows()]
            if len(coords) >= 4:
                return coords[0:4]
        case _:
            raise NotImplementedError(f"Parsing coordinates not supported for {catheter.value}")
    log.warning(f"Incorrect # of coordinates in {filepath.split(os.path.sep)[-1]}, file skipped.")


def _map_tag_channels(catheter: cath.Catheter, abl_interval: dict):
    elc2coords = dict(zip(MAP_ELECTRODE_SEQUENCES[catheter], abl_interval["coordinates"]))
    ch2coords = cath.electrodes_to_channels(catheter, elc2coords)
    return ptg.format_MAP_channels(channel_to_coordinates=ch2coords)


def _as_MAP_tag(sync_time: str, catheter: cath.Catheter, abl_interval: dict) -> dict:
    return {
        "start_time": pu.as_time_str(abl_interval["start_time"]),
        "end_time": pu.as_time_str(abl_interval["end_time"]),
        "centroid": abl_interval["centroid"],
        "catheter": catheter.value,
        "channels": _map_tag_channels(catheter, abl_interval),
        "radius": 0.8,
        "notes": "",
        "trace": f"SourceFile=={abl_interval['source_file']}",
        "time_synced_at": sync_time,
    }


def _offset_of(filename: str) -> int:
    return int(filename.split("_")[-1][:-4])  # .*_<time>.txt


def _map_id_of(filename: str, name_pattern: str) -> str:
    return filename[0 : filename.index(name_pattern) - 1]


def _pprint_map_bounds(map_offsets: dict[str, tuple[int, int]], time_0: datetime) -> None:
    vals = []
    for k, v in map_offsets.items():
        vals.append(
            (
                k,
                pu.as_time_str(time_0 + timedelta(milliseconds=v[0])),
                pu.as_time_str(time_0 + timedelta(milliseconds=v[-1])),
                len(v),
            )
        )
    print(
        pd.DataFrame(vals, columns=["map", "start_time", "end_time", "count"]).sort_values(
            by="start_time", ignore_index=True
        )
    )


def raw_map_positions(
    export_dir: str, catheter: cath.Catheter, time_0: datetime, file_pattern: str, file_list: list[str]
) -> list[Point]:
    "Returns a list of points sorted by time"
    map_data, prev_offset, map_offsets = [], None, {}
    for fname in sorted(file_list, key=_offset_of):
        map_id = _map_id_of(fname, file_pattern)
        offset = _offset_of(fname)
        map_offsets[map_id] = map_offsets.get(map_id, []) + [offset]
        if offset == prev_offset:
            continue
        coords = _parse_point_coords(os.path.join(export_dir, fname), catheter)
        if coords:
            map_data.append(
                Point(
                    fname,
                    map_id,
                    time_0 + timedelta(milliseconds=offset),
                    coords,
                    pu.centroid(coords),
                )
            )
            prev_offset = offset
    _pprint_map_bounds(map_offsets, time_0)
    return map_data


def map_file_list(export_dir: str, name_pattern: str) -> list[Point]:
    "Return a list of files containing Map positions"
    return [f for f in os.listdir(export_dir) if f.endswith(".txt") and name_pattern in f]


def _group_by_stability(points: list[Point]) -> list[list[Point]]:
    """Group and filter raw coordinate data to ensure that resulting points correspond to non-trivial duration points

    Args:
        points (list[tuple]): collection of (atria, time_offset_ms, coords) tuples
        time_0 (datetime): time corresponding to offset_ms of 0

    Returns:
        list[dict]: chronologically ordered collection of stable points
    """
    groups, group = [], [points[0]]
    for i in range(1, len(points) - 1):
        group_center = pu.centroid([g.centroid for g in group]) if len(group) > 1 else group[0].centroid
        dist = pu.euc_distance(group_center, points[i + 1].centroid)
        if dist > MAX_INTERGROUP_DISTANCE:
            groups.append(group)
            group = []
        group.append(points[i + 1])
    groups.append(group)
    return groups


def _unique_map_positions(groups: list[list[Point]]) -> list[dict]:
    candidates = []
    for points in groups:
        ms_pad = 250 if len(points) > 1 else 500
        coords = points[int(len(points) / 2)].coordinates
        candidates.append(
            {
                "start_time": points[0].offset - timedelta(milliseconds=ms_pad),
                "end_time": points[-1].offset + timedelta(milliseconds=ms_pad),
                "coordinates": coords,
                "centroid": pu.centroid(coords),
                "num_points": len(points),
                "source_file": points[0].source_file,
                "map_name": points[0].map_id,
            }
        )
    return candidates


def _parse_Map_tags_of_type(
    export_dir: str, map_catheter: cath.Catheter, file_pattern: str, time_0: datetime, opts: dict = {}
) -> list[dict]:
    file_list = map_file_list(export_dir, file_pattern)
    if len(file_list) < 1:
        return []
    log.info(f"Found {len(file_list)} files with {map_catheter} positions")
    raw_points = raw_map_positions(export_dir, map_catheter, time_0, file_pattern, file_list)
    log.debug(f"Filtered {len(raw_points)} unique raw {map_catheter} times")
    if min_ppm := opts.get("min_points_per_MAP", 3):
        if min_ppm < 2:
            intervals = _unique_map_positions([[p] for p in raw_points])
        else:
            points = _group_by_stability(raw_points)
            intervals = _unique_map_positions(points)
            intervals = [p for p in intervals if p["num_points"] >= min_ppm]
    sync_time = datetime.now().isoformat()
    log.info(f"Creating {len(intervals)} {map_catheter} tags")
    return [_as_MAP_tag(sync_time, map_catheter, p) for p in intervals]


def parse_Map_tags(export_dir: str, time_0: datetime, opts: dict = {}) -> list[dict]:
    "Calculate map points based solely on On_annotation files"
    _print_connector_mappings(export_dir)
    tags = []
    for fpattern, catheter in FPATTERN_TO_CATHETER.items():
        if catheter:
            tags += _parse_Map_tags_of_type(export_dir, catheter, fpattern, time_0, opts)
    tags = sorted(tags, key=itemgetter("start_time"))
    for i, t in enumerate(tags):
        t["label"] = f"MAP {i+1}"
    return tags
