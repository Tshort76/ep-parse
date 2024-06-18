import os
import pandas as pd
import re
from datetime import datetime, timezone, timedelta
from pprint import pprint
import logging


import ep_parse.utils as u
import ep_parse.common as pic
import ep_parse.catheters as cath

DXL_DATAFILE_RGX = re.compile(r"DxL_\d+.csv")
MAP_ID_RGX = re.compile(r"(.+)_\d{4}_\d{2}_\d{2}.*")
MAP_DISTANCE_THRESHOLD = 2

log = logging.getLogger(__name__)


def parse_DxL_file(filepath: str) -> pd.DataFrame:
    col_data = {}
    with open(filepath, "r") as fp:
        line = fp.readline().strip()
        while not line.lower().startswith("begin data"):
            line = fp.readline().strip()
        line = fp.readline().strip()  # skip over the line

        while ":" in line[:128]:
            idx = line.index(":")
            col_name = line[0:idx]
            data = line[idx + 1 :].split(",")
            if len(data) > 10:
                col_data[col_name] = data[1:]  # comma precedes first data point ... obviously ;)
            line = fp.readline().strip()
    return pd.DataFrame(col_data)


def _read_dxl_csvs(export_dir: str) -> pd.DataFrame:
    dxl_data = pd.DataFrame()
    for file in os.listdir(export_dir):
        if DXL_DATAFILE_RGX.match(file):
            df = parse_DxL_file(os.path.join(export_dir, file))
            df["filename"] = file
            dxl_data = pd.concat([dxl_data, df], ignore_index=True)
    return dxl_data


# Ensure that the points parsed map to a known catheter
def with_standard_channel_names(df: pd.DataFrame, print_mapping: bool = True):
    channel_mapping = pic.channel_name_mapping(pic.DataSource.ENSITE_VELOCITY)
    if print_mapping:
        print("Channel Mapping:")
        pprint({ch: channel_mapping.get(ch) for ch in df.columns})
    df.columns = df.columns.to_series().replace(channel_mapping)
    return df[sorted(df.columns)]


def parse_grid_points(export_dir: str) -> pd.DataFrame:
    "Creates coordinates time series data for each HDgrid channel"
    dxl_data = _read_dxl_csvs(export_dir)
    new_data, new_index = [], []
    for epoch_time, df in dxl_data.groupby("end time"):
        _time = datetime.fromtimestamp(float(epoch_time), timezone.utc)
        new_row = {}
        for row in df.to_dict(orient="records"):
            channel = row["rov trace"]
            xyz = [row[k] for k in ["roving x", "roving y", "roving z"]]
            new_row[pic.format_channel_name(channel)] = xyz
        if new_row:
            new_data.append(new_row)
            new_index.append(_time)

    return pd.DataFrame(new_data, index=new_index)


def _distance(p1, p2):
    return u.euc_distance([float(x) for x in p1], [float(x) for x in p2])


def group_into_tags(grid_points: pd.DataFrame, catheter: cath.Catheter, map_id: str = None) -> list[dict]:
    cath_props = cath.CATHETERS[catheter]
    if cath_props is None:
        log.warn(f"Ensite Tag creation is not supported for {catheter.value} points.")

    # Use the channel with the most non-nil instances to identify distinct locations
    sentinel = grid_points.notna().sum().idxmax()
    log.info(f"Channel coordinate counts: {grid_points.notna().sum().to_dict()}")
    coordinates = grid_points[sentinel].dropna().sort_index()

    results, prev_coords, curr = [], coordinates.iloc[0], [coordinates.index[0], coordinates.index[0]]
    for t, coords in coordinates.iloc[1:].items():
        if _distance(coords, prev_coords) > MAP_DISTANCE_THRESHOLD:
            results.append(curr)
            curr = [t, t]
        else:
            curr[1] = t
        prev_coords = coords
    results.append(curr)  # add last group

    map_tags = []
    for stime, etime in results:
        start = etime - timedelta(seconds=1) if stime == etime else stime
        d = {
            "start_time": u.as_time_str(start),
            "end_time": u.as_time_str(etime),
            "label": "dummy",
            "catheter": catheter.value,
            "radius": 6,
            "map_id": map_id,
        }

        tag_df = grid_points.loc[stime:etime]
        tag_df = tag_df.dropna(axis="columns", thresh=1)
        channels = tag_df.columns.intersection(cath_props["channels"]).tolist()

        if len(channels) < 2:
            continue

        if len(tag_df) == 1:
            coords = [tag_df[ch].iloc[0] for ch in channels]
        else:
            coords = [tag_df[ch].dropna().iloc[0] for ch in channels]  # first valid coord per channel
        coords = [[float(x) for x in co] for co in coords]
        ch2coords = cath.format_MAP_channels(channels, coords)
        for ch in cath_props["channels"]:
            if ch not in ch2coords:
                ch2coords[ch] = {"high_fidelity": True}
        d["channels"] = ch2coords
        d["centroid"] = u.centroid(coords)
        map_tags.append(d)

    return map_tags


def parse_map_id(dir_name: str, case_id: str) -> str:
    if dir_name.upper().startswith(case_id.upper() + "-"):
        dir_name = dir_name[len(case_id) + 1 :]
    _id = MAP_ID_RGX.findall(dir_name)
    assert (
        _id and len(_id) == 1
    ), f"Invalid Map folder name, expected CASEID-NAME_YYYY_MM_DD with optional time and case id, but found {dir_name}"
    return _id[0]
