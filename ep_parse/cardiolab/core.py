import logging
import re
import pandas as pd
import numpy as np
from datetime import datetime
import os

import ep_parse.utils as u
import ep_parse.constants as pc
import ep_parse.common as pic
import ep_parse.case_data as d

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _as_military_time(dt_str: str) -> str:
    spz = dt_str.strip().split(" ")
    if len(spz) == 3:
        spz = spz[1:]
    components = list(map(int, spz[0].split(":")))
    components[0] += 0 if (spz[1] == "AM" or components[0] == 12) else 12
    return ":".join(map(lambda x: str(x).zfill(2), components))


def _parse_meta_file(filename: str) -> dict:
    channel_start = r"^Channel Number"
    channel_rgx = r"\d+\s+(\w.*)"

    rgxs = {
        "num_channels": r"^Number of Channel[s]?\s*=\s*(\d+)",
        "points_per_channel": r"^Points for Each Channel\s*=\s*(\d+)",
        "sample_rate": r"^Data Sampling Rate\s*=\s*(\d+)",
        "start_datetime": r"^Start Time\s*=\s*(.+)",
        "end_datetime": r"^Stop Time\s*=\s*(.+)",
        "units": r"^Units\s*:\s*.*\s(mV).*",
    }

    _meta = {}
    with open(filename, "r") as f:
        for ln in f.readlines():
            for lbl, rgx in rgxs.items():
                m = re.search(rgx, ln, re.IGNORECASE)
                if m:
                    _meta[lbl] = m.group(1)
                    rgxs.pop(lbl)
                    break
            if _meta.get("channels") is None:
                if re.search(channel_start, ln, re.IGNORECASE):
                    _meta["channels"] = []
            elif len(_meta["channels"]) == int(_meta["num_channels"]):
                break
            else:
                channel = pic.format_channel_name(re.search(channel_rgx, ln).group(1))
                _meta["channels"] += [channel]

    assert "channels" in _meta, "Failed to parse the channels from the information file"

    for k in ["points_per_channel", "num_channels", "sample_rate"]:
        _meta[k] = int(_meta[k])
    _meta["start_time"] = _as_military_time(_meta["start_datetime"])
    _meta["end_time"] = _as_military_time(_meta["end_datetime"])

    return _meta


def _files_of_type(directory: str, file_extension: str) -> str:
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(file_extension)]


def parse_bin_file(meta: dict, bin_file: str, channels: list[str] = None, store: pd.HDFStore = None):
    log.debug(f"Parsing cardio data from {bin_file}")
    signal = np.fromfile(bin_file, dtype=np.float64)

    expected_len = meta["num_channels"] * meta["points_per_channel"]
    assert len(signal) == expected_len, f"Expected to read {expected_len} points, but only found {len(signal)}"
    stime, etime = [f"{pc.DEFAULT_DATE} {meta[x]}" for x in ["start_time", "end_time"]]
    PPS = meta["num_channels"] * meta["sample_rate"]
    # allow difference of 5 seconds in data points (file times are rounded to nearest second, so 2 should actually be enough)
    margin_of_error = PPS * 5
    data_dur = (datetime.fromisoformat(etime) - datetime.fromisoformat(stime)).total_seconds()
    if abs((data_dur * PPS) - len(signal)) > margin_of_error:
        log.warning(
            f"Case start_time, end_time, and sampling frequency are incompatible with the number of data points in the bin file!!!  Difference amounts to {abs((data_dur * PPS) - len(signal))/PPS:.1f} seconds"
        )

    df = pd.DataFrame(
        np.reshape(signal, (meta["points_per_channel"], meta["num_channels"])),
        index=pd.date_range(
            start=stime,
            tz="UTC",
            periods=meta["points_per_channel"],
            end=etime,
        ),
        columns=meta["channels"],
    )
    df = pic.standardize_channel_names(df)

    for group, chans in pc.CHANNEL_GROUPS.items():
        cols = set(chans).intersection(channels or chans)
        if len(df.columns.intersection(cols)) == 0:
            continue
        # Ensure that all columns are present to prevent write error (some files contain a subset of signals in other files)
        for missing_col in cols.difference(df.columns):
            df[missing_col] = np.nan
        store.append(f"signals/{group}", df[sorted(cols)], index=True)


def _inf_bin_pairs(export_dir: str) -> list[tuple[str, str]]:
    pairs = []
    for mfile in _files_of_type(export_dir, ".inf"):
        binfile = mfile[:-4] + ".bin"
        if os.path.exists(binfile):
            pairs.append((os.path.join(export_dir, mfile), os.path.join(export_dir, binfile)))
        else:
            log.warn(f"No bin file found for {export_dir}/{mfile}.  Add the bin or remove the inf file!")

    for f in os.listdir(export_dir):
        d = os.path.join(export_dir, f)
        if "." not in f and os.path.isdir(d):
            pairs += _inf_bin_pairs(d)

    return pairs


def meta_n_bin(case_dir) -> list[tuple[dict, str]]:
    "Returns meta data with associated BIN path, sorted by start time of the bin data"
    rval = []
    for inf, bn in _inf_bin_pairs(case_dir):
        meta = _parse_meta_file(inf)
        rval.append((meta, bn))
    return list(sorted(rval, key=lambda k: k[0]["start_time"]))


def parse_signals(case_id: str, case_dir: str, channels: list[str] = None, compress: bool = True):
    with d.case_signals_db(case_id, mode="w", compression=compress) as store:
        for meta, bn in meta_n_bin(case_dir):
            parse_bin_file(meta, bn, channels, store)


def parse_log(data_str: str) -> pd.DataFrame:
    splits = re.split(r"(\d{1,2}:\d{2}:\d{2} [AP]M)", data_str)

    _time = None
    valz = []
    for i, s in enumerate(splits[1:]):  # drop the header to get to the first time stamp
        if i % 2 == 0:
            mtime = _as_military_time(s)
            _time = datetime.strptime(f"{mtime}+00:00", f"{pc.TIME_FRMT}%z")
        elif s:
            _event = s.strip().replace("\n", "")
            valz.append([_event, _time])

    return pd.DataFrame(valz, columns=["event", "time"]).iloc[:-1]


def _raw_log_path(case_id: str) -> str:
    _path = d.case_file_path(case_id, d.FileType.CARDIOLAB_LOG)
    if os.path.exists(_path):
        return _path


def parse_log_file(case_id: str = None, filepath: str = None) -> pd.DataFrame:
    if not filepath:
        filepath = _raw_log_path(case_id)
    with open(filepath, "r") as f:
        return parse_log(f.read())


def parse_rfs(event_df: pd.DataFrame) -> pd.DataFrame:
    RF_ON_TEXT = r"Ablation (\d+) Start"
    RF_OFF_TEXT = r"Ablation \d+ End"

    rf_times = []
    for _, row in event_df.iterrows():
        s = row["event"]
        if _f := re.search(RF_ON_TEXT, s, re.IGNORECASE):
            on_time = row["time"]
            rf_num = int(_f.group(1))
        elif re.search(RF_OFF_TEXT, s, re.IGNORECASE):
            rf_times.append([rf_num, on_time, row["time"]])

    return pd.DataFrame(rf_times, columns=["RF", "start_time", "end_time"])


def _dir_coverage(subdir: str, name_mapping: dict) -> list[tuple]:
    row_data = []
    for f in os.listdir(subdir):
        if f.endswith(".inf"):
            meta_file = os.path.join(subdir, f)
            meta = _parse_meta_file(meta_file)
            stime = u.as_datetime(meta["start_time"])
            etime = u.as_datetime(meta["end_time"])
            for ch in meta["channels"]:
                pname = pic.std_channel_name(ch, name_mapping)
                row_data.append((stime, etime, meta_file.replace(".inf", ".BIN"), ch, pname, u.channel_group_of(pname)))
    return row_data


def bin_coverage(case_dir: str) -> pd.DataFrame:
    name_mapping, row_data = pic.channel_name_mapping(pic.DataSource.CARDIOLAB), []
    if top_lvl := _dir_coverage(case_dir, name_mapping):
        row_data = top_lvl
    else:
        for d in os.listdir(case_dir):
            subdir = os.path.join(case_dir, d)
            if "." not in d and os.path.isdir(subdir):
                row_data += _dir_coverage(subdir, name_mapping)
    return pd.DataFrame(
        row_data, columns=["start_time", "end_time", "filename", "raw_name", "std_name", "channel_group"]
    )
