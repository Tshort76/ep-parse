import logging
import os
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using slow pure-python")

from fuzzywuzzy import fuzz

import ep_parse.utils as u
import ep_parse.constants as pc
import ep_parse.common as pic
import ep_parse.case_data as d

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG) if u.is_dev_mode() else log.setLevel(logging.ERROR)

# SIG_SESSION_RGX = re.compile(r"([^.]+)\.Session (\d{1,3})", re.IGNORECASE)
SIG_TXT_RGX = re.compile(r"([^.]+)\.Session (\d{1,3}).+(\d)\.TXT", re.IGNORECASE)
SIG_SESSION_RGX = re.compile(r"([^.]+)\.Session (\d{1,3})", re.IGNORECASE)
SESSION_FINDER_RGX = re.compile(r"Session (\d{1,3})", re.IGNORECASE)


def parse_sess_num(filename: str) -> int:
    m = SESSION_FINDER_RGX.search(filename)
    return int(m.group(1)) if m else log.warning(f"No session # found in filename: {filename}")


def _parse_meta_file(filename: str) -> dict:
    assert (
        "page" not in filename.lower()
    ), f"TXT file, {filename}, found with BIN like name.  Did data accidently get exported in TXT format?"

    rgxs = {
        "date": r"date=(\d{2}/\d{2}/\d{4})",
        "time": r"^time=(\d{2}:\d{2}:\d{2})",
        "smp_rate": r"^sample rate=(\d+)\s*hz",
        "res_grp1": r"^signal resolution=(\d+)\s*nv/lsb",
        "res_grp2": r"^channel resolution=(\d+)\s*nv/lsb",
    }

    tmp, channels = {}, set()

    with open(filename, "r") as f:
        for ln in f.readlines():
            if len(rgxs) > 1:
                for lbl, rgx in rgxs.items():  # check for measurement meta fields
                    m = re.search(rgx, ln, re.IGNORECASE)
                    if m:
                        tmp[lbl] = m.group(1)
                        rgxs.pop(lbl)
                        break
            elif SIG_SESSION_RGX.match(ln):
                channels.add(pic.channel_name(ln, raw=True))

    assert (
        len(tmp) > 3
    ), f"Failed to parse all keys from meta file: {filename}, missing {set(rgxs.keys()).difference(tmp.keys())}"

    meta = {"date": tmp["date"], "time": tmp["time"]}
    meta["sample_freq_hz"] = int(tmp["smp_rate"])
    meta["sig_res_nv"] = int(tmp.get("res_grp1") or tmp.get("res_grp2"))
    meta["channels"] = channels

    return meta


def _bins_for_channel(dir: str, channel: str, use_raw_name: bool = False, mapping: dict = None) -> dict:
    return {
        parse_sess_num(f): os.path.join(dir, f)
        for f in os.listdir(dir)
        if f.endswith("BIN") and pic.channel_name(f, raw=use_raw_name, mapping=mapping) == channel
    }


def parse_session_txts(dir: str) -> dict:
    return {
        parse_sess_num(file): _parse_meta_file(os.path.join(dir, file))
        for file in os.listdir(dir)
        if file.endswith("TXT")
    }


def parse_signal_files(
    case_id: str,
    dir: str,
    channels: list[str] = None,
    normalize: bool = True,
    smooth: bool = True,
    compress: bool = True,
):
    "Assumes a single folder contains all signal data for a patient, where the data is stored across multiple sessions files."

    meta = parse_session_txts(dir)
    name_mapping = pic.channel_name_mapping(pic.DataSource.EPMED)
    with d.case_signals_db(case_id, mode="w", compression=compress) as store:
        # This grouping is done according to when the channels will be providing valuable data (i.e. mapping phase vs ablation phase)
        for group_name, group_channels in pc.CHANNEL_GROUPS.items():
            session_files = {}
            # attempt to load all if no channels are specified
            for channel in set(channels or group_channels).intersection(group_channels):
                # group bin files by session number so that we can chunk rows and store them
                for sess, bin_file in _bins_for_channel(dir, channel, mapping=name_mapping).items():
                    session_files[sess] = session_files.get(sess, []) + [(channel, os.path.join(dir, bin_file))]

            session_times = []
            for session_num in sorted(session_files.keys()):
                log.debug(f"Parsing {group_name} files for session {session_num}")
                _time, _freq = [meta[session_num][k] for k in ["time", "sample_freq_hz"]]
                assert _freq % 1000 == 0, f"Expected sampling rate that is divisible by 1000, found {_freq}"
                channel_data = {
                    ch: np.fromfile(bin_path, dtype=np.int32) for ch, bin_path in session_files[session_num]
                }
                data_length = max(map(len, channel_data.values()), default=0)
                if data_length < 100:
                    continue

                for ch, signal in channel_data.items():
                    if data_length > len(signal):
                        log.warn(f"Incomplete signal data for {ch} , session {session_num}.  Zero padding")
                        channel_data[ch] = np.resize(signal, data_length)

                df = pd.DataFrame(
                    channel_data,
                    index=pd.date_range(
                        start=f"{pc.DEFAULT_DATE} {_time}", tz="UTC", periods=data_length, freq=f"{1000/_freq}ms"
                    ),
                )
                if normalize:
                    df = u.normalize_2_mV(df)
                if smooth:
                    df = df.rolling(window=5).mean().iloc[5:]

                # Check for overlapping times since HDFS append does not and the data is known to have overlaps
                for stime, etime in session_times:
                    if not df.empty and df.index[0] <= etime and df.index[0] >= stime:
                        df = df[df.index > etime]
                    if not df.empty and df.index[-1] <= etime and df.index[-1] >= stime:
                        df = df[df.index < stime]

                if not df.empty:
                    session_times += [(df.index[0], df.index[-1])]

                # Ensure that all columns are present (empty if needed), write error if only some get written
                for missing_col in set(group_channels).difference(df.columns):
                    df[missing_col] = np.nan

                store.append(f"signals/{group_name}", pic.standardize_channel_names(df), index=True)


def parse_bookmark_data(data_str: str) -> pd.DataFrame:
    splits = re.split(r"(\d{2}:\d{2}:\d{2})", data_str)
    _time = None
    valz = []
    for i, s in enumerate(splits[1:]):  # drop the header to get to the first time stamp
        if i % 2 == 0:
            _time = datetime.strptime(f"{s}+00:00", f"{pc.TIME_FRMT}%z")
        elif s:
            _event = s.splitlines()[0].strip()
            valz.append([_event, _time])

    return pd.DataFrame(valz, columns=["event", "time"])


def parse_bookmark_file(case_id: str) -> pd.DataFrame:
    filepath = d.case_file_path(case_id, d.FileType.EPMED_LOG)
    if not os.path.exists(filepath):
        log.warning(f"{filepath} was not found, bookmark data will not be parsed")
        return
    with open(filepath, "r", encoding="utf-8") as f:
        return parse_bookmark_data(f.read())


def parse_rfs(event_df: pd.DataFrame) -> pd.DataFrame:
    RF_ON_TEXT1 = "RF turned ON-SESSION"
    RF_OFF_TEXT1 = "RF turned OFF"
    RF_ON_TEXT2 = "RF On - Session"
    RF_OFF_TEXT2 = "RF Off -"

    rf_time = []
    for _, row in event_df.iterrows():
        s = row["event"]
        if (
            fuzz.token_set_ratio(RF_ON_TEXT1, s[0 : len(RF_ON_TEXT1) + 2]) > 80
            or fuzz.token_set_ratio(RF_ON_TEXT2, s[0 : len(RF_ON_TEXT2) + 2]) > 80
        ):
            on_time = row["time"]
            rf_num = re.search(r"ION (\d{1,3})", s, re.IGNORECASE)
            rf_num = int(rf_num.groups()[0]) if rf_num else None
        elif (
            fuzz.token_set_ratio(RF_OFF_TEXT1, s[0 : len(RF_OFF_TEXT1) + 2]) > 80
            or fuzz.token_set_ratio(RF_OFF_TEXT2, s[0 : len(RF_OFF_TEXT2) + 2]) > 80
        ):
            rf_time.append([rf_num, on_time, row["time"]])

    rf_df = pd.DataFrame(rf_time, columns=["RF", "start_time", "end_time"])

    _max_rf = int(rf_df["RF"].max())
    _min_rf = int(rf_df["RF"].min())
    if _min_rf > 1:
        log.warning(f"First recorded RF session is {_min_rf}")

    missing = set(range(_min_rf, _max_rf + 1)) - set(rf_df["RF"].unique().tolist())
    assert not missing, f"Missing RFs: {missing}"

    return rf_df


def bin_coverage(case_dir: str) -> pd.DataFrame:
    session_meta = parse_session_txts(case_dir)
    channels = set([pic.channel_name(f, raw=True) for f in os.listdir(case_dir) if f.endswith("BIN")])
    name_mapping = pic.channel_name_mapping(pic.DataSource.EPMED)

    # file, determine associated session, raw channel name, and number of samples
    data_rows = []
    for ch in channels:
        std_name = pic.std_channel_name(ch, name_mapping)
        ch_group = u.channel_group_of(std_name)
        for session, bin_file in _bins_for_channel(case_dir, ch, use_raw_name=True).items():
            smeta = session_meta[session]
            start_time = u.as_datetime(smeta["time"])
            num_samples = os.path.getsize(bin_file) / 4  # 4 bytes per int32
            dur_s = num_samples / smeta["sample_freq_hz"]  # seconds of coverage in file
            end_time = start_time + timedelta(seconds=dur_s)
            data_rows.append((start_time, end_time, bin_file.split(os.sep)[-1], ch, std_name, ch_group))

    return pd.DataFrame(
        data_rows, columns=["start_time", "end_time", "filename", "raw_name", "std_name", "channel_group"]
    )
