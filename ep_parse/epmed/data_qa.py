import functools as ft
import os
import pandas as pd
import logging as log
import datetime as dt
import numpy as np
from IPython.display import display

import ep_parse.epmed.core as epm
import ep_parse.utils as u
import ep_parse.constants as pc
import ep_parse.case_data as d
import ep_parse.common as pic


def _data_per_session_check(channel_to_sessions: dict) -> dict:
    flat_data = [
        [session, {col: len(measurements)}, len(measurements)]
        for col in channel_to_sessions
        for session, measurements in channel_to_sessions[col].items()
    ]
    session2data = u.groupby(flat_data, lambda x: x[0])
    session2counts = {s: len(u.groupby(vals, lambda x: x[-1])) for s, vals in session2data.items()}

    return {
        s: ft.reduce(lambda x, y: {**x, **y}, map(lambda x: x[1], session2data[s]), {})
        for s in filter(lambda x: session2counts[x] > 1, session2counts)
    }


def meta_file_for_all_sessions_check(sig_to_sess_to_x: dict, sess_to_meta: dict):
    sessions_for_bin_type = set(ft.reduce(lambda x, y: x + list(sig_to_sess_to_x[y].keys()), sig_to_sess_to_x, []))
    missing_sessions = sessions_for_bin_type - set(sess_to_meta.keys())
    if missing_sessions:
        log.error(f"Missing session TXT files for sessions: {missing_sessions}")

    return missing_sessions


def _equal_num_measurements_per_channel_check(chan2sess: dict):
    "Do we have an equal number of measuremets for each channel?"
    channel_to_count = {channel: sum([len(data[sess]) for sess in data]) for channel, data in chan2sess.items()}

    if len(set(list(channel_to_count.values()))) > 1:
        z = pd.Series(channel_to_count, name="Minutes spanned").sort_values()
        log.warning(f"Unequal number of measurements for channels.\n{z/(2000 * 60)}\n")


def _equal_num_session_files_check(channel_to_sessions: dict):
    "Do we have an equal number of session files for each channel?"
    length_to_channel = u.groupby(channel_to_sessions, lambda col: len(channel_to_sessions[col]))
    if len(length_to_channel) > 1:
        z = pd.Series()  # this is just for a pretty display
        for x in length_to_channel:
            z = pd.concat([z, pd.Series(length_to_channel[x], name=f"{x} files")], axis=1)

        z = z.drop(0, axis="columns").fillna("-")

        log.warning(f"Unequal number of session files for channels.\n{z}\n")


def _equal_measurements_per_session_check(channel_to_sessions):
    issues = _data_per_session_check(channel_to_sessions)
    if issues:
        log.warning(f"Mismatch in the number of measurements in session files:\n{pd.DataFrame(issues)}\n")


def _time_covered(meta: dict, bin_data: dict):
    vals = []
    for s in meta:
        dur_ms = int(max([len(bin_data[c].get(s, [])) for c in bin_data.keys()]) * (1000 / meta[s]["sample_freq_hz"]))
        vals.append(
            {
                "session": s,
                "start_time": meta[s]["time"],
                "end_time": u.as_time_str(
                    dt.datetime.strptime(meta[s]["time"], pc.TIME_FRMT) + dt.timedelta(milliseconds=dur_ms)
                ),
                "duration_m": round(dur_ms / (1000 * 60), 2),
                "channels": meta[s]["channels"],
            }
        )

    return pd.DataFrame(vals).set_index("session").sort_index()


def _parse_BIN(dir: str, bin_file: str):
    return np.fromfile(os.path.join(dir, bin_file), dtype=np.int32)


def _parse_bins(dir: str, channels: list[str]):
    sig_to_session_to_data = {}
    name_mapping = pic.channel_name_mapping(pic.DataSource.EPMED)

    for file in os.listdir(dir):
        if file.endswith("BIN"):
            cname = epm.channel_name(file, mapping=name_mapping)
            sess_num = epm.parse_sess_num(file)  # we expect multiple sessions per cname
            if not channels or (cname in channels):
                if cname in sig_to_session_to_data:
                    sig_to_session_to_data[cname][sess_num] = _parse_BIN(dir, file)
                else:
                    sig_to_session_to_data[cname] = {sess_num: _parse_BIN(dir, file)}

    return sig_to_session_to_data


def raw_data_coverage(case_dir: str, channels: list[str]):
    session_meta = epm.parse_session_txts(case_dir)
    channel_to_session_to_data = _parse_bins(case_dir, channels)
    return _time_covered(session_meta, channel_to_session_to_data)


# def data_quality_checks(session_meta: dict, channel_to_session_to_data: dict):
def data_quality_checks(case_dir: str, channels: list[str]):
    channel_to_session_to_data = _parse_bins(case_dir, channels)
    _equal_num_measurements_per_channel_check(channel_to_session_to_data)
    _equal_num_session_files_check(channel_to_session_to_data)
    _equal_measurements_per_session_check(channel_to_session_to_data)


def analyze_data_coverage(case_id: str, ablation_df: pd.DataFrame, export_dir: str, channels: list):
    with d.case_signals_db(case_id) as st:
        stime, etime = st.select("signals/CS", stop=1).index[0], st.select("signals/CS", start=-1).index[0]
        print(f"BIN coverage starts at {stime} and ends at {etime}")
    display(ablation_df.iloc[[0, -1]])

    # Analyze intersession gaps
    session_meta = epm.parse_session_txts(export_dir)
    channel_to_session_to_data = _parse_bins(export_dir, channels)

    _df = _time_covered(session_meta, channel_to_session_to_data)

    gaps = []
    for i in range(1, len(_df)):
        s = dt.datetime.strptime(_df.iloc[i - 1]["end_time"], pc.TIME_FRMT)
        e = dt.datetime.strptime(_df.iloc[i]["start_time"], pc.TIME_FRMT)

        gaps.append(max(0, (e - s).total_seconds()))

    print(f"Gaps between Session files (seconds) - sum: {sum(gaps)}  mean: {sum(gaps)/len(gaps)}")
    print(f"Spans of Session files")
    display(_df)

    # measurements per Session BIN file
    tmp = {}
    for sig, sess in channel_to_session_to_data.items():
        tmp[sig] = {k: len(v) for k, v in sess.items()}

    _df = pd.DataFrame(tmp).sort_index()
    print("Measurements per Session BIN")
    display(_df)
