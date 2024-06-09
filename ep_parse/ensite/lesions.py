import pandas as pd
from datetime import timedelta, datetime
import logging
import os
from collections import deque
import toolz as tz

import ep_parse.utils as pu
import ep_parse.common as pic

log = logging.getLogger(__name__)


def _read_sloppy_csv(filepath: str, skiprows: int = 0) -> pd.DataFrame:
    _df = pd.read_csv(filepath, skiprows=skiprows)
    # trailing comma on data rows that isn't in header
    if _df.index[0] != 0:
        # drop the last column, insert the index as a column
        column_names = _df.columns
        _df = _df.drop(columns=[_df.columns[-1]]).reset_index(drop=False)
        _df.columns = column_names
    _df = _df.dropna(axis="index", thresh=int(len(_df.columns) / 2))
    return _df


def _parse_generic_csv(filepath: str, min_expected_columns: int = 10) -> pd.DataFrame:
    i = 0
    with open(filepath, "r") as fp:
        line, i = fp.readline().strip(), i + 1
        while len([x for x in line.split(",") if x]) < min_expected_columns:  # assume at least 10 data columns
            line, i = fp.readline().strip(), i + 1

    with open(filepath, "r") as fp:
        num_lines = len(fp.readlines())

    if num_lines - i > 5:
        return _read_sloppy_csv(filepath, skiprows=i - 1)


def rf_start_times(export_dir: str, rf_log: pd.DataFrame) -> list[str, str]:
    raw_lesions = parse_automark_RFs(os.path.join(export_dir, "AutoMarkSummaryList.csv"))
    if raw_lesions:
        return rf_log.iloc[0]["start_time"], raw_lesions[0]["start_time"]


def parse_lesions_file(filepath: str) -> list[dict]:
    lesion_data = _parse_generic_csv(filepath)
    return [
        {
            "start_time": pu.as_datetime(l["Time"]),
            "coordinates": [l[k] for k in ["xt", "yt", "zt"]],
            "lesion_id": l["Text"],
        }
        for l in lesion_data[lesion_data["Time"].notna()].to_dict(orient="records")
    ]


def parse_automark_RFs(filepath: str) -> list[dict]:
    lesions = _parse_generic_csv(filepath) if os.path.exists(filepath) else None
    if lesions is None:
        return  # no RFs found

    rf_tags = []
    for _, r in lesions.query("`Lesion ID` != '-'").iterrows():
        rf_tags.append(
            {
                "lesion_id": r["Lesion ID"],
                "start_time": pu.as_datetime(r["Start Time"]),
                "end_time": pu.as_datetime(r["End Time"]),
                "coordinates": [r[k] for k in ["X", "Y", "Z"]],
            }
        )
    return rf_tags


def _lesions_with_bookmark_RF(lesions: list[dict], rf_log: list[dict]) -> list[dict]:
    rf_Q = deque(rf_log)
    lesions_Q = deque(lesions)
    curr_rf, lesion, rf_lesions, current_lesions = rf_Q.popleft(), lesions_Q.popleft(), [], []
    prev_rf = curr_rf  # Track this for warning messages
    while curr_rf and lesions_Q:
        if lesion["start_time"] > curr_rf["start_time"] - timedelta(seconds=4):
            if lesion["start_time"] < curr_rf["end_time"]:
                current_lesions.append(lesion)
                lesion = lesions_Q.popleft()
            else:
                rf_lesions.append(tz.assoc(curr_rf, "lesions", current_lesions))
                current_lesions, prev_rf = [], curr_rf
                curr_rf = rf_Q.popleft() if rf_Q else None
        else:
            rf_context_str = f"Previous RF end: {pu.as_time_str(prev_rf['end_time'])}, Next RF start: {pu.as_time_str(curr_rf['start_time'])}"
            log.warn(
                f"Discarding outlier lesion {lesion['lesion_id']}: {pu.as_time_str(lesion['start_time'], False)} - {rf_context_str}"
            )
            lesion = lesions_Q.popleft()

    if lesions_Q:
        log.warn(f"Discarding the {len(lesions_Q)} Ensite lesions that occurred after the last epmed RF")
    return rf_lesions


def _lesion_end_time(lesion: dict, offset: timedelta, rf_end: datetime, nxt: dict) -> datetime:
    tag_end = lesion.get("end_time")
    if tag_end is None:  # lesion.csv has no end_times
        if nxt:  # use start_time of next lesion, if it is associated with same RF
            tag_end = nxt["start_time"] - timedelta(seconds=0.5)
        else:  # use RF end time as lesion end time
            tag_end = rf_end
    tag_end = min(tag_end + offset, rf_end)
    return pu.as_time_str(tag_end)


def _as_rf_tags(rfs_with_lesions: list[dict]) -> list[dict]:
    tags, sync_time = [], datetime.now().isoformat()
    for rf_data in rfs_with_lesions:
        if rf_lesions := rf_data["lesions"]:
            rf_offset = rf_data["start_time"] - rf_lesions[0]["start_time"]
            for i, lesion in enumerate(rf_lesions):
                next_lesion = rf_lesions[i + 1] if i + 1 < len(rf_lesions) else None
                tags.append(
                    pic.as_RF_tag(
                        rf_num=len(tags) + 1,
                        coords=lesion["coordinates"],
                        stime=pu.as_time_str(lesion["start_time"] + rf_offset),
                        etime=_lesion_end_time(lesion, rf_offset, rf_data["end_time"], next_lesion),
                        sync_time=sync_time,
                        trace=("Automarksummary_LesionID==" if "end_time" in lesion else "Lesions_Text==")
                        + lesion["lesion_id"],
                    )
                )
    return tags


def parse_RF_tags(export_dir: str, rf_log: pd.DataFrame, epmed_offset=timedelta) -> list[dict]:
    raw_lesions = parse_automark_RFs(os.path.join(export_dir, "AutoMarkSummaryList.csv"))
    if raw_lesions:
        log.debug(f"Parsed {len(raw_lesions)} lesions from AutoMarkSummaryList.csv")
    else:
        raw_lesions = parse_lesions_file(os.path.join(export_dir, "Lesions.csv"))
        log.debug(f"Parsed {len(raw_lesions)} lesions from Lesions.csv")

    for l in raw_lesions:
        l["start_time"] = l["start_time"] + epmed_offset
        if "end_time" in l:
            l["end_time"] = l["end_time"] + epmed_offset
    rfs_with_lesions = _lesions_with_bookmark_RF(raw_lesions, rf_log.to_dict(orient="records"))
    return _as_rf_tags(rfs_with_lesions)
