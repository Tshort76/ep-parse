import logging
from importlib import reload
import os
import re
from collections import namedtuple
from datetime import datetime
from functools import singledispatch
from logging.config import dictConfig
from operator import itemgetter
from typing import Union

import numpy as np
import pandas as pd
import yaml

import ep_parse.constants as pc


def is_dev_mode() -> str:
    return bool(os.environ.get("DEV_MODE"))


def configure_logging() -> None:
    config_file = "logs/dev_logging_configs.yaml" if is_dev_mode() else "logs/logging_configs.yaml"
    print(f"Configuring logging using {config_file}")
    reload(logging)
    with open(config_file, "r") as fp:
        dictConfig(yaml.safe_load(fp))


log = logging.getLogger(__name__)

_date_start = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")
_time_only = re.compile(r"(\d{1,2}):\d{1,2}:\d{1,2}[.]?\d*")
_ms_shorthand = re.compile(r".+\.\d{0,2}$")
_tz_aware = re.compile(r"\d{1,2}:\d{1,2}:\d{1,2}[.]?\d*[+-]\d{2}")

SPLIT_FIT = namedtuple("SPLIT_FIT", "rss slope length")


def channel_group_of(channel: str) -> str:
    for g, channels in pc.CHANNEL_GROUPS.items():
        if channel in channels:
            return g


def poly_fit(signal: np.array, degree: int = 1) -> SPLIT_FIT:
    try:
        _fit = np.polyfit(x=range(0, len(signal)), y=signal, deg=degree, full=True)
    except np.linalg.LinAlgError:
        return
    m, *_ = _fit[0]
    rss = _fit[1][0] if len(_fit[1]) else np.nan
    return SPLIT_FIT(rss, m, len(signal))


def std_col_2_hdf(orig: str) -> str:
    return orig.replace("-", "__")


def hdf_2_std_col(orig: str) -> str:
    return orig.replace("__", "-")


def coerce_wavelet_configs(defaults: dict, PPS: int, user_configs: dict = {}) -> dict:
    configs = {**defaults, **user_configs}
    for k in ["max_iso_gap", "max_non_prom_gap"]:
        configs[k] = ms_as_points(configs[k], PPS)  # user inputs in milliseconds, convert to samples
    return configs


def points_per_second(index: pd.DatetimeIndex) -> int:
    if index is None or index.empty:
        log.debug("Attempted to calculate points per second for an empty frame, returning 1000")
        return 1000
    return round(1 / (index[1] - index[0]).total_seconds())


def points_as_ms(num_points: int, points_per_second: int) -> int:
    return int((1000 * num_points) / points_per_second)


def ms_as_points(ms: int, points_per_second: int) -> int:
    return int(ms * (points_per_second / 1000))


def as_time_str(date_like: Union[str, datetime], ms_precision: bool = True) -> str:
    if isinstance(date_like, datetime):
        frmt = pc.TIME_WITH_MS_FRMT if ms_precision else pc.TIME_FRMT
        return date_like.time().strftime(frmt)[0:12]
    if _date_start.match(date_like):
        date_like = date_like[11:]
    if len(date_like) > 8:  # datetime str
        if "." in date_like[8:10]:
            return date_like[0:10].ljust(12, "0")
        date_like = date_like[0:8]
    return date_like


def as_datetime(date_like: Union[str, datetime]) -> datetime:
    time_str = date_like.isoformat() if isinstance(date_like, datetime) else date_like

    if _date_start.match(time_str):
        time_str = time_str[11:]
    if hour := _time_only.findall(time_str):
        if len(hour[0]) < 2:
            time_str = "0" + time_str
        time_str = f"{pc.DEFAULT_DATE}T{time_str}"
    if _ms_shorthand.match(time_str):
        time_str = time_str.ljust(23, "0")
    if not _tz_aware.search(time_str):
        time_str += "+00:00"

    try:
        return datetime.fromisoformat(time_str)
    except ValueError as e:
        log.warn(f"Skipping invalid date string: {time_str}")


def seconds_between(date1: Union[str, datetime], date2: Union[str, datetime]) -> float:
    "parse date1 and date2 as dates, return (d2 - d1).total_seconds()"
    d1, d2 = [as_datetime(x) for x in (date1, date2)]
    return (d2 - d1).total_seconds()


@singledispatch
def min_max(l, **kwargs):
    raise TypeError(f"Cannot run min/max normalization on {l}")


@singledispatch
def min_max(df: pd.DataFrame, bounds: tuple[float, float] = None) -> pd.DataFrame:
    "Calculate the min max norm for a dataframe so that the range is [0,1]"
    if bounds:
        return (df - bounds[0]) / (bounds[1] - bounds[0])
    df_min = df.min()
    return (df - df_min) / (df.max() - df_min)


@singledispatch
def min_max(signal: np.ndarray, bounds: tuple[float, float] = None) -> np.ndarray:
    "Calculate the min max norm for a nparray so that the range is [0,1]"
    if bounds:
        mn, mx = bounds[0], bounds[1]
    else:
        mn, mx = np.min(signal), np.max(signal)
    if mx > mn:
        return (signal - mn) / (mx - mn)
    return np.zeros_like(signal)


def setup_env():
    return


def groupby(seq: list, key_fn) -> dict:
    r = {}
    for itm in seq:
        x = key_fn(itm)
        if x in r:
            r[x].append(itm)
        else:
            r[x] = [itm]
    return r


def partition_by(items: list, split_fn) -> list:
    """Partition a list of items using a splitting function that compares the previous and current item
    and returns a truthy value (returns true at splitting points).

    Args:
        items (list): The sequence of items to be partitioned
        split_fn (function): Predicate function with signature fn(current_item, previous_item)

    Returns:
        list: list of partitions
    """
    if not items:
        return []

    partitions = []
    _partition = [items[0]]
    prev = items[0]
    for itm in items[1:]:
        if split_fn(itm, prev):
            partitions.append(_partition)
            _partition = []
        _partition.append(itm)
        prev = itm

    if _partition:
        partitions.append(_partition)

    return partitions


def add_position_indexing(df: pd.DataFrame, time_index: Union[pd.DatetimeIndex, pd.Index]) -> pd.DataFrame:
    """Add start_index and end_index columns to the dataframe that is currently partitioned by time

    Args:
        df (pd.DataFrame): dataframe with start_time and end_time columns
        time_index (Union[pd.DatetimeIndex, pd.Index]): index of dataframe as datetimes or iso strings for times

    Returns:
        pd.DataFrame: copy of df with new start_index and end_index columns
    """
    if df.empty:
        return df

    str_times = [as_time_str(s) for s in time_index] if isinstance(time_index, pd.DatetimeIndex) else time_index
    df["start_index"] = np.searchsorted(str_times, df["start_time"].values, side="left")
    df["end_index"] = np.searchsorted(str_times, df["end_time"].values, side="left")

    return df


def sign(x: float, thresh: float = 0) -> int:
    return 0 if abs(x) < thresh else 1 if x > 0 else -1


def is_between(a, x0, x1, strict: bool = False) -> bool:
    if strict:
        return a > x0 and a < x1
    return a >= x0 and a <= x1


def _to_set(row: pd.Series) -> set:
    return set(range(int(row["start_index"]), int(row["end_index"])))


def overlap_stats(truth: pd.DataFrame, guess: pd.DataFrame) -> tuple:
    """Calculate intersection over union (IoU), recall, and precision for sets of intervals

    Args:
        truth (pd.DataFrame): DataFrame with start_index and end_index columns to denote true interval boundaries
        guess (pd.DataFrame): DataFrame with start_index and end_index columns to denote guessed interval boundaries

    Returns:
        [tuple]: IoU, recall, precision
    """
    s1 = set()
    for _, row in truth.iterrows():
        s1.update(_to_set(row))

    s2 = set()
    for _, row in guess.iterrows():
        s2.update(_to_set(row))

    intersect = len(s1.intersection(s2))
    return intersect / len(s1.union(s2)), intersect / len(s1), intersect / len(s2)


def IoU(a: pc.Interval, b: pc.Interval, within: pc.Interval = None) -> float:
    overlap = min(a.end_index, b.end_index) - max(a.start_index, b.start_index)
    if within:
        return overlap / (within.end_index - within.start_index)
    u = (b.end_index - b.start_index) + (a.end_index - a.start_index)
    return float(overlap / (u - overlap))


def first_overlap(x: pc.Interval, intervals: list[pc.Interval], min_IoU: float = 0.5) -> pc.Interval:
    "Returns the first overlap"
    for qti in intervals:
        if x.start_index < qti.end_index and x.end_index > qti.start_index:
            if IoU(x, qti, x) > min_IoU:
                return qti


def ms_since_midnight(t: datetime.time) -> int:
    return (t.hour * 3600000) + (t.minute * 60000) + (t.second * 1000) + (t.microsecond / 1000)


# TODO singledispatch
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    "Normalize each signal by the standard deviation of the signal"
    return df / df.std()


def normalize_2_mV(df: pd.DataFrame, raw_to_mV: float = 78e-6) -> pd.DataFrame:
    "Normalize each signal to millivolts"
    return df * raw_to_mV


def suppress_channels(
    frame_signals_df: pd.DataFrame, frame_time: str, suppression_intervals: list[dict]
) -> pd.DataFrame:
    _df = frame_signals_df
    for intv in suppression_intervals:
        if frame_time >= intv["start_time"] and frame_time < intv["end_time"]:
            _df = _df.drop(columns=intv["channels"], errors="ignore")
    return _df


def merge_with_overrides(defaults: dict, overrides: dict) -> dict:
    if overrides is None:
        return defaults
    result = {}
    for k, v in defaults.items():
        if type(v) == dict:
            result[k] = merge_with_overrides(v, overrides.get(k))
        else:
            result[k] = overrides.get(k, v)
    return result


# deep merge
ABSENT = "__absent__"


def merge_dicts(v1, v2) -> dict:
    "Merge two dicts, giving v2 values precedence.  Preserve None values if explicitly set"
    if isinstance(v1, dict) and isinstance(v2, dict):
        return {k: merge_dicts(v1.get(k, ABSENT), v2.get(k, ABSENT)) for k in (list(v1.keys()) + list(v2.keys()))}
    else:
        return v1 if v2 == ABSENT else v2


def update_ntuple(x: tuple, new_vals: list[tuple]):
    for k, v in new_vals:
        x = x._replace(**{k: v})
    return x


def euc_distance(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    return sum([(x - y) ** 2 for x, y in zip(p1, p2)]) ** 0.5


def centroid(points: list[tuple[float]]) -> tuple[float]:
    "Calculate the centroid of a group of 3D points"
    return list(map(lambda idx: sum([p[idx] for p in points]) / len(points), [0, 1, 2]))


####################### Interval based utilities  ###############################


def intervals_from(intervals: Union[list[dict], pd.DataFrame]) -> list[pc.Interval]:
    if len(intervals):
        if isinstance(intervals, pd.DataFrame):
            intervals = intervals.to_dict(orient="records")
        return [pc.Interval(r["start_index"], r["end_index"]) for r in sorted(intervals, key=itemgetter("start_index"))]
    return []


def bitmask(intervals: list[pc.Interval], vector_len: int) -> np.ndarray:
    b, prev_x1 = np.array([]), 0
    for intv in intervals:
        x0, x1 = max(0, intv.start_index, prev_x1), min(vector_len, intv.end_index)
        b = np.concatenate([b, np.zeros(x0 - prev_x1), np.ones(x1 - x0)])
        prev_x1 = x1
    return np.concatenate([b, np.zeros(vector_len - prev_x1)])


def as_activity_vector(channel_to_intervals: dict[str, list[pc.Interval]], vector_len: int) -> np.ndarray:
    activity_vec = np.zeros(vector_len)
    for intervals in channel_to_intervals.values():
        activity_vec += bitmask(intervals, vector_len)
    return activity_vec


def intervals_complement(intervals: list[pc.Interval], max_index: int) -> list[pc.Interval]:
    compl, prev_end = [], 0
    for intv in intervals:
        compl.append(pc.Interval(prev_end, intv.start_index))
        prev_end = intv.end_index
    if max_index > prev_end:
        compl.append(pc.Interval(prev_end, max_index))
    return compl


def _group_into_intervals(activity_vector: np.ndarray, threshold: int) -> list[pc.Interval]:
    rvals, curr_x0 = [], None
    for i, v in enumerate(activity_vector):
        if v >= threshold:
            if not curr_x0:
                curr_x0 = i
        else:
            if curr_x0:
                rvals.append((curr_x0, i - 1))  # interval just ended
                curr_x0 = None
    if curr_x0:
        rvals.append((curr_x0, i))
    return [pc.Interval(*r) for r in rvals]


def intervals_of_activity(
    activity_threshold: int,
    channel_to_intervals: dict[str, list[pc.Interval]] = None,
    vector_len: int = None,
    activity_vector: np.ndarray = None,
) -> list[pc.Interval]:
    if activity_vector is None:
        activity_vector = as_activity_vector(channel_to_intervals, vector_len)
    return _group_into_intervals(activity_vector, activity_threshold)
