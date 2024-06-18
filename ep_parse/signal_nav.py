from datetime import datetime, timedelta
from typing import Union
import pandas as pd
import ep_parse.utils as u
from ep_parse.constants import CHANNEL_GROUPS
import ep_parse.case_data as d


def rf_to_stime(ablation_df: pd.DataFrame, rf_num: int, seconds_before: int = 0) -> str:
    rf_time = ablation_df[ablation_df["RF"] == rf_num]["start_time"].iloc[0]
    start_time = u.as_time_str(rf_time - timedelta(seconds=seconds_before))
    return start_time


def rf_to_etime(ablation_df: pd.DataFrame, rf_num: int, seconds_after: int = 0) -> str:
    rf_time = ablation_df[ablation_df["RF"] == rf_num]["end_time"].iloc[0]
    end_time = u.as_time_str(rf_time + timedelta(seconds=seconds_after))
    return end_time


def _query_with_open_store(signals_store: pd.HDFStore, channels: list[str], where: str) -> pd.DataFrame:
    "Run a query with the assumption that the signals_store is already open"
    rval = []
    if channels is None:
        for tbl in signals_store.keys():
            if tbl.startswith("/signals/"):
                rval.append(signals_store.select(tbl, where=where))
    else:
        table_2_channels = {
            k: set(channels).intersection(v) for k, v in CHANNEL_GROUPS.items() if set(channels).intersection(v)
        }
        for tbl, cols in table_2_channels.items():
            if f"/signals/{tbl}" in signals_store.keys():
                rval.append(signals_store.select(f"signals/{tbl}", columns=cols, where=where))
    return rval


def _query_by_time(
    start_time: datetime,
    end_time: datetime,
    channels: list[str],
    signals_store: pd.HDFStore = None,
    case_id: str = None,
) -> pd.DataFrame:
    result = None
    where = f"index >= '{start_time.isoformat()}' and index < '{end_time.isoformat()}'"
    if signals_store is None:
        assert case_id, "User must pass an open HDFStore object or a case id to query by time"
        with d.case_signals_db(case_id=case_id, mode="r") as signals_store:
            result = _query_with_open_store(signals_store, channels, where)
    else:
        result = _query_with_open_store(signals_store, channels, where)

    if result:
        df = pd.concat(result, axis="columns")
        return df.dropna(axis="columns", thresh=int(len(df) / 4))
        # return pic.standardize_channel_names(pd.concat(r, axis="columns"))
    return pd.DataFrame()


def lookup_by_time(
    start_time: Union[str, datetime],
    window_size: int = 4,
    end_time: Union[str, datetime] = None,
    channels: list[str] = None,
    signals_store: pd.HDFStore = None,
    case_id: str = None,
) -> pd.DataFrame:
    if isinstance(start_time, str):
        start_time = u.as_datetime(start_time)
    if end_time:
        end_time = u.as_datetime(end_time) if isinstance(end_time, str) else end_time
    else:
        end_time = start_time + timedelta(seconds=window_size)

    return _query_by_time(start_time, end_time, channels, signals_store=signals_store, case_id=case_id)


def lookup_by_rf(
    ablation_df: pd.DataFrame,
    rf_num: int,
    seconds_before: int = 4,
    seconds_after: int = 0,
    over_ON_interval: bool = False,
    channels: list[str] = None,
    signals_store: pd.HDFStore = None,
    case_id: str = None,
) -> pd.DataFrame:
    """Lookup signals by RF number

    Args:
        ablation_df (pd.DataFrame): ablations events for the case
        rf_num (int): The RF number to lookup
        seconds_before (int, optional): Seconds of data to fetch from before the instance. Defaults to 4.
        seconds_after (int, optional): Seconds data to fetch from after the instance. Defaults to 0.
        over_ON_interval (bool, optional): Return data for the entire RF ON interval. Defaults to False.
        channels (list[str], optional): List of channels to fetch, fetches all if None. Defaults to None.
        signals_store (pd.HDFStore, optional): HDF store containing the case signals. Defaults to None.
        case_id (str, optional): Id of the case. Defaults to None.

    Returns:
        pd.DataFrame: time series signal data for the channels
    """
    if ablation_df is None and case_id:
        ablation_df = d.load_local_rfs(case_id)
    _rf_row = ablation_df[ablation_df["RF"] == rf_num].iloc[0]
    s_time = _rf_row["start_time"]
    e_time = (_rf_row["end_time"] if over_ON_interval else s_time) + timedelta(seconds=seconds_after)
    s_time -= timedelta(seconds=seconds_before)
    return _query_by_time(s_time, e_time, channels, signals_store=signals_store, case_id=case_id)


def lookup_by_event_id(
    event_df: pd.DataFrame,
    idx: int,
    seconds_before: int = 2,
    seconds_after: int = 2,
    channels: list[str] = None,
    signals_store: pd.HDFStore = None,
    case_id: str = None,
) -> pd.DataFrame:
    evt_time = event_df["time"].iloc[idx]
    s_time = evt_time - timedelta(seconds=seconds_before)
    e_time = evt_time + timedelta(seconds=seconds_after)
    return _query_by_time(s_time, e_time, channels, signals_store=signals_store, case_id=case_id)


def _tdiff(bkmrk_t, t) -> int:
    return (t - bkmrk_t).seconds if t > bkmrk_t else (bkmrk_t - t).seconds


def most_recent_rf(ablations: pd.DataFrame, time_str: str, seconds_prior: int = 0, strictly_before: bool = True) -> int:
    _time = u.as_datetime(time_str) + timedelta(seconds=seconds_prior)
    i = ablations["start_time"].apply(lambda t_on: _tdiff(t_on, _time)).idxmin()
    if strictly_before:
        i = i - (1 if ablations["start_time"].iloc[i] > _time else 0)
    return ablations["RF"].iloc[i]


def time_str_seq(start_time: str, end_time: str, step: int = 5) -> list:
    """Return a sequence of HH:mm:ss formatted times, starting at start_time and ending
    at end_time.  Each item will be <step> seconds later than the previous entry.

    Args:
        start_time (str): starting time for the sequence.  Format is HH:MM:ss
        end_time (str): ending time for the sequence.  Format is HH:MM:ss
        step (int, optional): sequence step size, in seconds. Defaults to 5.
    """

    stime = u.as_datetime(start_time)
    interval_len = (u.as_datetime(end_time) - stime).seconds
    assert interval_len >= 0, "start time must be less than end time"

    return [u.as_time_str(stime + timedelta(seconds=step * i)) for i in range(int(interval_len / step))]


def parse_time_str(_time: str) -> list:
    parts = _time.strip().split("|")
    if len(parts) == 1:
        return [parts[0]]

    assert len(parts) == 4, "Invalid time string format.  Format is '11:20:01' or '11:22:12|before|12|2'"
    _time = u.as_datetime(parts[0])
    _dur, _step = int(parts[2]), int(parts[3])
    times = []

    if parts[1] == "before":
        _time = _time - timedelta(seconds=_dur)

    for i in range(0, _dur, _step):
        times.append(u.as_time_str(_time + timedelta(seconds=i)))

    return sorted(times)
