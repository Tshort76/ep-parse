import os
import json
import toolz as tz
import yaml
import pandas as pd
import logging
from operator import itemgetter
from enum import StrEnum
from dotenv import load_dotenv

load_dotenv()

import ep_parse.utils as pu
import ep_parse.constants as pc
import ep_parse.yaml_helper as ymlh
import ep_parse.common as pic
import ep_parse.tag as tg

log = logging.getLogger(__name__)


class FileType(StrEnum):
    META = "meta.yaml"
    TAGS = "tags.json"
    EPSYSTEM_EVENTS = "epsystem_events.csv"
    SIGNALS = "signals.h5"
    EPLOG = "epmed_bookmarks.txt"
    CARDIOLAB_LOG = "cardiolab_logs.txt"


def case_file_path(case_id: str, file_type: FileType | pic.DataSource) -> str:
    return os.path.join(
        os.environ["data_filepath"], case_id, (f"{case_id}_" if isinstance(file_type, FileType) else "") + file_type
    )


def case_signals_db(case_id: str, mode: str = "r", compression: bool = True) -> pd.HDFStore:
    """Create the HDF store resource associated with a case, if it exists.  User is responsible for opening/closing the resource.

    Args:
        case_id (str): Case Id for the case to be loaded
        mode (str, optional): "r" - read, "w" - write. Defaults to "r".

    Returns:
        HDFStore: HDF store containing signals timeseries data, located under keys /signals/<signal_group>
    """
    if case_id:
        hdf_file = case_file_path(case_id, FileType.SIGNALS)
        if mode == "r":
            assert os.path.exists(hdf_file), f"Case signals have not been loaded for {case_id}"
        return pd.HDFStore(hdf_file, complevel=1 if compression else None, mode=mode)


def case_PPS(case_id: str):
    with case_signals_db(case_id) as s:
        for g in s.keys():
            if ("signals/") in g:
                return pu.points_per_second(s.select(g, stop=2).index)


def case_channels(case_id: str):
    channels = []
    with case_signals_db(case_id) as s:
        for g in s.keys():
            if ("signals/") in g:
                a = s.select(g, stop=10).notna().sum()
                channels += a[a > 5].index.to_list()
    return channels


def store_pps_in_meta(case_id: str) -> int:
    pps = case_PPS(case_id)
    update_case_meta(case_id, ["signal_sample_rate_hz"], pps)


def create_case_meta(case_id: str) -> None:
    meta_file = case_file_path(case_id, FileType.META)
    if os.path.exists(meta_file):
        return
    dir_path = os.path.join(os.environ["data_filepath"], case_id)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # with open("resources/default_meta_attributes.json", "r") as fp:
    #     meta_data = json.load(fp)
    meta_data = {"case_id": case_id}

    for epsys in pic.EP_SYSTEM:
        if os.path.exists(os.path.join(dir_path, epsys.value)):
            meta_data["ep_system"] = epsys.value
            break

    for mapsys in pic.MAPPING_SYSTEM:
        if os.path.exists(os.path.join(dir_path, mapsys.value)):
            meta_data["mapping_system"] = mapsys.value
            break

    meta_data["md_catheter?"] = os.path.exists(os.path.join(dir_path, "md_catheter_logs"))

    with open(meta_file, "w") as fp:
        yaml.dump(dict(sorted(meta_data.items())), fp, indent=2)


def load_case_tags(case_id: str):
    tags_file = case_file_path(case_id, FileType.TAGS)
    if case_id and os.path.exists(tags_file):
        with open(tags_file, "r") as fp:
            tags = json.load(fp)

        return list(sorted(tags, key=itemgetter("start_time")))
    return []


def write_case_tags(case_id: str, tags: list[dict], mode: str = "w"):
    "modes: a = append, w = write (truncate), r = replace (by type)"
    if case_id:
        tags_file = case_file_path(case_id, FileType.TAGS)
        if not os.path.exists(tags_file):
            mode = "w"

        if os.path.exists(tags_file):
            old_tags = load_case_tags(case_id)
            if mode == "a":
                tags = old_tags + tags
            elif mode == "r":
                replace_type = tg.tag_type(tags[0])
                tags = tags + [t for t in old_tags if tg.tag_type(t) != replace_type]

        with open(tags_file, "w") as fp:
            json.dump(list(sorted(tags, key=itemgetter("start_time"))), fp, indent=2)


def all_tagged_cases() -> list[str]:
    tagged_cases = [
        x for x in os.listdir(os.environ["data_filepath"]) if os.path.exists(case_file_path(x, FileType.TAGS))
    ]
    return list(sorted(tagged_cases))


def associated_maps(case_id: str) -> list[str]:
    "list of all unique map_ids associated with the case"
    tags = load_case_tags(case_id)
    maps = {m.get("map_id") for m in tags}
    maps.discard(None)
    return list(sorted(maps))


# TODO memoize this
def load_case_meta(case_id: str) -> dict:
    meta_file = case_file_path(case_id, FileType.META)
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = yaml.safe_load(f)
    else:
        meta = {}

    meta["feature_detection_params"] = pu.merge_with_overrides(
        pc.DEFAULT_FEATURE_DETECTION_PARAMS,
        meta.get("feature_detection_params"),
    )
    meta = tz.assoc_in(meta, ["feature_detection_params", "qrs"], meta.get("qrs_meta"))
    return meta


def update_case_meta(case_id: str, key_seq: list, data, append: bool = True):
    meta_file = case_file_path(case_id, FileType.META)
    return ymlh.append_to_yaml(meta_file, key_seq, data, append)


def load_local_rfs(case_id: str = None, event_df: pd.DataFrame = None) -> pd.DataFrame:
    if event_df is None:
        return load_local_events(case_id, rfs_only=True)
    rf_df = event_df[event_df["event"].str.startswith("RF")].reset_index(drop=True)
    rf_df["RF"] = rf_df["event"].str[3:].astype("int32")
    return rf_df[["RF", "start_time", "end_time"]]


def load_local_events(case_id: str, rfs_only: bool = False) -> pd.DataFrame:
    "Read local tag file and parse out events and rfs"
    tags = load_case_tags(case_id)
    if tags:
        event_df = pd.DataFrame(tags)
        event_df = event_df[event_df.columns.intersection(["label", "start_time", "end_time", "notes"])]
        for f in ["start_time", "end_time"]:
            event_df[f] = event_df[f].map(lambda s: pu.as_datetime(pu.as_time_str(s)) if s else pd.NaT)
        event_df.rename(columns={"label": "event"}, inplace=True)
        if rfs_only:
            event_df = event_df[event_df["event"].str.startswith("RF")]
            event_df = load_local_rfs(event_df=event_df)
        return event_df