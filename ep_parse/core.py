import logging
import os
import pandas as pd
from typing import Union

import ep_parse.utils as u
import ep_parse.epmed.core as epm
import ep_parse.epmed.file_cleanup as epcln
import ep_parse.common as pic
import ep_parse.cardiolab.core as clc
import ep_parse.carto.core as crto
import ep_parse.case_data as d
import ep_parse.ensite.core as ensc

# We import these for convenience so that callers can just import data_import for 90% of import functions
from ep_parse.case_data import load_case_meta
from ep_parse.signal_nav import lookup_by_time, lookup_by_rf, lookup_by_event_id
from ep_parse.misc_sources import md_catheter_logs_as_bookmark_file, store_md_cath_offset
from ep_parse.ensite.core import store_ensite_offset, import_ensite_export

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG) if u.is_dev_mode() else log.setLevel(logging.ERROR)


SIGNAL_SOURCE_BY_SYSTEM = {pic.EP_SYSTEM.CARDIOLAB: pic.DataSource.CARDIOLAB, pic.EP_SYSTEM.EPMED: pic.DataSource.EPMED}


def only_rf_events(case_id: str, event_df: pd.DataFrame) -> pd.DataFrame:
    if "notes" in event_df:  # tags
        return d.load_local_rfs(event_df=event_df)
    match d.get_EP_system(case_id):
        case pic.EP_SYSTEM.EPMED:
            return epm.parse_rfs(event_df=event_df)
        case pic.EP_SYSTEM.CARDIOLAB:
            return clc.parse_rfs(event_df)


def format_and_store_case_export(
    case_id: str, case_directory: str = None, only: list[str] = None, compress: bool = True
):
    "Parse epsystem files, convert them into DataFrames, and stores those DataFrames in a standard location"
    only = only or {"signals", "events"}
    ep_system = d.get_EP_system(case_id)
    case_directory = case_directory or d.case_file_path(case_id, SIGNAL_SOURCE_BY_SYSTEM[ep_system])
    if "signals" in only:
        assert (
            case_directory is not None
        ), "Could not find the export directory for the case, please put it in a standard location, update your case_path var, or pass an explicit case_dir"

        case_meta = d.load_case_meta(case_id=case_id)
        channels = (
            case_meta["channels"]
            if (case_meta and "channels" in case_meta)
            else log.warning(f"No meta file found for {case_id}, loading all signals")
        )

        match ep_system:
            case pic.EP_SYSTEM.EPMED:
                epcln.clean_epmed_export(case_directory, qa_checks=False)
                epm.parse_signal_files(case_id, case_directory, channels=channels, compress=compress)
            case pic.EP_SYSTEM.CARDIOLAB:
                clc.parse_signals(case_id, case_directory, channels=channels, compress=compress)

    # ? might consider storing events_df within the case HDF file (misc/events) instead of in a separate parquet file
    if "events" in only:
        match ep_system:
            case pic.EP_SYSTEM.EPMED:
                event_df = epm.parse_bookmark_file(case_id=case_id)
            case pic.EP_SYSTEM.CARDIOLAB:
                event_df = clc.parse_log_file(case_id=case_id)

        if event_df is not None:
            csv_file = d.case_file_path(case_id, d.FileType.EPSYSTEM_EVENTS)
            event_df.replace("", pd.NA).dropna().to_csv(csv_file, index=False)


def load_events(case_id: str, outputs: list[str]) -> Union[pd.DataFrame, list[pd.DataFrame]]:
    """Load a variable number of event logs associated with the case

    Args:
        case_id (str): case_id
        outputs (list[str]): sequence of event dataframes to return. Subset of {'atrium_events', 'atrium_RFs', 'ep_system_events', 'ep_system_RFs', 'RFs'}

    Returns:
        list[pd.DataFrame]: A single dataframe, or list representing the requested event types, order mirrors outputs param.
    """
    results = {}
    if {"atrium_events", "atrium_RFs", "RFs"}.intersection(set(outputs)):
        results["atrium_events"] = d.load_local_events(case_id)
        if results["atrium_events"] is not None:
            results["atrium_RFs"] = d.load_local_rfs(event_df=results["atrium_events"])
            if "RFs" in outputs and results["atrium_RFs"] is not None and not results["atrium_RFs"].empty:
                results["RFs"] = results["atrium_RFs"]

    still_need_rfs = "RFs" in outputs and "RFs" not in results
    if {"ep_system_events", "ep_system_RFs"}.intersection(set(outputs)) or still_need_rfs:
        ep_log_file = d.case_file_path(case_id, d.FileType.EPSYSTEM_EVENTS)
        if not os.path.exists(ep_log_file):
            log.info(f"No epmed system log file found for {case_id}, parsing and writing file now ...")
            format_and_store_case_export(case_id, only={"events"})
        if not os.path.exists(ep_log_file):
            return None
        results["ep_system_events"] = pd.read_csv(ep_log_file, parse_dates=["time"]).sort_values(
            "time", ignore_index=True
        )
        if "ep_system_RFs" in outputs or still_need_rfs:
            results["ep_system_RFs"] = only_rf_events(case_id, results["ep_system_events"])
            if still_need_rfs:
                results["RFs"] = results["ep_system_RFs"]

    out = [results[k] for k in outputs]
    return out[0] if len(outputs) == 1 else out


def load_rf_events(case_id: str) -> pd.DataFrame:
    return load_events(case_id, outputs=["ep_system_RFs"])


def load_case_signals(case_id: str) -> pd.HDFStore:
    signals_file = d.case_file_path(case_id, d.FileType.SIGNALS)
    if not os.path.exists(signals_file):
        log.info(f"Signals HDF file not found for {case_id}, parsing and storing now ...")
        format_and_store_case_export(case_id, only={"signals"})
    return d.case_signals_db(case_id)


def generate_tags_from_ensite(case_id: str, parse_types: list[str] = ["geometry", "RF", "MAP"]):
    return import_ensite_export(case_id, parse_types, lambda cid: load_events(cid, ["ep_system_RFs"]))


def infer_map_system_offset(case_id: str) -> float:
    match d.get_MAPPING_system(case_id):
        case pic.MAPPING_SYSTEM.ENSITE:
            return ensc.infer_ensite_offset(case_id, lambda cid: load_events(cid, ["ep_system_RFs"]))


def generate_tags_from_carto(case_id: str, parse_types: list[str] = ["geometry", "RF", "MAP"], opts: dict = {}):
    return crto.import_carto_export(case_id, parse_types, lambda cid: load_events(cid, ["ep_system_RFs"]), opts)


def _update_meta_channels(case_id: str, raw_channel_names: list[str]) -> None:
    raw_to_atrium = pic.raw_name_to_std_name(raw_channel_names)
    meta_channels = [x for x in raw_to_atrium.values() if x]
    d.update_case_meta(case_id, ["channels"], meta_channels[::-1], append=False)


def plot_raw_data_coverage(case_id: str):
    ep_system = d.get_EP_system(case_id)
    case_directory = d.case_file_path(case_id, SIGNAL_SOURCE_BY_SYSTEM[ep_system])
    assert case_directory, f"Could not find the raw export directory for {case_id}"

    match ep_system:
        case pic.EP_SYSTEM.EPMED:
            bookmarks_df = load_rf_events(case_id)
            epcln.clean_epmed_export(case_directory, qa_checks=False)
            coverage_df = epm.bin_coverage(case_directory)
        case pic.EP_SYSTEM.CARDIOLAB:
            bookmarks_df = load_rf_events(case_id)
            coverage_df = clc.bin_coverage(case_directory)
        case _:
            raise NotImplementedError("Raw data coverage is not supported for this case")

    _update_meta_channels(case_id, coverage_df["raw_name"].unique().tolist())
    return pic.data_coverage_plot(coverage_df, bookmarks_df)
