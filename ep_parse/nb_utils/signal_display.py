from datetime import timedelta
import pandas as pd

import ep_parse.signal_nav as sn
import ep_parse.core as core
import ep_parse.utils as u
import ep_parse.plots.matplot_plots as m_plt
import ep_parse.case_data as d
import ep_parse.nb_utils.common as nbc


def _plot_raw_signals(signals: pd.DataFrame, configs: dict, title: str, channels: list[str] = None):
    y_ax = configs.get("y_scale", 2)
    figsize = (configs.get("width", 50), min(30, configs.get("height")))
    columns = signals.columns.intersection(channels) if channels else signals.columns
    m_plt.plot_raw_signals(signals_df=signals[columns], title=title, y_axis=(-y_ax, y_ax), figsize=figsize)


def _signals_for_frame_id(
    _id: int,
    signals_store: pd.HDFStore,
    configs: dict,
    event_df: pd.DataFrame = None,
    tag_by_label: dict[str, dict] = None,
    case_annotations: pd.DataFrame = None,
) -> tuple[pd.DataFrame, str]:
    sbefore, safter, delineate_by = [configs.get(k) for k in ["prepend_seconds", "append_seconds", "delineate_by"]]
    match delineate_by:
        case "TIME":
            _time = u.as_datetime(_id)
            frame_signals = sn.lookup_by_time(
                signals_store=signals_store,
                start_time=_time - timedelta(seconds=sbefore),
                end_time=_time + timedelta(seconds=safter),
            )
            notes = None
        case "TAG":
            tag = tag_by_label[_id]
            stime = u.as_datetime(tag["start_time"])
            etime = u.as_datetime(tag["end_time"])
            notes = f"Start: {u.as_time_str(tag['start_time'], False)} , End: {u.as_time_str(tag['end_time'], False)}\n{tag.get('notes') or ''}"
            frame_signals = sn.lookup_by_time(
                signals_store=signals_store,
                start_time=stime - timedelta(seconds=sbefore),
                end_time=etime + timedelta(seconds=safter),
            )
        case "RF":
            frame_signals = sn.lookup_by_rf(
                signals_store=signals_store,
                ablation_df=event_df,
                rf_num=_id,
                seconds_before=sbefore,
                seconds_after=safter,
                over_ON_interval=True if configs.get("entire_RF_ON_duration") else False,
            )
            notes = (
                f"RF ON Time: {u.as_time_str(frame_signals.index[0] + timedelta(seconds=sbefore), False)}"
                if len(frame_signals.index)
                else None
            )
        case "EVENT":
            frame_signals = sn.lookup_by_event_id(
                event_df, _id, seconds_before=sbefore, seconds_after=safter, signals_store=signals_store
            )
            notes = (
                f"Event Time: {u.as_time_str(frame_signals.index[0] + timedelta(seconds=sbefore), False)}"
                if len(frame_signals.index)
                else None
            )
        case "ANNOTATION_LINE":
            ann = case_annotations.loc[_id]
            stime, etime = u.as_datetime(ann["start_time"]), u.as_datetime(ann["end_time"])
            frame_signals = sn.lookup_by_time(
                signals_store=signals_store,
                start_time=stime - timedelta(seconds=sbefore),
                end_time=etime + timedelta(seconds=safter),
            )
            if configs.get("only_annotated_channel"):
                frame_signals = frame_signals[[ann["channel"]]]  # as dataframe
            notes = None
        case _:
            frame_signals, notes = None, ""
    return frame_signals, notes


def plot_undecorated_signals(case_id: str, configs: dict, event_df: pd.DataFrame = None):
    sconfigs = configs["signals_plot"]
    delineate_by, frame_ids = [configs.get(k) for k in ["delineate_by", "delineation_ids"]]
    rf_df = core.only_rf_events(case_id, event_df=event_df) if delineate_by == "RF" else None

    case_hdf = d.case_file_path(case_id, d.FileType.SIGNALS)
    with pd.HDFStore(case_hdf, mode="r") as sstore:
        for i, _id in enumerate(frame_ids):
            frame_signals, notes = _signals_for_frame_id(
                _id, sstore, {**sconfigs, "delineate_by": delineate_by}, event_df=rf_df
            )

            plot_channels = frame_signals.columns.difference(sconfigs.get("suppress_channels", [])).tolist()
            frame_signals = frame_signals[plot_channels]
            title = f"{delineate_by}: {frame_ids[i]} : {notes}"
            _plot_raw_signals(frame_signals, sconfigs, title, plot_channels)


def simple_signals_plot(case_id: str, delineate_by: str, delin_id_str: str, user_plot_configs: dict):
    assert delineate_by in ("RF", "TIME"), "Invalid delineate by option.  Must be 'TIME' or 'RF'"
    id_parse_fn = nbc.parse_rf_str if delineate_by == "RF" else nbc.parse_user_times

    pconfigs = {
        "signals_plot": user_plot_configs,
        "delineate_by": delineate_by,
        "delineation_ids": id_parse_fn(delin_id_str),
    }

    event_df = core.load_events(case_id, outputs=["ep_system_events"]) if delineate_by == "RF" else None

    return plot_undecorated_signals(case_id, pconfigs, event_df)
