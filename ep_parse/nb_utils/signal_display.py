from datetime import timedelta
import pandas as pd
from IPython.display import display

import ep_parse.signal_nav as sn
import ep_parse.plots.frame as ppf
import ep_parse.core as core
import ep_parse.features.core as pfc
import ep_parse.utils as pu
import ep_parse.plots.matplot_plots as pv
import ep_parse.case_data as cdata
from ep_parse.constants import CHANNEL_GROUPS, DEFAULT_PWAVE_PARAMS
import ep_parse.catheters as cath


def _disease_channels(ann_channels: list[str]) -> list[str]:
    return list(set(cath.mapping_channels()).intersection(ann_channels))


def _add_vline(fig, x, label: str):
    fig.add_vline(
        x=x,
        line={"dash": "solid", "color": "magenta", "width": 3},
        annotation={
            "text": label,
            "xanchor": "right",
        },
    )
    return fig


def _plot_signals(frame: dict, configs: dict, title: str, filename: str = None):
    y_ax = configs.get("y_scale", 2)
    defl_level = (
        True
        if configs.get("show_p_intervals")
        else ("extra_wavelet_only" if configs.get("show_wavelet_residuals") else None)
    )
    fig = ppf.plot_frame(
        frame,
        fig_opts={"title": title, "y_axis": (-y_ax, y_ax)},
        display_opts={
            **configs,
            "disease_channels": configs.get("disease_channels", []),
            "show_deflections": defl_level,
            "signal_hoverinfo": "empty" if configs.get("clickable_lines", False) else None,
        },
        filename=filename,
    )

    _scale = frame.get("meta", {}).get("points_per_second", 2000) / 1000
    _key = next(iter(frame["channel_to_features"].keys()), None)
    sig_len_ms = (len(frame["channel_to_features"][_key].get("signal", [])) if _key else 0) / _scale

    sbefore, safter = [configs.get(k) for k in ["prepend_seconds", "append_seconds"]]
    x0, x1 = [sbefore * _scale * 1000, (sig_len_ms - (safter * 1000)) * _scale]
    if abs(x1 - x0) < 50:
        fig = _add_vline(fig, x0, "")
    else:
        fig = _add_vline(fig, x0, "start")
        fig = _add_vline(fig, x1, "end")

    _autosize = ("width_ms" not in configs) or (sig_len_ms < 100)
    if not _autosize:
        _width = int((2000 / configs.get("width_ms")) * sig_len_ms)

        fig.update_layout(
            autosize=False,
            width=_width,
            height=configs.get("height", 300),
        )

    return fig


def _plot_raw_signals(signals: pd.DataFrame, configs: dict, title: str, channels: list[str] = None):
    y_ax = configs.get("y_scale", 2)
    figsize = (configs.get("width", 50), min(30, configs.get("height")))
    columns = signals.columns.intersection(channels) if channels else signals.columns
    pv.plot_raw_signals(signals_df=signals[columns], title=title, y_axis=(-y_ax, y_ax), figsize=figsize)


def _add_sme_annotations_to_frame(anns: pd.DataFrame, frame: dict, line_num: int = None):
    ch_features = frame["channel_to_features"]
    channels = list(ch_features.keys())

    stime, etime = [pu.as_time_str(frame["index"][i], False) for i in (0, -1)]

    frame_anns = anns[(anns["channel"].isin(channels)) & (anns["start_time"] >= stime) & (anns["end_time"] <= etime)]
    if line_num:
        frame_anns = frame_anns.loc[line_num:line_num]

    for line_num, ann in frame_anns.to_dict(orient="index").items():
        ch = ann["channel"]
        # is the channel present and decorated?
        if ch in ch_features:  # and len(ch_features[ch].keys()) > 1:
            ch_features[ch]["sme_annotations"] = pd.concat(
                [
                    ch_features[ch].get("sme_annotations", pd.DataFrame()),
                    pd.DataFrame([{**ann, "line_num": line_num, "color": None}]),
                ]
            )


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
            _time = pu.as_datetime(_id)
            frame_signals = sn.lookup_by_time(
                signals_store=signals_store,
                start_time=_time - timedelta(seconds=sbefore),
                end_time=_time + timedelta(seconds=safter),
            )
            notes = None
        case "TAG":
            tag = tag_by_label[_id]
            stime = pu.as_datetime(tag["start_time"])
            etime = pu.as_datetime(tag["end_time"])
            notes = f"Start: {pu.as_time_str(tag['start_time'], False)} , End: {pu.as_time_str(tag['end_time'], False)}\n{tag.get('notes') or ''}"
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
                f"RF ON Time: {pu.as_time_str(frame_signals.index[0] + timedelta(seconds=sbefore), False)}"
                if len(frame_signals.index)
                else None
            )
        case "EVENT":
            frame_signals = sn.lookup_by_event_id(
                event_df, _id, seconds_before=sbefore, seconds_after=safter, signals_store=signals_store
            )
            notes = (
                f"Event Time: {pu.as_time_str(frame_signals.index[0] + timedelta(seconds=sbefore), False)}"
                if len(frame_signals.index)
                else None
            )
        case "ANNOTATION_LINE":
            ann = case_annotations.loc[_id]
            stime, etime = pu.as_datetime(ann["start_time"]), pu.as_datetime(ann["end_time"])
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


def _relevant_sme_annotations(case_id: str = None, sigUI_annotations: pd.DataFrame = None) -> pd.DataFrame:
    sigUI_annotations = sigUI_annotations if sigUI_annotations is not None else cdata.load_sigUI_annotations(case_id)
    cat_props = cdata.load_ui_configs("signal_annotation")["annotation_target_properties"]
    sigUI_annotations["interval_type"] = [
        cat_props.get(c, {}).get("interval") for c in sigUI_annotations["category"].values
    ]
    sigUI_annotations["color"] = [cat_props.get(c, {}).get("color") for c in sigUI_annotations["category"].values]
    return sigUI_annotations[
        ~sigUI_annotations["category"].isin([c for c in cat_props if cat_props[c].get("hide_on_reload")])
    ]


def nbplot_signals(
    case_id: str,
    configs: dict,
    event_df: pd.DataFrame = None,
    tags: list[dict] = [],
    case_annotations: pd.DataFrame = None,
    ui_environment: str = "jupyter",
):
    configs = configs.copy()
    sconfigs = configs["signals_plot"]
    if sconfigs.get("show_disease_bars"):
        sconfigs["disease_channels"] = _disease_channels(sconfigs["annotated_channels"])
        configs["feature_detection_params"] = cdata.load_models(configs["feature_detection_params"])
    if sconfigs.get("show_p_intervals") or sconfigs.get("show_p_waves"):
        configs["feature_detection_params"]["p_waves"] = (
            configs["feature_detection_params"].get("p_waves") or DEFAULT_PWAVE_PARAMS
        )
    # else:
    #     configs["feature_detection_params"].pop("disease_scores", None)
    delineate_by, frame_ids = [configs.get(k) for k in ["delineate_by", "delineation_ids"]]
    tag_by_label = {tag["label"]: tag for tag in tags if tag.get("label")}
    if delineate_by == "RF":
        event_df = core.only_rf_events(case_id, event_df=event_df)

    # add SME annotations from signal annotation UI
    sme_annotations = _relevant_sme_annotations(case_id, case_annotations)

    j = 0
    # Generate the png files.  Give them a padded integer filename to ensure ordering during conversion to pdf
    id_to_fig = {}
    with cdata.case_signals_db(case_id) as sstore:
        for i, _id in enumerate(frame_ids):
            annotated_channels = sconfigs.get("annotated_channels", [])
            frame_signals, notes = _signals_for_frame_id(
                _id,
                sstore,
                {**sconfigs, "delineate_by": delineate_by},
                event_df=event_df,
                tag_by_label=tag_by_label,
                case_annotations=case_annotations,
            )

            if sconfigs.get("display_channels"):
                plot_channels = frame_signals.columns.intersection(
                    sconfigs.get("display_channels") + annotated_channels
                )
            else:
                plot_channels = frame_signals.columns.difference(sconfigs.get("suppress_channels", []))

            if sconfigs.get("only_annotated_channel"):
                annotated_channels, plot_channels = set(frame_signals.columns), set(frame_signals.columns)

            bare_channels = list(plot_channels.difference(annotated_channels))
            frame_signals = frame_signals[plot_channels]
            title = f"{delineate_by}: {frame_ids[i]}" + (
                f"  {event_df.iloc[_id]['event']}" if delineate_by == "EVENT" else ""
            )
            fname = str(j).zfill(4) if (configs.get("create_pdf") and ui_environment == "jupyter") else None

            if annotated_channels or ui_environment == "dash":
                frame = pfc.decorate_frame(
                    frame_signals,
                    configs["feature_detection_params"],
                    opts={"bare_channels": bare_channels},
                )
                _add_sme_annotations_to_frame(sme_annotations, frame)  # modifies frame in place
                sconfigs["clickable_lines"] = bool(ui_environment == "dash")
                fig = _plot_signals(frame, sconfigs, title, filename=fname)
                if ui_environment == "dash":
                    fid = pu.as_time_str(frame_signals.index[0]) if len(frame_signals) > 1 else _id
                    fig.update_layout(hoverdistance=5)
                    id_to_fig[fid] = fig
                else:
                    display(fig)
            else:
                _plot_raw_signals(frame_signals, sconfigs, title, bare_channels)

            if notes and ui_environment == "jupyter":
                print(notes)
            j += 1

    return id_to_fig


def plot_undecorated_signals(case_id: str, configs: dict, event_df: pd.DataFrame = None):
    sconfigs = configs["signals_plot"]
    delineate_by, frame_ids = [configs.get(k) for k in ["delineate_by", "delineation_ids"]]
    if delineate_by == "RF":
        event_df = core.only_rf_events(case_id, event_df=event_df)

    case_hdf = cdata.case_file_path(case_id, cdata.FileType.SIGNALS)
    with pd.HDFStore(case_hdf, mode="r") as sstore:
        for i, _id in enumerate(frame_ids):
            frame_signals, notes = _signals_for_frame_id(
                _id, sstore, {**sconfigs, "delineate_by": delineate_by}, event_df=event_df
            )

            plot_channels = frame_signals.columns.difference(sconfigs.get("suppress_channels", [])).tolist()
            frame_signals = frame_signals[plot_channels]
            title = f"{delineate_by}: {frame_ids[i]} : {notes}"
            _plot_raw_signals(frame_signals, sconfigs, title, plot_channels)
