import pandas as pd
import logging
import toolz as tz

import ep_parse.constants as pc
import ep_parse.utils as pu
import ep_parse.catheters as cath
import ep_parse.features.signal as pfs
import ep_parse.features.deflections as dfl
import ep_parse.features.wavelets as pfw
import ep_parse.features.qx_intervals as qxi

log = logging.getLogger(__name__)


def _frame_qa_checks(signals: pd.DataFrame) -> None:
    # highest legit value seen is 0.7 range
    if extreme_val_cols := signals.columns[(signals.max() > 10)].to_list():
        log.warn(f"Extreme signal values found at {pu.as_time_str(signals.index[0])} for columns: {extreme_val_cols}")


def decorate_signal(norm_signal: pd.Series, feature_params: dict = {}, exo_signal: dict = {}) -> dict:
    """Decorates a normalized signal by creating a dictionary.  The deflections key is
       always calculated, regardless of feature_params values.

    Arguments:
        norm_signal {pd.Series} -- A continuous sequence of voltage measurements

    Keyword Arguments:
        feature_params {dict} -- A specification of the form : desired_feature_name -> parameters for feature extractor
                                (e.g.: {'deflections' : {'min_slope' : 0.05}})

    Returns:
        dict -- requested_feature -> feature data
    """
    if norm_signal.empty or (sum(norm_signal.isna()) / len(norm_signal) > 0.5):
        log.debug("Attempted to decorate a signal that was mostly or entirely empty!")
        return {}

    PPS = pu.points_per_second(norm_signal.index)
    if PPS == 0:
        log.debug(f"Points per second was calculated as 0, check data for time {norm_signal.index[0]}")
        return {}
    signal = norm_signal.reset_index(drop=True)
    if bw_filter_cutoff := feature_params.get("butterworth_cutoff"):
        import ep_parse.bandwidth_filter as bwf

        signal = pd.Series(bwf.butter_filter(signal, PPS, bw_filter_cutoff))
    rval = {"signal": signal, "index": norm_signal.index}

    # deflections are always added
    deflect_params = feature_params.get("deflections", {})
    rval["deflections"] = dfl.extract_deflections(signal, PPS, **deflect_params)

    qrs_intervals = exo_signal.get("qrs_intervals")
    if qrs_intervals is not None:
        rval["deflections"] = qxi.remove_deflections_in_qrs(rval["deflections"], qrs_intervals)

    if len(rval["deflections"]) < 2:
        return {"signal": signal}

    if wparams := feature_params.get("wavelets"):
        rval["wavelets"] = pfw.partition_into_wavelets(signal, PPS, rval["deflections"], wparams)
        rval["interwavelet"] = pfs.interwavelet_features(
            rval["wavelets"], PPS, qrs_intervals, pu.points_as_ms(len(signal), PPS)
        )

    if feature_params.get("feature_vector", True) and "wavelets" in rval:
        rval["feature_vector"] = {**pfs.core_signal_features(signal, PPS, rval["deflections"]), **rval["interwavelet"]}

    return rval


def _mapping_catheter_data(signals_df: pd.DataFrame, feature_params: dict, ignore_channels: list[str] = None):
    if ignore_channels:
        signals_df = signals_df.drop(columns=ignore_channels, errors="ignore")
    features_by_channel = {}
    for _name, channels in pc.CHANNEL_GROUPS.items():
        if "mapping" != tz.get_in(["type"], cath.chGroup_to_catheter_attrs(_name)):
            continue

        for ch in channels.intersection(set(signals_df.columns)):
            features_by_channel[ch] = decorate_signal(
                signals_df[ch], feature_params=tz.assoc(feature_params, "wavelets", pc.DEFAULT_MAP_CATH_WAVELET_PARAMS)
            )

    return features_by_channel


def decorate_frame(signals: pd.DataFrame, feature_params: dict = None, opts: dict = {}) -> dict:
    """Decorate a frame by calculating the characteristics of signals contained within.  The resulting dictionary will look like:
     channel_to_features, qrs_intervals, disease_severity

    Args:
        signals_df (pd.DataFrame): signal data for all channels in the frame
        feature_params (dict, Optional): parameters for feature extraction. Valid keys are: deflections, ecg_wavelets, cs_wavelets, etc.
        opts (dict, Optional): options to control processing.  optional keys are 'bare_channels', 'qrs_intervals'

    Returns:
        dict: {'channel_to_features': {'ABLd': {'deflections': DataFrame, 'wavelets': DataFrame, 'features': dict}, ...},
               'qrs_intervals': list[Interval],
               'disease_of_intervals': DataFrame,
               'arrhythmia_severity': float,
               'cs_org_pairs': DataFrame,
               'cs_org_stats': DataFrame,
               'feature_vector': dict
        }
    """

    feature_params = feature_params or pc.DEFAULT_FEATURE_DETECTION_PARAMS
    PPS = pu.points_per_second(signals.index)

    if not opts.get("skip_qa_checks"):
        _frame_qa_checks(signals)

    # bare channels
    bare_channels = set(opts.get("bare_channels", []))
    bare_signals = {c: {"signal": signals[c].reset_index(drop=True)} for c in bare_channels if c in signals}

    # ecg
    ecg_to_features = {}
    for ecg in (set(signals.columns).intersection(pc.CHANNEL_GROUPS[pc.ChannelGroup.ECG])).difference(bare_channels):
        ecg_to_features[ecg] = decorate_signal(
            signals[ecg],
            {
                "deflections": {
                    "min_duration_ms": 10,
                    "separate_tails": True,
                    "split_by_jerk": True,
                },
                "butterworth_cutoff": feature_params.get("butterworth_cutoff", 30),
            },
        )

    # QRS zones
    qrs_intervals, qt_intervals = [], []
    if opts.get("qrs_intervals", True) and ecg_to_features:
        if feature_params.get("qrs"):
            qrs_intervals = qxi.qrs_intervals(feature_params.get("qrs"), PPS, ecg_to_features)
            qt_intervals = qxi.qt_intervals(qrs_intervals, feature_params.get("qrs"), PPS)
        else:
            log.warning("QRS measurements were not found in the meta file, QRS zones will not be calculated.")

    # CS characteristics
    cs_to_features = {}
    cs_wavelet_params = feature_params.get("cs_wavelets", pc.DEFAULT_CS_WAVELET_PARAMS)
    for cs in (set(signals.columns).intersection(pc.CHANNEL_GROUPS[pc.ChannelGroup.CS])).difference(bare_channels):
        cs_to_features[cs] = decorate_signal(
            signals[cs],
            feature_params={**feature_params, "wavelets": cs_wavelet_params},
            exo_signal={"qrs_intervals": qrs_intervals},
        )

    # HRA characteristics
    hra_to_features = {}
    hra_wavelet_params = feature_params.get("cs_wavelets", pc.DEFAULT_CS_WAVELET_PARAMS)
    for hra in (set(signals.columns).intersection(pc.CHANNEL_GROUPS[pc.ChannelGroup.HRA])).difference(bare_channels):
        hra_to_features[hra] = decorate_signal(
            signals[hra],
            feature_params={**feature_params, "wavelets": hra_wavelet_params},
            exo_signal={"qrs_intervals": qrs_intervals},
        )

    frame = {
        "index": signals.index,
        "channel_to_features": {
            **cs_to_features,
            **hra_to_features,
            **_mapping_catheter_data(signals, feature_params, bare_channels),
            **ecg_to_features,
            **bare_signals,
        },
        "qrs_intervals": qrs_intervals,
        "qt_intervals": qt_intervals,
        "meta": {"points_per_second": pu.points_per_second(signals.index)},
    }

    return frame
