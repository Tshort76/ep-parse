import pandas as pd
import numpy as np
import ep_parse.utils as pu
from ep_parse.constants import Deflection, Interval
import logging
from operator import add, attrgetter
import ep_parse.features.deflections as dfl
import toolz as tz

log = logging.getLogger(__name__)


def remove_deflections_in_qrs(deflections: list[Deflection], zones: list[Interval]) -> pd.DataFrame:
    if not deflections:
        return []

    _iter = iter(deflections)
    d = next(_iter)
    new_defl = []
    for s, e in zones:
        while d and d.start_index < s:  # add all deflections before the current zone
            new_defl.append(d)
            d = next(_iter, None)
        while d and d.start_index < e:  # skip over deflections inside of the current zone
            d = next(_iter, None)

    while d:  # add remaining deflections
        new_defl.append(d)
        d = next(_iter, None)

    return new_defl


def _max_deflection_per_qrs(deflections: list[Deflection], zones: list[Interval]) -> list[Deflection]:
    if not deflections:
        return []

    rvals, d_idx = [], 0
    dummy = Deflection(0, 0, 0, 0, 0, 0, 0, 0)
    for s, e in zones:
        s, e = s - 15, e + 15  # slightly pad intervals since R peak won't be aligned across channels
        mx = dummy
        while d_idx < len(deflections) and deflections[d_idx].end_index < s:
            d_idx += 1
        while d_idx < len(deflections) and deflections[d_idx].start_index < e:
            mx = mx if mx.max_voltage > deflections[d_idx].max_voltage else deflections[d_idx]
            d_idx += 1
        if mx.max_voltage > 0:  # dont add dummy record
            rvals.append(mx)
    return rvals


def _intervals_by_channel(
    ecg_to_features: dict[str, dict], qrs_meta: dict, pps: int, course_zones: list[Interval]
) -> list[Interval]:
    QRS_candidates = {}
    for channel, fdict in ecg_to_features.items():
        ch_meta = qrs_meta.get(channel, {})
        if ch_meta:
            qr, rs = pu.ms_as_points(ch_meta["QR"], pps), pu.ms_as_points(ch_meta["RS"], pps)
            candidates = [d for d in fdict.get("deflections", []) if d.max_voltage == abs(d.start_voltage)]
            # top voltage within each qrs_zone
            ch_peaks = _max_deflection_per_qrs(candidates, course_zones)  # qrs zones calculated from signals sum vector
            QRS_candidates[channel] = [Interval(p.start_index - qr, p.start_index + rs) for p in ch_peaks]
    return QRS_candidates


def _course_qrs_intervals(sum_sig: pd.Series, pps: int, qrs_meta: dict):
    deflections = dfl.extract_deflections(sum_sig, pps)

    # Sort candidates by voltage, look for when there is big drop
    candidates = [d for d in deflections if d.max_voltage == abs(d.start_voltage)]
    sorted_candidates = sorted(candidates, key=attrgetter("max_voltage"), reverse=True)
    n = int(pu.points_as_ms(len(sum_sig), pps) / 1000)  # 1 peak per second ... very conservative
    if len(candidates) <= n:
        return []
    min_voltage = sorted_candidates[n].max_voltage * 0.7
    peaks = [s for s in sorted_candidates if s.max_voltage > min_voltage]

    # remove peaks that are too close to each other, taking the largest of each proximal group
    ST = pu.ms_as_points(qrs_meta["ST_duration_ms"], pps)
    sorted_peaks = sorted(peaks, key=attrgetter("start_index"))
    qrs_defls = [sorted_peaks[0]]
    for d in sorted_peaks[1:]:
        if d.start_index - qrs_defls[-1].start_index < ST:  # Ignore T-waves
            if d.max_voltage > qrs_defls[-1].max_voltage:  # this one bigger than previous one
                qrs_defls = qrs_defls[:-1] + [d]  # replace previous with current
            continue  # dont add the current one, it is smaller
        qrs_defls.append(d)

    qrs_peaks = sorted([p.start_index for p in qrs_defls])

    # expand peaks to zones using QR and RS measurements
    qr_durations = [v["QR"] for v in qrs_meta.values() if isinstance(v, dict)]
    rs_durations = [v["RS"] for v in qrs_meta.values() if isinstance(v, dict)]
    qr, rs = int(np.median(qr_durations)), int(np.median(rs_durations))

    return [Interval(x - qr, x + rs) for x in qrs_peaks]


def qrs_intervals(qrs_meta: dict, PPS: int, ecg_to_features: dict[str, dict]) -> list[Interval]:
    signals = [e["signal"].abs() for e in ecg_to_features.values() if "signal" in e]
    if not signals:
        return []
    ecg_sum_signal = tz.reduce(add, signals)
    course_zones = _course_qrs_intervals(ecg_sum_signal, PPS, qrs_meta)
    # Check for too much chaos in ECG signals
    if len(course_zones) > (
        pu.points_as_ms(len(ecg_sum_signal), PPS) / (qrs_meta["QS_duration_ms"] + qrs_meta["ST_duration_ms"])
    ):
        log.warning("ECG signals are too chaotic for QRS zone detection")
        return []
    ecg_to_intervals = _intervals_by_channel(ecg_to_features, qrs_meta, PPS, course_zones)
    min_channels = max(3, int(len(ecg_to_intervals) * 0.667))
    return pu.intervals_of_activity(min_channels, ecg_to_intervals, vector_len=len(ecg_sum_signal))


def qt_intervals(qrs_intervals: list[Interval], qrs_meta: dict, PPS: int) -> list[Interval]:
    st_duration = pu.ms_as_points(qrs_meta["ST_duration_ms"], PPS)
    return [Interval(q.start_index, q.end_index + st_duration) for q in qrs_intervals]
