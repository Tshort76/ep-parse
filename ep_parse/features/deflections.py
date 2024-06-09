import pandas as pd
import numpy as np
from typing import Union
import ep_parse.utils as pu
import ep_parse.constants as pc


# TODO make faster by stepping forward by X at a time, with X > 1, and then refining the zero point when -+ or +- is hit
# For example, step by 4 and if a change is detected across that span then reinspect that interval with a smaller step
def zeros_of(signal: np.ndarray) -> list[int]:
    return [i for i in range(1, len(signal)) if pu.sign(signal[i]) != pu.sign(signal[i - 1])]


def local_optima(signal: np.ndarray, of_derivative: int = 0) -> list[int]:
    # Return index for each each split point, using the local optima of the signal or its second derivative function
    dsignal = signal
    for _ in range(of_derivative + 1):
        dsignal = np.diff(dsignal)
    return [0] + zeros_of(dsignal) + [len(signal) - 1]


def merge_similar(signal, splits: list[int], min_defl_dur: int, threshold: float = 0.33) -> list[int]:
    if len(splits) < 3:
        return splits
    _new = [0]
    for i in range(1, len(splits) - 1):
        # TODO max(X, y) for the denominator of both slope_0 and slope_1
        d0 = splits[i] - _new[-1]
        d1 = splits[i + 1] - splits[i]
        slope_0 = 0 if d0 == 0 else (signal[splits[i]] - signal[_new[-1]]) / d0
        slope_1 = 0 if d1 == 0 else (signal[splits[i + 1]] - signal[splits[i]]) / d1
        denom = min(abs(slope_1), abs(slope_0))
        ratio = abs((slope_0 - slope_1) / max(0.0001, denom))
        if (splits[i] - _new[-1]) < min_defl_dur:
            slope_prev = 9e9 if len(_new) < 2 else (signal[_new[-1]] - signal[_new[-2]]) / (_new[-1] - _new[-2])
            if abs(slope_prev - slope_0) < abs(slope_1 - slope_0):
                _new.pop()
                _new.append(splits[i])
        elif ratio > threshold:
            _new.append(splits[i])
    return _new + [splits[-1]]


def _as_deflection(signal: np.ndarray, PPS: int, s_idx: int, e_idx: int) -> pc.Deflection:
    dur = pu.points_as_ms(e_idx - s_idx, PPS)
    s_volts = signal[s_idx]
    e_volts = signal[e_idx]
    v_delta = e_volts - s_volts
    slope = v_delta / max(1, dur)

    # Calculate max only at the endpoints, since these deflections are monotonic
    mx = max(abs(e_volts), abs(s_volts))
    slope_per_volts = abs(slope) / (0.00000001 + mx)  # add to the denominator to avoid division by 0
    return pc.Deflection(s_idx, e_idx, dur, v_delta, s_volts, mx, slope, slope_per_volts)


def subdivide_deflections(signal: np.ndarray, deflections: list[pc.Deflection], min_defl_dur: int):
    _new = []
    slope_80 = np.quantile([abs(d.slope) for d in deflections], 0.825)
    for d in deflections:
        if d.end_index - d.start_index >= min_defl_dur and abs(d.slope) < slope_80:
            indices = local_optima(signal[d.start_index : d.end_index], 2)
            if len(indices) > 2:
                # add start_index to split indices
                _new += [i + d.start_index for i in indices[0:-1]]
                continue
        _new.append(d.start_index)
    return _new + [len(signal) - 1]


def _tail_status(defl: pc.Deflection, signal: np.ndarray, threshold: int = 5, win_size=20) -> str:
    """Return a string describing the (flat) tail status of the deflection

    Args:
        defl (pc.Deflection): The deflection candidate
        signal (np.ndarray): The signal associated with frame containing the deflection
        threshold (int, optional): The threshold for the voltage change ratio across the candidate splits. Defaults to 5.
        win_size (int, optional): Window size of the candidates when splitting. Defaults to 20.

    Returns:
        str: String containing "L" if tail on left and/or "R" if tail on right
    """
    if abs(defl.voltage_change) < 0.075 or defl.duration <= win_size:
        return ""
    fsig = signal[defl.start_index : defl.end_index]
    # check if change over last L samples ~= change over first L samples
    s_start = fsig[win_size] - fsig[0]
    s_end = fsig[len(fsig) - 1] - fsig[len(fsig) - (win_size + 1)]
    # TODO what if ratio was (s - e) / max(s,e) instead of s/e ??
    r = abs(s_start / s_end)
    tail_left, tail_right = r < 1 / threshold, r > threshold
    # print(f"parabolic score: {r}, needs to be < {1/threshold} or > {threshold}")
    for a in [1, 2, 3]:
        i = int((a * len(fsig) / 4) - (win_size / 2))
        if i + win_size < 0 or i + win_size >= len(fsig):
            continue
        d = fsig[i + win_size] - fsig[i]
        # print(f"hyp score: {abs(d / s_start)}, needs to be > {threshold}")
        tail_left = tail_left or abs(d / s_start) > threshold
        tail_right = tail_right or abs(d / s_end) > threshold

    tail_codes = "R" if tail_right else ""
    tail_codes += "L" if tail_left else ""
    return tail_codes


def deflections_with_tails(
    deflections: list[pc.Deflection], signal: np.ndarray, tail_threshold: int = 5
) -> list[tuple[int, str]]:
    """Identify deflections that have a flat tail on either end and describe that tail, "L"(eft), "R"(ight), or "LR".

    Args:
        deflections (list[pc.Deflection]): deflections to analyze
        signal (np.ndarray): signal associated with deflections
        tail_threshold (float): a tail has a slope that is 1/tail_threshold of some other segment of the deflection signal

    Returns:
        list[tuple[int,str]]: list of (start_index, tail_description) tuples
    """
    to_split = []
    for d in deflections:
        if tail_code := _tail_status(d, signal, threshold=tail_threshold):
            to_split.append((d, tail_code))
    return to_split


def _split_at(sig: np.ndarray, win_size: int, right_to_left: bool = True) -> int:
    idx0, idxN, step = 0, len(sig) - 1, int(win_size / 4)
    if right_to_left:
        idx0, idxN, win_size, step = len(sig) - 1, 0, win_size * -1, step * -1
    best_split = (0, 0)
    for i in range(idx0, idxN, step):
        j, k = i + win_size, i + (2 * win_size)
        if k >= len(sig) or k < 0:
            break
        distal = sig[i] - sig[j]  # outside segment
        proximal = sig[j] - sig[k]  # segment closer to deflection center
        ratio = abs(proximal / distal)
        # print(j, ratio)
        if ratio > best_split[1]:
            best_split = (j, ratio)
    return best_split[0]


def _decompose_deflection(
    deflection: pc.Deflection, signal: np.ndarray, tail_types: str, min_tail_dur: int
) -> list[pc.Deflection]:
    s, e = deflection.start_index, deflection.end_index
    sig_seg = signal[s:e]
    win_size = 8 if len(sig_seg) <= 120 else 16 if len(sig_seg) <= 240 else 32
    splits = [s]
    if "L" in tail_types:
        best_split_idx = _split_at(sig_seg, win_size, False)
        if best_split_idx > min_tail_dur:
            splits.append(best_split_idx + s)  # adjust to global index
    if "R" in tail_types:
        best_split_idx = _split_at(sig_seg, win_size, True)
        if (e - (best_split_idx + s)) > min_tail_dur:
            splits.append(best_split_idx + s)
    return splits + [e]


def with_tails_separated(deflections: list[pc.Deflection], signal: np.ndarray, pps: int) -> list[pc.Deflection]:
    to_split = deflections_with_tails(deflections, signal)
    min_tail_dur = pu.ms_as_points(10, pps)
    split_lookup = {
        d.start_index: _decompose_deflection(d, signal, tail_types, min_tail_dur) for d, tail_types in to_split
    }
    new_deflections = []
    for d in deflections:
        splits = split_lookup.get(d.start_index)
        if splits and len(splits) > 2:
            for i in range(1, len(splits)):
                new_deflections.append(_as_deflection(signal, pps, splits[i - 1], splits[i]))
        else:
            new_deflections.append(d)
    return new_deflections


def extract_deflections(
    signal: Union[pd.Series, np.ndarray],
    PPS: int,
    min_volts: float = 0,
    min_v_change: float = 0,
    min_slope: float = 0,
    min_duration_ms: int = 3,
    separate_tails: bool = False,
    split_by_jerk: bool = False,
):
    if isinstance(signal, pd.Series):
        signal = signal.to_numpy()
    splits = local_optima(signal, of_derivative=0)
    deflections = [_as_deflection(signal, PPS, splits[i - 1], splits[i]) for i in range(1, len(splits))]
    if separate_tails:
        deflections = with_tails_separated(deflections, signal, PPS)
    if split_by_jerk:
        min_duration = pu.ms_as_points(min_duration_ms, PPS)
        splits = subdivide_deflections(signal, deflections, min_duration)
        splits = merge_similar(signal, splits, min_duration)
        deflections = [_as_deflection(signal, PPS, splits[i - 1], splits[i]) for i in range(1, len(splits))]

    deflections = [
        d
        for d in deflections
        if abs(d.voltage_change) >= min_v_change and d.max_voltage >= min_volts and abs(d.slope) >= min_slope
        # and d.duration >= min_duration_ms
    ]
    return deflections
