import pandas as pd
import ep_parse.features.signal as pfs
import logging
import numpy as np
from collections import namedtuple

import ep_parse.utils as pu
from ep_parse.constants import MIN_CYCLE_LENGTH, DEFAULT_CS_WAVELET_PARAMS, Deflection

log = logging.getLogger(__name__)
Zone = namedtuple("Zone", "start_index end_index duration max_voltage idxmax max_slope max_vchange num_deflections")

# DEBUG = True
DEBUG = False


def _max_v(deflects: list):
    mx = [0, 0]
    for d in deflects:
        if d.max_voltage > mx[0]:
            mx = [d.max_voltage, d.end_index]
    return mx


def _max(deflections: list, attr: str):
    i = Deflection._fields.index(attr)
    return max([abs(d[i]) for d in deflections]) if deflections else 0.0


def _as_zones(zone_deflections: list[Deflection], PPS: int) -> list[Zone]:
    rvals = []
    for deflections in zone_deflections:
        if len(deflections) > 1:
            s = deflections[0].start_index
            e = deflections[-1].end_index
            dur = pu.points_as_ms(e - s, PPS)
            max_volts, peak_idx = _max_v(deflections)
            max_slope = _max(deflections, "slope")
            max_vchange = _max(deflections, "voltage_change")
            rvals.append(Zone(s, e, dur, max_volts, peak_idx, max_slope, max_vchange, len(deflections)))
    return rvals


def _join_wavelets(zones: list[Zone], PPS: int) -> Zone:
    s = min([z.start_index for z in zones])
    e = max([z.end_index for z in zones])
    mx_idx = np.nanargmax([z.max_voltage for z in zones])
    return Zone(
        s,
        e,
        pu.points_as_ms(e - s, PPS),
        zones[mx_idx].max_voltage,
        zones[mx_idx].idxmax,
        max([z.max_slope for z in zones]),
        max([z.max_vchange for z in zones]),
        sum([z.num_deflections for z in zones]),
    )


def _local_Vfloor(local_max: float, configs: dict):
    local_min_ratio, voltage_ceiling = [configs[k] for k in ["local_volt_ratio", "voltage_ceiling"]]
    return local_min_ratio * min(voltage_ceiling, local_max)


def _split_zone(deflections: list[Deflection], configs: dict) -> list:
    local_Vmax, _ = _max_v(deflections)
    volt_floor = _local_Vfloor(local_Vmax, configs)
    slope_floor = _max(deflections, "slope") * configs["local_slope_ratio"]
    return [d for d in deflections if d.max_voltage > volt_floor and abs(d.slope) > slope_floor]


def _of_interest(defl: Deflection, zone: list[Zone], global_v_floor: float, configs: dict):
    local_max_v = max(map(lambda x: x.max_voltage, zone)) if zone else 0
    volt_floor = max(global_v_floor, _local_Vfloor(local_max_v, configs))
    slope_floor = _max(zone, "slope") * configs["local_slope_ratio"]
    if DEBUG:
        print(f"Of interest: {defl.start_index, defl.max_voltage, defl.slope} \n {volt_floor} {slope_floor}")
    return defl.max_voltage >= volt_floor and abs(defl.slope) > slope_floor


def _zone_it(deflections: list[Deflection], PPS: int, min_defl_voltage: float, configs: dict) -> list[Zone]:
    max_non_prom_gap, max_iso_gap = [configs[k] for k in ["max_non_prom_gap", "max_iso_gap"]]
    zones = []
    _zone = []

    s = 0
    for d in deflections:
        # Check to see if the zone should be split and a new one created.  Note that (iso electric) segments are implicit (i.e. no deflections for an interval)
        if (d.start_index - s > max_iso_gap) or (_zone and (d.start_index - _zone[-1].end_index) > max_non_prom_gap):
            if _zone:
                zones.append(_split_zone(_zone, configs))
                _zone = []

        if _of_interest(d, _zone, min_defl_voltage, configs):
            if DEBUG:
                print(f"Found zone of interest: {d.start_index}")
            _zone.append(d)

        s = d.end_index

    if _zone:
        zone = _split_zone(_zone, configs)
        zones.append(zone)

    if DEBUG:
        print(f"Found {len(zones)} zones")
    return _as_zones(zones, PPS)


def _remove_trivial_zones(zones: list[Zone], frame_len_ms: int, configs: dict) -> list[Zone]:
    n = int(frame_len_ms * (configs["wavelets_per_second"] / 1000))

    if n > len(zones):
        return zones

    # normalize our thresholds using the median value of the top N values of a feature
    x = np.array([z.max_voltage for z in zones])
    volt_floor = min(np.median(x[np.argsort(x)[-n:]]), configs["voltage_ceiling"]) * configs["filter_volt_ratio"]

    x = np.array([z.max_slope for z in zones])
    slope_floor = min(np.median(x[np.argsort(x)[-n:]]), 0.01) * configs["filter_slope_ratio"]

    x = np.array([z.max_vchange for z in zones])
    vchange_floor = (
        min(np.median(x[np.argsort(x)[-n:]]), configs["voltage_ceiling"] * 2) * configs["filter_vchange_ratio"]
    )

    return [
        z
        for z in zones
        if (z.max_voltage >= volt_floor)
        and (z.max_vchange >= vchange_floor)
        and (z.max_slope >= slope_floor)
        and (z.num_deflections > 1)
    ]


def _merge_zones(zones: list[Zone], PPS: int, configs: dict) -> list[Zone]:
    zone_stack, max_pt_diff = [], pu.ms_as_points(MIN_CYCLE_LENGTH, PPS)
    for zone in zones:
        if zone_stack:
            prev = zone_stack[-1]
            if abs(zone.idxmax - prev.idxmax) < max_pt_diff:
                if (min(zone.max_voltage, prev.max_voltage) / max(zone.max_voltage, prev.max_voltage)) > configs[
                    "join_volt_ratio"
                ]:
                    zone = _join_wavelets([zone_stack.pop(), zone], PPS)
                else:
                    # wavelets are too close, remove the smaller of the two
                    prev_zone = zone_stack.pop()
                    if prev.max_voltage > zone.max_voltage:
                        zone = prev_zone
        zone_stack.append(zone)

    return zone_stack


def _as_wavelet(raw_signal: np.ndarray, PPS: int, deflects: list[Deflection], interval: tuple) -> dict:
    f = pfs.core_signal_features(raw_signal, PPS, deflects)
    f["start_index"] = interval[0]
    f["end_index"] = interval[1]
    f["duration"] = pu.points_as_ms(len(raw_signal), PPS)
    f["idxmax"] = np.nanargmax(np.abs(raw_signal))
    return f


def _as_wavelets(signal: np.ndarray, PPS: int, deflections: list[Deflection], raw_zones: list[Zone]) -> list[dict]:
    wavelets = []
    for zone in raw_zones:
        start, end = zone.start_index, zone.end_index
        raw_sig = signal[start:end]
        defls = [d for d in deflections if d.start_index >= start and d.start_index < end]
        if len(raw_sig) > 0:
            wavelets.append(_as_wavelet(raw_sig, PPS, defls, (start, end)))
    return wavelets


def partition_into_wavelets(signal: np.ndarray, PPS: int, deflections: list[Deflection], user_configs: dict = {}):
    if not deflections:
        return pd.DataFrame()
    configs = pu.coerce_wavelet_configs(DEFAULT_CS_WAVELET_PARAMS, PPS, user_configs)
    prom_defl_floor = user_configs.get("prominent_deflection_floor")
    if prom_defl_floor is None:
        defl_iso_ratio = np.sum([d.duration for d in deflections]) / pu.points_as_ms(len(signal), PPS)
        coeffs = [-0.19439403, 0.07053378, 0.44519571, -0.02154149]
        Q = min(0.33, max(0.02, sum([coeffs[i] * (defl_iso_ratio ** (3 - i)) for i in range(len(coeffs))])))
        prom_defl_floor = np.quantile([d.max_voltage for d in deflections], Q)
    zones = _zone_it(deflections, PPS, prom_defl_floor, configs)
    if zones:
        zones = _remove_trivial_zones(zones, pu.points_as_ms(len(signal), PPS), configs)
        zones = _merge_zones(zones, PPS, configs)
        zones = _as_wavelets(signal, PPS, deflections, zones)
    return pd.DataFrame(zones)
