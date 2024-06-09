import pandas as pd
import numpy as np
import ep_parse.utils as pu
import ep_parse.constants as pc


def _centroid(sig: np.ndarray, start_idx: float):
    abs_sig = np.abs(sig)
    mask = abs_sig > 0.5 * np.max(abs_sig)
    p = abs_sig[mask]
    return int(np.sum((p * mask.nonzero()) / np.sum(p)) + start_idx)


def _attr_list(defl_list: list[pc.Deflection], attr: str) -> list:
    i = pc.Deflection._fields.index(attr)
    return [d[i] for d in defl_list]


def core_signal_features(raw_signal: pd.Series, PPS: int, deflects: list[pc.Deflection]) -> dict:
    LOW_V = 0.05
    PERCENTILES = [0.25, 0.5, 0.75, 0.95]
    sig_len_ms = pu.points_as_ms(len(raw_signal), PPS)
    sig_numpy = raw_signal.to_numpy() if isinstance(raw_signal, pd.Series) else raw_signal
    vchanges = [d.voltage_change for d in deflects]
    durations = np.array([d.duration for d in deflects])
    f = {}

    qz = np.quantile(np.abs(sig_numpy), PERCENTILES)
    for p, q in zip(PERCENTILES, qz):
        f[f"volt_{p:.0%}"] = q
    f["volt_iqr"] = qz[2] - qz[0]

    colz = ["voltage_change", "slope", "slope_per_volts"]
    if len(deflects) == 0:
        for c in colz:
            for p in PERCENTILES:
                f[f"{c}_{p:.0%}"] = 0
    else:
        for c in colz:
            qz = np.quantile(np.abs(_attr_list(deflects, c)), PERCENTILES)
            for p, v in zip(PERCENTILES, qz):
                f[f"{c}_{p:.0%}"] = v
            f[f"{c}_iqr"] = qz[2] - qz[0]

    if len(deflects) <= 1:
        for c in colz:
            f[f"{c}_iqr"] = 0

    defl_dur = max(1, np.sum(durations))

    f["num_deflections"] = len(deflects)
    f["deflection_density"] = len(deflects) / sig_len_ms
    f["deflection_density_no_iso"] = len(deflects) / defl_dur

    v_dists = np.abs(np.diff(sig_numpy))
    v_dist = np.sum(v_dists)
    f["volt_distance"] = v_dist

    # Voltage dispersion
    if v_dist:
        inc = int(len(sig_numpy) / 4)
        for i in range(1, 4):
            f[f"vdist_{25*i:d}%"] = np.sum(v_dists[: i * inc]) / v_dist

    # voltage distance normalized by wavelet length
    f["volt_distance_ratio"] = v_dist / sig_len_ms

    # % of signal comprised of deflections
    f["defl_iso_ratio"] = defl_dur / sig_len_ms

    # % of deflections which have low voltage changes
    mask = np.abs(vchanges) < LOW_V
    f["lowVdelta_ratio"] = np.sum(durations * mask) / defl_dur

    # % of deflections which are low voltage
    mask = np.array([d.max_voltage for d in deflects]) < LOW_V
    f["lowV_ratio"] = np.sum(durations * mask) / defl_dur

    f["centroid_idx"] = _centroid(sig_numpy, deflects[0].start_index) if len(deflects) else np.nan

    # Start Ryan's requested characteristics ... of questionable value for ML but good for manual analysis
    f["juicy_new1"] = f["volt_distance_ratio"] / (f["volt_50%"] ** 2)
    f["juicy_new2"] = f["juicy_new1"] * len(deflects)
    f["juicy_new1_a"] = 0.6 * np.sum(1 / np.abs(vchanges)) / defl_dur

    # voltage distance normalized by wavelet length and max voltage (?)
    f["distance_ratio"] = v_dist / (f["volt_95%"] * sig_len_ms)

    return f


def as_pairs(seq: pd.Series):
    return [[seq.iloc[i - 1], seq.iloc[i]] for i in range(1, len(seq))]


# less efficient than pandas operations (I presume), but easier to audit calculations visually
def _append_pairs(waves: pd.DataFrame, pairs: dict) -> dict:
    if waves.empty:
        return
    pairs["wave_idx"] += waves.index[:-1].to_list()
    pairs["cl"] += as_pairs(waves["centroid_idx"])
    pairs["iso"] += zip(waves["end_index"], waves["start_index"].shift(-1)[:-1])
    pairs["iso+wave"] += as_pairs(waves["start_index"])


def wavelet_dispersal_features(
    wavelets: pd.DataFrame, PPS: int, qrs_intervals: list[pc.Interval], output_as: str = "stats"
) -> dict:
    assert output_as in {
        "stats",
        "distances",
        "pairs",
    }, f"Invalid output_as argument.  Must be one of 'stats', 'distances', 'pairs'"
    metrics = ["cl", "iso", "iso+wave"]
    pairs = {k: [] for k in (metrics + ["wave_idx"])}
    s = 0
    zones = [] if qrs_intervals is None else qrs_intervals
    for zone in zones:
        waves = wavelets[(wavelets["start_index"] > s) & (wavelets["start_index"] < zone[0])]
        s = zone[1]
        _append_pairs(waves, pairs)
    _append_pairs(wavelets[(wavelets["start_index"] > s)], pairs)

    if output_as == "pairs":
        return pairs

    # convert to milliseconds
    dists = {k: [pu.points_as_ms(x[1] - x[0], PPS) for x in pairs[k]] for k in metrics}
    if output_as == "distances":
        return dists

    if len(pairs["cl"]) == 0:
        return {f: np.nan for f in ["cycle_length_50%", "cycle_length_iqr", "iso_duration_50%", "iso_duration_iqr"]}

    cl_q25, cl_q50, cl_q75 = np.quantile(dists["cl"], [0.25, 0.5, 0.75])
    iso_q25, iso_q50, iso_q75 = np.quantile(dists["iso"], [0.25, 0.5, 0.75])

    return {
        "cycle_length_50%": cl_q50,
        "cycle_length_iqr": cl_q75 - cl_q25,
        "iso_duration_50%": iso_q50,
        "iso_duration_iqr": iso_q75 - iso_q25,
    }


def interwavelet_features(
    wavelets: pd.DataFrame, PPS: int, qrs_intervals: list[pc.Interval], signal_len_ms: float
) -> dict:
    if wavelets.empty:
        return {}

    medians_features = {"duration", "num_deflections", "volt_50%"}
    nstd_features = set(
        wavelets.drop(columns=["idxmax", "centroid_idx", "start_index", "end_index"], errors="ignore").columns.to_list()
    )

    f = {}
    for col in medians_features.union(nstd_features):
        wave_attr = wavelets[col]
        med = wave_attr.median() if len(wavelets) > 1 else 0
        if col in medians_features:
            f["wavelet_" + col + "_50%"] = med
        if col in nstd_features:
            f["wavelet_" + col + "_nstd"] = (wave_attr.std(ddof=1) / (med or 1)) if len(wavelets) > 1 else 0

    f["prominence_ratio"] = wavelets["duration"].sum() / signal_len_ms

    return {**f, **wavelet_dispersal_features(wavelets, PPS, qrs_intervals, output_as="stats")}


def as_feature_dict(
    raw_signal: pd.Series, PPS: int, deflects: list[pc.Deflection], wavelets: pd.DataFrame, exo_signal: dict = {}
) -> dict:
    core_features = core_signal_features(raw_signal, PPS, deflects)
    wv_meta = interwavelet_features(
        wavelets, PPS, exo_signal.get("qrs_intervals"), signal_len_ms=pu.points_as_ms(len(raw_signal), PPS)
    )
    return {**core_features, **wv_meta}
