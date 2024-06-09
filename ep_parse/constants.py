from collections import namedtuple
from enum import StrEnum
import toolz as tz

Deflection = namedtuple(
    "Deflection", "start_index end_index duration voltage_change start_voltage max_voltage slope slope_per_volts"
)
Interval = namedtuple("Interval", "start_index end_index")
Pwave = namedtuple(
    "Pwave",
    [
        "start_index",
        "end_index",
        "slope",
        "voltage_change",
        "max_voltage",
        "idxmax",
        "iso_prior",
        "max_slope_prior",
        "dist_to_max_prior",
        "vchange_prior",
        "duration",
        "AuC",
        "defl_count",
        "iso_after",
        "max_slope_after",
        "dist_to_max_after",
        "vchange_after",
        "sync_3",
        "sync_6",
        "sync_all",
        "IoU",
        "overlap_count",
        "channel_count",
        "iso_prior_3",
        "iso_prior_6",
    ],
)

TIME_FRMT = "%H:%M:%S"
TIME_WITH_MS_FRMT = "%H:%M:%S.%f"
DEFAULT_DATE = "1900-01-01"


class ChannelGroup(StrEnum):
    ABL = "ABL"
    ECG = "ECG"
    CS = "CS"
    HRA = "HRA"
    PENTARAY = "PENTARAY"
    HDGRID = "HDGRID"
    OCTARAY = "OCTARAY"
    LASSO = "LASSO"
    HALO = "HALO"

    # def __str__(self) -> str:
    #     return str.__str__(self)


CHANNEL_GROUPS = {
    ChannelGroup.ABL.value: ["ABLd", "ABLp", "ABLm"],
    ChannelGroup.ECG.value: ["I", "II", "III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    ChannelGroup.CS.value: ["CS_9-10", "CS_7-8", "CS_5-6", "CS_3-4", "CS_1-2"],
    ChannelGroup.HRA.value: [
        "HRA_19-20",
        "HRA_17-18",
        "HRA_15-16",
        "HRA_13-14",
        "HRA_11-12",
        "HRA_9-10",
        "HRA_7-8",
        "HRA_5-6",
        "HRA_3-4",
        "HRA_1-2",
    ],
    ChannelGroup.PENTARAY.value: [
        "PA_1-2",
        "PA_3-4",
        "PB_5-6",
        "PB_7-8",
        "PC_9-10",
        "PC_11-12",
        "PD_13-14",
        "PD_15-16",
        "PE_17-18",
        "PE_19-20",
    ],
    ChannelGroup.HDGRID.value: [
        "A1-A2",
        "A2-A3",
        "A3-A4",
        "B1-B2",
        "B2-B3",
        "B3-B4",
        "C1-C2",
        "C2-C3",
        "C3-C4",
        "D1-D2",
        "D2-D3",
        "D3-D4",
        "A1-B1",
        "A2-B2",
        "A3-B3",
        "B1-C1",
        "B2-C2",
        "B3-C3",
        "C1-D1",
        "C2-D2",
        "C3-D3",
    ],
    ChannelGroup.OCTARAY.value: [
        "OA_1-2",
        "OA_3-4",
        "OA_5-6",
        "OB_1-2",
        "OB_3-4",
        "OB_5-6",
        "OC_1-2",
        "OC_3-4",
        "OC_5-6",
        "OD_1-2",
        "OD_3-4",
        "OD_5-6",
        "OE_1-2",
        "OE_3-4",
        "OE_5-6",
        "OF_1-2",
        "OF_3-4",
        "OF_5-6",
        "OG_1-2",
        "OG_3-4",
        "OG_5-6",
        "OH_1-2",
        "OH_3-4",
        "OH_5-6",
    ],
    ChannelGroup.LASSO.value: [
        "LASSO_1-2",
        "LASSO_3-4",
        "LASSO_5-6",
        "LASSO_7-8",
        "LASSO_9-10",
        "LASSO_11-12",
        "LASSO_13-14",
        "LASSO_15-16",
        "LASSO_17-18",
        "LASSO_19-20",
    ],
    ChannelGroup.HALO.value: [
        "HALO_1-2",
        "HALO_3-4",
        "HALO_5-6",
        "HALO_7-8",
        "HALO_9-10",
        "HALO_11-12",
        "HALO_13-14",
        "HALO_15-16",
        "HALO_17-18",
        "HALO_19-20",
    ],
}

ALL_CHANNELS = list(tz.concat(CHANNEL_GROUPS.values()))

MAX_CYCLE_LENGTH = 800  # 800 ms, per SME
MIN_CYCLE_LENGTH = 80  # milliseconds SME
DEFAULT_CHANNELS = [
    "I",
    "II",
    "III",
    "aVL",
    "aVR",
    "aVF",
    "V1",
    "CS_1-2",
    "CS_3-4",
    "CS_5-6",
    "CS_7-8",
    "CS_9-10",
]

DEFAULT_ECG_CHANNELS = ["I", "II", "III", "aVL", "aVR", "aVF", "V1"]

DEFAULT_CS_WAVELET_PARAMS = {
    "local_volt_ratio": 0.35,
    "local_slope_ratio": 0.2,
    "max_iso_gap": 15,  # ms
    "max_non_prom_gap": 30,  # ms
    "filter_volt_ratio": 0.2,
    "filter_slope_ratio": 0.2,
    "filter_vchange_ratio": 0.2,
    "join_volt_ratio": 0.35,
    "wavelets_per_second": 2,
    "voltage_ceiling": 0.3,
}

DEFAULT_ECG_WAVELET_PARAMS = {
    "local_volt_ratio": 0.35,
    "local_slope_ratio": 0.15,
    "max_iso_gap": 12,  # ms
    "max_non_prom_gap": 20,  # ms
    "filter_volt_ratio": 0.4,
    "filter_slope_ratio": 0.4,
    "filter_vchange_ratio": 0.4,
    "join_volt_ratio": 0.9,
    "wavelets_per_second": 1,
    "voltage_ceiling": 0.3,
}

DEFAULT_ABL_WAVELET_PARAMS = {
    "local_volt_ratio": 0,
    "local_slope_ratio": 0,
    "max_iso_gap": 15,  # ms
    "max_non_prom_gap": 30,  # ms
    "filter_volt_ratio": 0,
    "filter_slope_ratio": 0,
    "filter_vchange_ratio": 0,
    "join_volt_ratio": 0,
    "wavelets_per_second": 6,
    "voltage_ceiling": 0.3,
}

DEFAULT_MAP_CATH_WAVELET_PARAMS = {
    "local_volt_ratio": 0.3,
    "local_slope_ratio": 0.1,
    "max_iso_gap": 10,  # ms
    "max_non_prom_gap": 20,  # ms
    "filter_volt_ratio": 0.25,
    "filter_slope_ratio": 0.1,
    "filter_vchange_ratio": 0.1,
    "join_volt_ratio": 0.2,
    "wavelets_per_second": 3,
    "voltage_ceiling": 0.3,
}

DEFAULT_DEFLECTION_PARAMS = {"min_v_change": 0.01, "min_slope": 0.001, "min_volts": 0.005}

DEFAULT_FEATURE_DETECTION_PARAMS = {
    "deflections": DEFAULT_DEFLECTION_PARAMS,
    "cs_wavelets": DEFAULT_CS_WAVELET_PARAMS,
}

DEFAULT_DISEASE_PARAMS = {
    "ABL_model": "resources/models/2023-08-25_ABLd_disease_scorer.joblib",
    "HDGRID_model": "resources/models/2023-08-25_HDgrid_disease_scorer.joblib",
    "PENTARAY_model": "resources/models/2023-08-25_Pentaray_disease_scorer.joblib",
    "dispersion_tolerance": 1,  # number of channels that can be active during dispersion iso-electric period
    "wavelets": DEFAULT_MAP_CATH_WAVELET_PARAMS,
}

DEFAULT_ARRHY_STATE_DETECTION_PARAMS = {
    "detector_sensitivity": 0,  # value in [-1,1]. 0 is default, towards 1 for more sensitivity and -1 for less
    "zone_threshold_ratio": 0.75,  # state change stability zones are split/defined using baseline_max * zone_threshold_ratio
    "min_duration_bl_max": 40,  # number of seconds that adjacent stability zones must last for a change to qualify as the max change
}

DEFAULT_PWAVE_PARAMS = {
    "min_lowV_prior_ms": 10,  # min iso duration that must precede pwave
    "min_lowV_after_ms": 10,  # min iso duration that must follow a pwave
    "max_pwave_duration_ms": 200,  # max pwave duration in milliseconds
    "lowV_slope_ratio": 0.7,  # slope ratio that defines low voltage (lowV or iso)
    "min_pwave_morph_score": 0.5,  # P-waves with a morphology-based confidence score less than this are not localizable (see p-wave hover hint for score)
    "min_pwave_overlap_score": 0.5,  # P-waves with an overlap-based confidence score less than this are not localizable (see p-wave hover hint for score)
    "min_filtered_pwaves_in_QT": 3,  # Ignore QT intervals that have fewer than this many p-waves in them
}


ARRHY_HL_FEATURES = [
    "cs_CL",
    "cs_CL_v2",
    "cs_CL-stab",
    "cs_CL-stab_v2",
    "cs_Disc",
    "cs_Disc_v2",
    "cs_Sync",
    "cs_Morph-stab",
    "hra_CL",
    "hra_CL_v2",
    "hra_CL-stab",
    "hra_CL-stab_v2",
    "hra_Disc",
    "hra_Disc_v2",
    "hra_Sync",
    "hra_Morph-stab",
]

HL_FEATURE_DISPLAY_BOUNDS = {"CL": [70, 300], "Disc": [0, 100], "Sync": [-120, 120]}

DISEASE_METRICS = {
    "dispersion": {"min": 0, "max": 1},
    "prominence_ratio": {"min": 0.2, "max": 1},
    "deflection_density": {"min": 0.025, "max": 0.225},
    "distance_ratio": {"min": 0.025, "max": 0.3},
    "lowV_ratio": {"min": 0.3, "max": 1},
    "lowVdelta_ratio": {"min": 0.3, "max": 1},
    "cycle_length_50%": {"min": 90, "max": 400, "invert": True},
    "model": {"min": 0, "max": 1},
    "signal_similarity": {"min": 0, "max": 1, "disease_score": False},
}
