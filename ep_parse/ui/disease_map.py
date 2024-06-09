import toolz as tz
import logging

import atrium.ingress.iresources as ir
import atrium.signal_nav as sn
import atrium.features.core as pfc
import atrium.utils as pu

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _hf_channels(tag: dict) -> dict:
    return [ch for ch, v in tag.get("channels", {}).items() if v.get("high_fidelity")]


def disease_data_for_tags(case_id: str, feature_params: dict):
    tags = ir.load_case_tags(case_id=case_id)
    mapping_channels = set(tz.concat([tag.get("channels", {}).keys() for tag in tags]))
    lbl_to_data = {}

    with ir.case_signals_db(case_id) as sig_store:
        for tag in tags:
            if "label" not in tag:
                continue
            _dscores = None
            if tag_channels := _hf_channels(tag):
                log.debug(f"Calculating disease scores for {tag['label']}")
                signals = sn.lookup_by_time(
                    signals_store=sig_store,
                    start_time=pu.as_time_str(tag["start_time"]),
                    end_time=pu.as_time_str(tag["end_time"]),
                    channels=mapping_channels,
                )
                tag_channels = signals.columns.intersection(set(tag_channels))
                frame = pfc.decorate_frame(
                    signals[tag_channels],
                    feature_params=feature_params,
                    opts={"qrs_intervals": False},
                )
                _dscores = {ch: df.max().fillna(-1).to_dict() for ch, df in frame["disease_scores"].items()}

            if _dscores:
                lbl_to_data[tag["label"]] = {"disease_scores": _dscores}

    return lbl_to_data
