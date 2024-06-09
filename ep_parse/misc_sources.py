import re
import os
from datetime import timedelta
import logging

import ep_parse.utils as pu
import ep_parse.case_data as cdata
import ep_parse.common as pic

log = logging.getLogger(__name__)

##############################   Medtronic Catheter Log files
MD_LOG_FILE = re.compile(r".+\d{2}\.log")


def parse_RF_times(filepath: str, offset: timedelta) -> tuple[str, str]:
    times = []
    with open(filepath, "r") as fp:
        for line in fp.readlines():
            tokens = line.split()
            if tokens and pu._time_only.match(tokens[0]):
                _time = pu.as_datetime(tokens[0]) + offset
                times.append(pu.as_time_str(_time, ms_precision=False))

    if len(times) < 2:
        log.warning(f"Insufficient time covered by {filepath}, skipping file")
        return

    return times[0], times[-1]


def store_md_cath_offset(case_id: str, epmed_time: str, md_cath_time: str, force: bool = False) -> None:
    if not force:
        if offset := cdata.load_case_meta(case_id).get("md_catheter_time_offset"):
            log.error(
                f"An MD catheter offset of {offset} already exists for this case.  If that value is correct, ignore this warning.  Otherwise, rerun with FORCE = True"
            )
            return
    offset = (pu.as_datetime(epmed_time) - pu.as_datetime(md_cath_time)).total_seconds()
    cdata.update_case_meta(case_id, ["md_catheter_time_offset"], offset)
    return offset


def md_catheter_logs_as_bookmark_file(case_id: str) -> None:
    export_dir = cdata.case_file_path(case_id, pic.DataSource.MD_CATH)
    assert export_dir, f"Expected a md_cathether_logs subdirectory in the case export data directory"
    offset = cdata.load_case_meta(case_id).get("md_catheter_time_offset")
    assert offset, "No MD catheter offset has been stored for this case!"
    offset = timedelta(seconds=offset)

    times = []
    for filename in os.listdir(export_dir):
        if MD_LOG_FILE.match(filename):
            if rf_times := parse_RF_times(os.path.join(export_dir, filename), offset):
                times.append(rf_times)

    rf_data = ""
    for i, (stime, etime) in enumerate(sorted(times, key=lambda x: x[0])):
        rf_data += f" {stime} RF On - Session {i+1} {etime} RF Off - 10 s " + ("\n" if i % 5 == 4 else "")

    fpath = os.path.join(cdata.case_data_directory(case_id), f"{case_id}_bookmarks.txt")
    with open(fpath, "a") as fp:
        fp.write("\n" + rf_data)
