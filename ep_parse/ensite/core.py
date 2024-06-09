import os
import logging
from datetime import datetime, timedelta
import toolz as tz
from operator import itemgetter

import ep_parse.ensite.geometries as geo
import ep_parse.ensite.lesions as el
import ep_parse.ensite.map_points as emp
import ep_parse.utils as pu
import ep_parse.constants as pc
import ep_parse.case_data as cdata
import ep_parse.common as pic
import ep_parse.catheters as cath


log = logging.getLogger(__name__)
_ = log.setLevel(logging.DEBUG) if pu.is_dev_mode() else log.setLevel(logging.ERROR)


def _sync_tag_times(tags: list[dict], offset: timedelta) -> None:
    "Update **in place** using an offset (ENsite_time + offset = EPmed_time)"
    for tag in tags:
        tag["start_time"] = pu.as_time_str(pu.as_datetime(tag["start_time"]) + offset)
        tag["end_time"] = pu.as_time_str(pu.as_datetime(tag["end_time"]) + offset)
        tag["time_synced_at"] = datetime.now().isoformat()


def store_ensite_offset(case_id: str, epmed_time: str, ensite_time: str, force: bool = False) -> float:
    if not force:
        if offset := cdata.load_case_meta(case_id).get("ensite_time_offset"):
            log.error(
                f"An ensite offset of {offset} already exists for this case!  If that value is correct, ignore this warning.  Otherwise, rerun with FORCE = True"
            )
            return
    offset = (pu.as_datetime(epmed_time) - pu.as_datetime(ensite_time)).total_seconds()
    cdata.update_case_meta(case_id, ["ensite_time_offset"], offset)
    return offset


def infer_ensite_offset(case_id: str, event_loader_fn) -> float:
    export_dir = cdata.case_file_path(case_id, pic.DataSource.ENSITE_VELOCITY)
    rf_log = event_loader_fn(case_id)
    if rf_times := el.rf_start_times(export_dir, rf_log):
        return store_ensite_offset(case_id, rf_times[0], rf_times[1], True)


# Lesions and Geo files are replicated in each subfolder
def import_ensite_export(case_id: str, parse_types: list[str], event_loader_fn) -> None:
    export_dir = cdata.case_file_path(case_id, pic.DataSource.ENSITE_VELOCITY)
    assert export_dir, f"No ensite_velocity folder found for {case_id}"
    epmed_offset = cdata.load_case_meta(case_id).get("ensite_time_offset")
    assert epmed_offset is not None, "No ensite time offset has been stored for this case!"
    epmed_offset = timedelta(seconds=epmed_offset)

    if "geometry" in parse_types:
        two_volume_file = os.path.join(export_dir, "ModelGroups.xml")
        single_volume_file = os.path.join(export_dir, "DxLandmarkGeo.xml")
        model_file = two_volume_file if os.path.exists(two_volume_file) else single_volume_file
        vtk_files = geo.geoXML_as_vtk(case_id, model_file)
        log.debug(f"Parsed {model_file} and wrote vtk files: {vtk_files}")

    rf_tags = []
    if "RF" in parse_types:
        rf_tags = el.parse_RF_tags(export_dir, event_loader_fn(case_id), epmed_offset)
        geo.adjust_geo_meta(case_id, rf_tags, adjust_points=True)  # assign to atria file, snap to surface
        cdata.write_case_tags(case_id, rf_tags, mode="r")

    map_tags = []
    if "MAP" in parse_types:
        subdirs = [s for s in os.listdir(export_dir) if not s.startswith(".")]  # ignore hidden directories
        assert (
            subdirs
        ), "Directory structure incorrect, expected ensite_velocity parent with subdirectories for each Map"

        for subdir in subdirs:
            if os.path.isfile(os.path.join(export_dir, subdir)):
                continue
            map_id = emp.parse_map_id(subdir, case_id)
            raw_grid_points = emp.parse_grid_points(os.path.join(export_dir, subdir))
            raw_grid_points = emp.with_standard_channel_names(raw_grid_points, print_mapping=True)

            for group, channels in pc.CHANNEL_GROUPS.items():
                catheter = cath.chGroup_to_catheter(group)
                if not cath.is_mapping_cath(catheter):
                    continue
                if group_channels := raw_grid_points.columns.intersection(channels).to_list():
                    group_df = raw_grid_points[group_channels]
                    if len(group_df) > 5:
                        log.debug(f"Parsed {len(group_df)} {catheter} locations from {subdir} DxL files")
                        cath_dir_tags = emp.group_into_tags(group_df, catheter, map_id=map_id)
                        map_tags += cath_dir_tags
                        log.debug(f"Created {len(cath_dir_tags)} {catheter} MAP tags for map {map_id}")

        # add label to MAP tags according to chronological order
        map_tags = [
            tz.assoc(d, "label", f"MAP {i+1}") for i, d in enumerate(sorted(map_tags, key=itemgetter("start_time")))
        ]
        geo.adjust_geo_meta(case_id, map_tags, adjust_points=False)  # assign to atria file

        #  Update tag times to reflect EPmed system time
        _sync_tag_times(map_tags, epmed_offset)

        log.debug(f"Saving {len(map_tags)} new MAP tags")
        cdata.write_case_tags(case_id, map_tags, mode="r")
