import os
import pandas as pd
import logging
from datetime import datetime, timedelta

import ep_parse.utils as pu
import ep_parse.case_data as cdata
import ep_parse.geo_parsing as geo
import ep_parse.carto.geometries as cgeo
import ep_parse.carto.ablations as cab
import ep_parse.carto.map_points as cmp
import ep_parse.common as pic


log = logging.getLogger(__name__)
_ = log.setLevel(logging.DEBUG) if pu.is_dev_mode() else log.setLevel(logging.WARN)


# Because all times are given as an offset from some unknown time
def _case_time_0(export_dir: str, bookmarks: pd.DataFrame) -> datetime:
    # parse sessions because sites might be missing low duration sessions and we need alignment with bookmarks
    filepath = os.path.join(export_dir, "VisiTagExport", "VisiTagSessions.txt")
    visi = pd.read_csv(filepath, delim_whitespace=True).sort_values("StartTs", ignore_index=True)

    n = min(len(bookmarks), len(visi))
    bkmrk_durs = [x.total_seconds() for x in (bookmarks["end_time"] - bookmarks["start_time"])][0:n]
    visi_durs = ((visi["EndTs"] - visi["StartTs"]) / 1000)[0:n]

    # check for duration alignment with 5 consecutive RFs
    bk_idx = 0
    for i in range(n):
        if abs(visi_durs[i] - bkmrk_durs[bk_idx]) < 1:
            bk_idx += 1
        else:
            bk_idx = 0
        if bk_idx == 5:
            break

    if not bk_idx:
        debug_df = pd.DataFrame({"bkmk": bkmrk_durs, "visi": visi_durs, "visi_times": (visi["StartTs"] / 1000)[0:n]})
        # debug_df.to_csv("alignment.csv", index=False)
        print(debug_df)
        assert False, "Unable to align CARTO Visitag sessions with EPmed bookmarks file!"

    offset = (i + 1) - bk_idx
    log.info(f"Aligning RFs for carto time ... found {len(bookmarks)} bookmark RFs and {len(visi)} Visitag sessions")
    if offset:
        log.warning(f"Bookmark RF 1 aligns with the nth={offset+1} Visitag session.  Adjusting ...")
    return bookmarks.iloc[0]["start_time"] - timedelta(milliseconds=int(visi.iloc[offset]["StartTs"]))


def import_carto_export(case_id: str, parse_types: list[str], event_loader_fn, opts: dict = {}) -> None:
    export_dir = cdata.case_file_path(case_id, pic.DataSource.CARTO)
    log.info(f"Loading {case_id} carto export data at location: {export_dir}")
    atria_meshes = cgeo.atrium_geo_meshes(export_dir)

    if "geometry" in parse_types:
        # needs to be done in order of LA, RA?
        la_file = os.path.join(export_dir, atria_meshes["LA"]) if atria_meshes.get("LA") else None
        ra_file = os.path.join(export_dir, atria_meshes["RA"]) if atria_meshes.get("RA") else None
        geo_vtks = cgeo.parse_meshes_as_vtk(case_id, la_file=la_file, ra_file=ra_file)
        log.info(f"Parsed geo files: {geo_vtks}")

    if set(parse_types).intersection(("RF", "MAP")):
        time_0 = _case_time_0(export_dir, event_loader_fn(case_id))

    rf_tags = []
    if "RF" in parse_types:
        rf_tags = cab.parse_RF_tags(export_dir, time_0)
        geo.update_vtk_fields(case_id, rf_tags, adjust_points=True)
        log.info(f"Parsed {len(rf_tags)} RF tags from Lesions file")
        cdata.write_case_tags(case_id, rf_tags, mode="r")

    map_tags = []
    if "MAP" in parse_types:
        map_tags = cmp.parse_Map_tags(export_dir, time_0, opts)
        geo.update_vtk_fields(case_id, map_tags, adjust_points=False)
        log.info(f"Parsed {len(map_tags)} MAP tags from Map files")
        cdata.write_case_tags(case_id, map_tags, mode="r")
