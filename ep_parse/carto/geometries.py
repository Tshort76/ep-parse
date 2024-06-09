import re
import os
import logging

import ep_parse.geo_parsing as geo
import ep_parse.utils as pu
import ep_parse.case_data as cdata

log = logging.getLogger(__name__)
_ = log.setLevel(logging.DEBUG) if pu.is_dev_mode() else log.setLevel(logging.WARN)

LA_VTK = re.compile(r".*\d-la\.mesh", re.IGNORECASE)
RA_VTK = re.compile(r".*\d-ra\.mesh", re.IGNORECASE)


def _best_atria_mesh(export_dir: str, candidates: list[str]) -> str:
    "best atria mesh is the one with the greatest number of associated data files"
    map_data_counts = {c[:-5]: 0 for c in candidates}
    for f in os.listdir(export_dir):
        for k, v in map_data_counts.items():
            if f.startswith(f"{k}_") and f.endswith(".xml"):
                map_data_counts[k] = v + 1
    return max(map_data_counts.items(), key=lambda x: x[1])[0] + ".mesh"


def atrium_geo_meshes(export_dir: str) -> list[str]:
    atrium_meshes = {"LA": [], "RA": []}
    for f in os.listdir(export_dir):
        if f.endswith(".mesh"):
            if RA_VTK.match(f):
                atrium_meshes["RA"].append(f)
            elif LA_VTK.match(f):
                atrium_meshes["LA"].append(f)

    if sum([len(v) for v in atrium_meshes.values()]) == 0:
        raise FileNotFoundError(f"Unable to locate atria geometries files of form #-la.mesh or #-ra.mesh")

    result = {}
    for k, v in atrium_meshes.items():
        if len(v) == 1:
            result[k] = v[0]
        elif len(v) > 1:
            log.debug(f"Found multiple mesh files for {k}, choosing best from {v}")
            result[k] = _best_atria_mesh(export_dir, v)
        else:
            result[k] = None
            log.warning(f"Could not find a mesh file for the {k} atrium, proceeding without it")

    log.info(f"Using the following atrium files: {result}")
    return result


def _mesh_as_geo(mesh_file: str) -> dict:
    geo_data = {"points": [], "polygons": []}
    section = "attributes"
    with open(mesh_file, "r", encoding="utf-8", errors="ignore") as fp:
        for l in fp.readlines():
            if section in ("points", "polygons"):
                if l.lstrip().startswith(";"):
                    continue
                rdata = l.split()
                if len(rdata) > 5 and rdata[1] == "=":
                    if section == "polygons":
                        geo_data[section].append([int(x) for x in rdata[2:5]])
                    else:
                        geo_data[section].append(rdata[2:5])
                elif "TrianglesSection" in l:
                    section = "polygons"
                elif "Section]" in l:
                    break
            elif section == "attributes" and "VerticesSection" in l:
                section = "points"
    return geo_data


def parse_meshes_as_vtk(case_id: str, la_file: str = None, ra_file: str = None):
    files_written, case_dir = [], cdata.case_data_directory(case_id)
    if ra_file:
        ra_geo = _mesh_as_geo(ra_file)
        filepath = os.path.join(case_dir, f"{case_id}_RA.vtk")
        geo.geo_data_as_vtk(ra_geo, filepath)
        files_written.append(f"{case_id}_RA.vtk")

    if la_file:
        la_geo = _mesh_as_geo(la_file)
        filepath = os.path.join(case_dir, f"{case_id}_LA.vtk")
        geo.geo_data_as_vtk(la_geo, filepath)
        files_written.append(f"{case_id}_LA.vtk")

    if ra_file and la_file:
        # join LA and RA to create LA_and_RA file
        combined = geo.combine_geos((ra_geo, la_geo))
        filepath = os.path.join(case_dir, f"{case_id}_LA_and_RA.vtk")
        geo.geo_data_as_vtk(combined, filepath)
        files_written.append(f"{case_id}_LA_and_RA.vtk")

    return files_written
