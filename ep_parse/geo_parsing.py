import numpy as np
import os
from collections.abc import Collection
import logging

import ep_parse.ui.heart_model as hm
from ep_parse.ui.common import centroid


log = logging.getLogger(__name__)


def geo_data_as_vtk(geo_data: dict, output_file: str):
    with open(output_file, "w") as fp:
        fp.write("# vtk DataFile Version 2.0\nWritten by tlong\nASCII\nDATASET UNSTRUCTURED_GRID\n")

        # omit the last point, which seems to cause errors if present
        fp.write(f"POINTS {len(geo_data['points'])} float\n")
        for coords in geo_data["points"]:
            fp.write(" ".join(map(str, coords)) + "\n")

        polys = geo_data["polygons"]
        fp.write(f"CELLS {len(polys)} {4*len(polys)}\n")
        for indices in polys:
            fp.write(f"3 {' '.join(map(str, indices))}\n")

        fp.write(f"CELL_TYPES {len(polys)}\n")
        for i in range(len(polys)):
            fp.write("5\n")  # 1 for vertex, 5 for triangle


def combine_geos(geos: list[dict]):
    combined = {k: v for k, v in geos[0].items()}
    for geo in geos[1:]:
        _offset = len(combined["points"])
        if "normals" in combined:
            combined["normals"] += geo.get("normals", [])
        combined["points"] += geo["points"]
        combined["polygons"] += [[x + _offset for x in poly] for poly in geo["polygons"]]
    combined["volume"] = "ALL"
    return combined


def _assoc_with_atria(case_id: str, centroids: list[np.array], tags: list[dict], adjust_points: bool):
    la_model = hm.Heart3dModel("LA", case_id=case_id)
    la_surface_coords = la_model.closest_coordinates_on_surface(centroids)
    la_dists = [np.linalg.norm(c - np.array(p)) for c, p in zip(centroids, la_surface_coords)]

    # RA
    ra_model = hm.Heart3dModel("RA", case_id=case_id)
    ra_surface_coords = ra_model.closest_coordinates_on_surface(centroids)
    ra_dists = [np.linalg.norm(c - np.array(p)) for c, p in zip(centroids, ra_surface_coords)]

    for i in range(len(tags)):
        if min(la_dists[i], ra_dists[i]) > 5:
            log.warning(f"tag found that is far from a surface: {tags[i]['label']}")
            tags[i]["vtk_file"] = "no_surface.vtk"
            continue

        atria = "LA" if la_dists[i] < ra_dists[i] else "RA"
        tags[i]["vtk_file"] = f"{case_id}_{atria}.vtk"
        if adjust_points:
            raw_coords = tags[i]["coordinates"]
            if isinstance(raw_coords[0], Collection):  # Many coordinates
                hmodel = la_model if atria == "LA" else ra_model
                coords = hmodel.closest_coordinates_on_surface(raw_coords)
                tags[i]["coordinates"] = coords
                tags[i]["centroid"] = centroid(coords)
            else:
                coords = la_surface_coords[i] if atria == "LA" else ra_surface_coords[i]
                tags[i]["coordinates"] = coords
                tags[i]["centroid"] = coords


def update_vtk_fields(case_id: str, tags: list[dict], adjust_points: bool = True) -> None:
    centroids = [np.array(x["centroid"]) for x in tags]

    case_la_vtk = hm.case_specific_vtk_file("LA", case_id, full_path=True)
    case_ra_vtk = hm.case_specific_vtk_file("RA", case_id, full_path=True)

    if os.path.exists(case_la_vtk) and os.path.exists(case_ra_vtk):
        return _assoc_with_atria(case_id, centroids, tags, adjust_points)

    # single model
    assert os.path.exists(
        hm.case_specific_vtk_file("LA+RA", case_id, full_path=True)
    ), f"No case specific heart model exists for {case_id}.  Be sure to parse the 'geometry' file for this case while/before processing RF and Map points"
    model = hm.Heart3dModel("LA+RA", case_id=case_id)
    if adjust_points:
        surface_coords = model.closest_coordinates_on_surface(centroids)

    for i in range(len(tags)):
        tags[i]["vtk_file"] = f"{case_id}_LA_and_RA.vtk"
        if adjust_points:
            raw_coords = tags[i]["coordinates"]
            if isinstance(raw_coords[0], Collection):  # Many coordinates for the tag
                coords = model.closest_coordinates_on_surface(raw_coords)
                tags[i]["coordinates"] = coords
                tags[i]["centroid"] = centroid(coords)
            else:  # single point for the tag
                coords = surface_coords[i]
                tags[i]["coordinates"] = coords
                tags[i]["centroid"] = coords
