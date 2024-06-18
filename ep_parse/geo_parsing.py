import numpy as np
import os
from collections.abc import Collection
import logging

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
