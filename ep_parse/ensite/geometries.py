import os
import ep_parse.geo_parsing as geo
import ep_parse.case_data as cdata


def _format_data(dtype: str, data_line: str) -> list:
    match dtype:
        case "points" | "normals":
            return list(map(float, data_line.split()))
        case "polygons":
            # zero index the point indices for vtk
            return list(map(lambda x: int(x) - 1, data_line.split()))
        case _:
            return data_line


def _parse_geoXML_data(model_file: str) -> list[str]:
    "Creates vtk files for each volume in the file, returns the filepaths for the vtk files"

    all_vtk_data, curr_tag, volume_names = [], {}, []
    with open(model_file, "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if line.startswith("<Volume "):
                vol_name = line.split('"')[1]
                volume_names.append(vol_name)
            elif line.startswith("<Vertices"):
                curr_tag = {"header": "points", "volume": vol_name, "data": []}
            elif line.startswith("<Polygons"):
                curr_tag = {"header": "polygons", "volume": vol_name, "data": []}
            elif line.startswith("<Normals"):
                curr_tag = {"header": "normals", "volume": vol_name, "data": []}
            elif line.startswith("</Vertices") or line.startswith("</Polygons") or line.startswith("</Normals"):
                all_vtk_data.append(curr_tag)
                curr_tag = {}
            elif curr_tag.get("header", ""):
                curr_tag["data"].append(_format_data(curr_tag["header"], line))

    rvals = []
    for vname in volume_names:
        _data = {"volume": vname}
        for d in all_vtk_data:
            if d["volume"] == vname:
                _data[d["header"]] = d["data"]
        rvals.append(_data)

    return rvals


def geoXML_as_vtk(case_id: str, filepath: str) -> list[str]:
    "Parses the geometries from the file and creates vtk files for them.  Returns vtk filenames"
    geometries, filenames = _parse_geoXML_data(filepath), []
    case_dir = cdata.case_data_directory(case_id)

    if len(geometries) > 1:
        for geo_data in geometries:
            fname = f"{case_id}_{geo_data['volume'][0]}A.vtk"
            output_file = os.path.join(case_dir, fname)
            geo.geo_data_as_vtk(geo_data, output_file)
            filenames.append(output_file)

    # merged geo file
    output_file = os.path.join(case_dir, f"{case_id}_LA_and_RA.vtk")
    geo_data = geo.combine_geos(geometries)
    geo.geo_data_as_vtk(geo_data, output_file)
    return filenames + [output_file]


def adjust_geo_meta(case_id: str, tags: list[dict], adjust_points: bool = True) -> None:
    return geo.update_vtk_fields(case_id, tags, adjust_points)
