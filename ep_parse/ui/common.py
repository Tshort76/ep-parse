import toolz as tz
from enum import Enum
import json
import dash_vtk

import ep_parse.plots.common as ppc
from ep_parse.utils import euc_distance, centroid

TIME_ENTRY_FORMAT = r"\d{2}:\d{2}:\d{2}(\.\d{1,6})?"  # HH:mm:ss(.mmmmmm)


class Colors(Enum):
    CATH_ELECTRODE = [0.47, 0.8, 0.95]
    CATH_ORIENT_ELECTRODE = [0.1, 0.26, 1]
    RF = [1, 1, 0.9]
    SELECTED = [0.27, 0.27, 0.27]
    TAG = [0.58, 0, 1]
    MARKED = [0.12, 0.66, 0.95]
    HEART_MODEL = [1, 1, 1]

    def as_255(self) -> tuple:
        return [int(255 * x) for x in self.value]


MAPPING_COLOR_SEQ = [
    [69, 214, 239],
    [64, 198, 223],
    [54, 169, 195],
    [44, 137, 165],
    [36, 115, 144],
    [28, 92, 121],
    [19, 69, 98],
]

HIGHLIGHT_ROW_PROPS = {"backgroundColor": "rgba(0, 116, 217, 0.3)", "border": "1px solid rgb(0, 116, 217)"}
DARK_CELL = {"backgroundColor": "rgb(50, 50, 50)", "color": "white"}
LIST_INPUT_STYLE = {"margin-left": "10px", "margin-right": "5px"}


class PlotClickMode(str, Enum):
    SHOW_CS_SIGNALS = "Show CS Signals"
    MARK_VERIFIED = "Mark (verified)"
    MARK_UNVERIFIED = "Mark (unverified)"
    AUDIT_ALERT = "Show Alert Details"

    def __str__(self) -> str:
        return str.__str__(self)


class HeartClickMode(str, Enum):
    PLOT_DISEASE_SIGNALS = "Show Mapping Signals"
    HIGHLIGHT_ON_PLOT = "Show Occurrence In Time"
    PLOT_SIGNAL_SIMILARITY = "Show Similar Signals"
    SHOW_SIMILAR_LOCATIONS = "Show Similar Locations"

    def __str__(self) -> str:
        return str.__str__(self)


class CatheterPlacementMode(str, Enum):
    PLACE_SPLINES = "center + 2 tips"
    CENTER_ONLY = "center only"

    def __str__(self) -> str:
        return str.__str__(self)


# Note dimension[0] is the radius of the first assistant sphere, dimension[1] is the radius of the second assistant sphere

# If desired, dimensions were set for a model with the following surface area.  Code could use multiply dimensions if needed
# by (Heart3dModel().surface_area / HEART_SURFACE_AREA) if desired
HEART_SURFACE_AREA = 344  # dimensions were set for a heart with this surface area


def _base_radius(surface_area: float) -> float:
    return min(0.5, max(0.025, (surface_area**0.5) / 800))


def electrode_radius(surface_area: float):
    return 3 * _base_radius(surface_area)


def ablation_radius(surface_area: float):
    return 7 * _base_radius(surface_area)


def cuboid_radius(points: list[tuple], center: tuple = None) -> float:
    center = center or centroid(points)
    return max([euc_distance(p, center) for p in points])


def disease_color(score: float, scale_min: float = 0, scale_max: float = 1) -> tuple[float, float, float]:
    return ppc.normalized_color(score, ppc.ColorMap.DISEASE, min=scale_min, max=scale_max, fmt="int_tuple")


def response_color(
    score: float, scale_min: float = 0, scale_max: float = 1, fmt="int_tuple"
) -> tuple[float, float, float]:
    return ppc.normalized_color(score, ppc.ColorMap.RESPONSE, min=scale_min, max=scale_max, fmt=fmt)


def _expand_config_hierarchy(raw_config: dict) -> dict:
    results = {}
    for k, val in raw_config.items():
        key_seq = k.split("-")
        if key_seq[-1] == "checklist":  # list of True items
            for v in val:
                results = tz.assoc_in(results, key_seq[:-1] + [v], True)
        else:
            results = tz.assoc_in(results, key_seq, val)
    return results


def _id_vals(obj: dict, found: dict):
    if "props" not in obj:
        return found
    if "id" in obj["props"] and obj.get("type") in {"Input", "Checklist", "Dropdown"}:
        return {**found, obj["props"]["id"]: obj["props"].get("value")}
    elif kids := obj["props"].get("children"):
        if isinstance(kids, dict):
            return _id_vals(kids, found)
        elif isinstance(kids, list):
            kids_found = tz.reduce(lambda x, k: {**x, **_id_vals(k, {})}, kids, {})
            return {**found, **kids_found}
    return found


def _form_input_values(form_children: list) -> dict:
    return tz.reduce(lambda x, k: {**x, **_id_vals(k, {})}, form_children, {})


def parse_configs_form(form_children: list):
    return _expand_config_hierarchy(_form_input_values(form_children))


def dict_as_pandas_json(d: dict) -> str:
    return f"\\{json.dumps(d)}\\"


def cube_geometry(
    parent_surface_area: float, center: tuple[float], color: tuple = None, radius: float = None
) -> dash_vtk.GeometryRepresentation:
    _radius = radius or electrode_radius(parent_surface_area)
    return dash_vtk.GeometryRepresentation(
        property={"color": (color or Colors.RF.value), "opacity": 1},
        children=[
            dash_vtk.Algorithm(
                vtkClass="vtkCubeSource",
                state={"xLength": _radius, "yLength": _radius, "zLength": _radius, "center": center},
            )
        ],
    )


def sphere_geometry(
    center: tuple, radius: float, color: tuple = Colors.RF.value, opacity: float = 1
) -> dash_vtk.GeometryRepresentation:
    return dash_vtk.GeometryRepresentation(
        property={"color": color, "opacity": opacity},
        children=[dash_vtk.Algorithm(vtkClass="vtkSphereSource", state={"radius": radius, "center": center})],
    )
