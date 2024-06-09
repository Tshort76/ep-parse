import logging
import os
from enum import Enum
import numpy as np
import toolz as tz

import dash_vtk
import dash_vtk.utils as dvu
from vtkmodules.vtkCommonCore import mutable, vtkIdList, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkCellLocator, vtkPointLocator, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkMassProperties, vtkTriangleFilter
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

import ep_parse.ui.common as uic
import ep_parse.tag as ptg
import ep_parse.catheters as cath
import ep_parse.case_data as cdata

log = logging.getLogger(__name__)

HEART_REGIONS = ["LA", "RA", "LA+RA"]


def case_specific_vtk_file(region: str, case_id: str, full_path: bool = False) -> str:
    clean_region = region.replace("+", "_and_")
    fname = f"{case_id}_{clean_region}.vtk"
    return os.path.join(cdata.case_data_directory(case_id), fname) if full_path else fname


def _has_case_specific_geometry(region: str, case_id: str) -> bool:
    if case_id:
        case_vtk = case_specific_vtk_file(region, case_id)
        return os.path.exists(os.path.join(cdata.case_data_directory(case_id), case_vtk))


def model_file_of(case_id: str, region: str) -> str:
    case_vtk = case_specific_vtk_file(region, case_id)
    if os.path.exists(os.path.join(cdata.case_data_directory(case_id), case_vtk)):
        return case_vtk
    return f"{region.replace('+','_and_')}_v1.vtk"


class LocatorType(Enum):
    CELL = 1
    POINT = 2


class Heart3dModel:
    def __init__(self, region: str, case_id: str = None, opacity: float = 1):
        case_vtk = case_specific_vtk_file(region, case_id)
        self.case_specific = _has_case_specific_geometry(region, case_id)
        log.debug(f"Loading {case_vtk if self.case_specific else 'generic'} heart geometry")
        self.case_id = case_id
        self.vtk_file = case_vtk if self.case_specific else f"{region.replace('+','_and_')}_v1.vtk"
        self.region = region
        self._polydata, self.volume, self.surface_area = self.__load_polydata()
        self._colors = {}
        self._opacity = opacity
        self.refresh_geometry()
        self._cell_locator = self.__build_locator(LocatorType.CELL)
        self._point_locator = self.__build_locator(LocatorType.POINT)

    def __load_polydata(self) -> tuple[vtkPolyData, float, float]:
        reader = vtkUnstructuredGridReader()
        if self.case_specific:
            fpath = os.path.join(cdata.case_data_directory(self.case_id), self.vtk_file)
        else:
            fpath = os.path.join("resources", "geometries", self.vtk_file)
        assert os.path.exists(fpath), f"Heart model file not found: {fpath}"

        reader.SetFileName(fpath)
        geo_filter = vtkGeometryFilter()
        geo_filter.SetInputConnection(reader.GetOutputPort())
        geo_filter.Update()
        raw_poly = geo_filter.GetOutput()

        tri = vtkTriangleFilter()
        tri.SetInputData(raw_poly)
        tri_out = tri.GetOutputPort()
        clean = vtkCleanPolyData()
        clean.SetInputConnection(tri_out)
        clean.Update()

        massP = vtkMassProperties()
        massP.SetInputConnection(tri_out)
        massP.Update()
        return clean.GetOutput(), massP.GetVolume(), massP.GetSurfaceArea()

    def refresh_geometry(self):
        self.geometry = dash_vtk.GeometryRepresentation(
            children=[dash_vtk.Mesh(id="heart-mesh", state=dvu.to_mesh_state(self._polydata, "colors"))],
            property={"edgeVisibility": False, "opacity": self._opacity},
        )

    def __build_locator(self, ltype: LocatorType) -> vtkCellLocator:
        locator = vtkCellLocator() if ltype == LocatorType.CELL else vtkPointLocator()
        locator.SetDataSet(self._polydata)
        locator.BuildLocator()
        return locator

    def relevant_tags(self, tags: list[dict], include_misc: bool = False) -> list[dict]:
        fileset = set([self.vtk_file])
        if "+" in self.region:
            if self.case_specific:
                case_id = "_".join(self.vtk_file.split("_")[:-3])
                fileset.add(case_specific_vtk_file("LA", case_id))
                fileset.add(case_specific_vtk_file("RA", case_id))
            else:
                fileset = {"RA_v1.vtk", "LA_v1.vtk"}
        if include_misc:  # include tags without a vtk_file (e.g. bookmarks)
            fileset.add(None)

        return [t for t in tags if t.get("vtk_file") in fileset]

    def closest_coordinates_on_surface(self, target_points: list[tuple]) -> list[tuple]:
        cpoint = [0.0, 0.0, 0.0]
        cell_id = mutable(0)
        sub_id = mutable(0)
        d = mutable(0.0)

        coords = []
        for p in target_points:
            self._cell_locator.FindClosestPoint(p, cpoint, cell_id, sub_id, d)
            coords.append(cpoint[:])  # use slice to make a copy
        return coords

    def coordinates_on_surface_within(self, origin: tuple[float], radius: float) -> list[np.ndarray]:
        ids = vtkIdList()
        self._point_locator.FindPointsWithinRadius(radius, origin, ids)
        points = self._polydata.GetPoints().GetData()
        return [np.array(points.GetTuple(ids.GetId(i))) for i in range(ids.GetNumberOfIds())]

    def closest_point_ids(self, target_points: list[tuple], mode: str = "all") -> list[int]:
        points = [uic.centroid(target_points)] if mode == "centroid" else target_points
        return {self._point_locator.FindClosestPoint(p) for p in points}

    def update_colors(self, idx_to_color: dict):
        Colors = vtkUnsignedCharArray()
        Colors.SetNumberOfComponents(3)
        Colors.SetName("colors")
        for idx in range(self._polydata.GetNumberOfPoints()):
            color = idx_to_color.get(idx, uic.Colors.HEART_MODEL.as_255())
            Colors.InsertNextTuple3(*color)
        self._polydata.GetPointData().SetScalars(Colors)

    def color_by_tags(self, tags: list[dict], opts: dict = {}):
        model_tags = self.relevant_tags(tags, False)
        colors = {}
        for i, tag in enumerate(model_tags):
            if cath.is_mapping_cath(tag.get("catheter")):
                if opts.get("Color Mapped Areas"):
                    color_idx = int((i * len(uic.MAPPING_COLOR_SEQ)) / len(tags))
                    colors = {
                        **colors,
                        **{
                            pid: uic.MAPPING_COLOR_SEQ[color_idx]
                            for pid in self.closest_point_ids(ptg.get_coordinates(tag))
                        },
                    }
                elif (dmetric := opts.get("Color by Disease Metric")) and "disease_scores" in tag:
                    scale_min, scale_max = opts.get("disease_color_max_min", [0, 1])
                    for ch, scores in tag["disease_scores"].items():
                        color = uic.disease_color(scores.get(dmetric, 0), scale_min=scale_min, scale_max=scale_max)
                        if coords := tz.get_in(["channels", ch, "xyz"], tag):
                            pid = next(iter(self.closest_point_ids([coords])), None)
                            # might want to check if pid in colors and 'merge' the two colors (midpoint, max ?) if it is
                            colors[pid] = color
            elif tag.get("catheter") == cath.Catheter.ABL and opts.get("Color Ablated Areas"):
                pid = next(iter(self.closest_point_ids([ptg.get_coordinates(tag)])), None)
                # colors = {**colors, pid: uic.Colors.RF.as_255()}
                colors = {**colors, pid: [168, 121, 50]}
        self.update_colors(colors)
        self.refresh_geometry()

    def color_by_similarity(
        self, tags: list[dict], tag_similarity: dict[str, float], color_vrange: tuple[float, float] = (0, 1)
    ):
        """Updates the heart models surface color according to tag similarity

        Args:
            tags (list[dict]): collection of tag objects associated with the heart
            tag_similarity (dict[str,float]): tag.label _ tag.channel -> similarity score (0,1)
        """
        model_tags = self.relevant_tags(tags, False)
        colors = {}
        for tag in model_tags:
            tag_colors, pids = [], []
            if tag.get("label", "").startswith("RF"):
                r = tag.get("radius", 0.35)
                similarity = tag_similarity.get(f"{tag['label']}_ABLd", 0)
                tag_color = uic.disease_color(similarity, *color_vrange)
                # color the outer radius of sphere
                coords = self.coordinates_on_surface_within(ptg.get_coordinates(tag), r * 0.75)
                pids = self.closest_point_ids(coords)
                tag_colors = [tag_color for _ in pids]
            elif "channels" in tag:
                tag_colors = [
                    uic.disease_color(tag_similarity.get(tag["label"] + "_" + ch, 0), *color_vrange)
                    for ch, v in tag["channels"].items()
                    if "xyz" in v
                ]
                pids = self.closest_point_ids(ptg.get_coordinates(tag))

            colors = {
                **colors,
                **dict(zip(pids, tag_colors)),
            }
        self.update_colors(colors)
        self.refresh_geometry()
