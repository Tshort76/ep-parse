from enum import Enum

import ep_parse.catheters as cath
import ep_parse.ui.common as uic


class TAG_TYPE(str, Enum):
    MAP = "MAP"
    RF = "RF"
    BOOKMARK = "BOOKMARK"
    TAG = "TAG"

    def __str__(self) -> str:
        return str.__str__(self)


def _tag_type(label: str) -> TAG_TYPE:
    for data in TAG_TYPE:
        if label.startswith(data.value):
            return data


def tag_type(tag: dict) -> TAG_TYPE:
    return _tag_type(tag["label"])


class Tag:
    def __init__(self, label: str, **kwargs):
        self.label = label
        self.tag_type = _tag_type(label)
        self.__dict__.update(kwargs)


def format_MAP_channels(
    channels: list[str] = None,
    coordinates: list[list[float]] = None,
    channel_to_coordinates: dict[str, list] = None,
    high_fidelity: bool = True,
) -> dict[str, list]:
    channel_to_coordinates = channel_to_coordinates or dict(zip(channels, coordinates))
    return {ch: {"xyz": coords, "high_fidelity": high_fidelity} for ch, coords in channel_to_coordinates.items()}


def update_fidelity_flag(channel_dict: dict[str, dict], high_fidelity_channels: list[str]) -> None:
    #  Setup this way to ensure that we mark channels as high fidelity even if we don't have coordinate data for it
    for ch in high_fidelity_channels:
        if ch not in channel_dict:
            channel_dict[ch] = {}
        channel_dict[ch]["high_fidelity"] = True
    for ch in set(channel_dict.keys()).difference(high_fidelity_channels):
        channel_dict[ch]["high_fidelity"] = False


def get_coordinates(tag: dict) -> list[tuple[float]]:
    return tag.get("coordinates") or [v["xyz"] for v in tag.get("channels", {}).values() if "xyz" in v]


def as_vtk_geometries(tag: dict, heart_area: float, is_selected: bool = False) -> list:
    "Returns a list of vtk geometry objects representing the tag"
    catheter = tag.get("catheter")
    if cath.is_mapping_cath(catheter):
        return cath.catheter_geometry(tag, heart_area)
    else:
        if is_selected:
            return [
                uic.sphere_geometry(
                    center=get_coordinates(tag),
                    color=uic.Colors.SELECTED.value,
                    radius=uic.ablation_radius(heart_area),
                )
            ]
        else:
            color = uic.Colors.TAG.value if catheter == cath.Catheter.TAG else uic.Colors.RF.value
            return [
                uic.sphere_geometry(
                    center=get_coordinates(tag),
                    color=color,
                    radius=uic.ablation_radius(heart_area),
                )
            ]
