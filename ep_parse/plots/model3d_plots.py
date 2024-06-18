import pyvista as pv
import ep_parse.case_data as d
import numpy as np

# Set/enable the backed
pv.set_jupyter_backend("trame")


def plot_heart(
    case_id: str,
    heart_vtk_file: str,
    mesh_color: str = "#f8f3f3",
    mesh_opacity: float = 1,
    background_color: str = "#3f3f3f",
    RF_tag_color: str = "#0fa6c8",
    MAP_tag_color: str = "#e90d0d",
    misc_tag_color: str = "#99b20a",
) -> None:
    vtk_mesh = pv.read(heart_vtk_file)
    tags = d.load_case_tags(case_id)

    tag_data = []
    for tag_set in (
        [t for t in tags if t["label"].startswith("MAP")],
        [t for t in tags if t["label"].startswith("RF")],
        [t for t in tags if not (t["label"].startswith("MAP") or t["label"].startswith("RF"))],
    ):
        tag_data.append((np.array([t["centroid"] for t in tag_set]), [t["label"] for t in tag_set]))

    # Create a plotter object
    plotter = pv.Plotter(notebook=True)

    plotter.add_mesh(vtk_mesh, color=mesh_color, opacity=mesh_opacity)
    plotter.set_background(background_color)

    # https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_point_labels.html#pyvista-plotter-add-point-labels
    for (centroids, labels), clr in zip(tag_data, (MAP_tag_color, RF_tag_color, misc_tag_color)):
        if labels:  # empty collection raises error
            plotter.add_point_labels(
                centroids,
                labels,
                font_size=8,
                point_color=clr,
                point_size=8,
                render_points_as_spheres=True,
                always_visible=False,
                tolerance=0.0001,
            )

    plotter.show()
