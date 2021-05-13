import os
from pathlib import Path

import fenics as fe
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio

try:
    _SHOW_PLOT = bool(int(os.getenv("FENICS_PLOTLY_SHOW", 1)))
except ValueError:
    _SHOW_PLOT = True


def set_renderer(renderer):
    pio.renderers.default = renderer


def savefig(fig, filename, save_config=None):
    """Save figure to file

    Parameters
    ----------
    fig : `plotly.graph_objects.Figure`
        This figure that you want to save
    filename : Path or str
        Path to the desitnation where you want to
        save the figure
    save_config : dict, optional
        Additionsal cofigurations to be assed
        to `plotly.offline.plot`, by default None
    """

    filename = Path(filename)
    outdir = filename.parent
    assert outdir.exists(), f"Folder {outdir} does not exist"

    config = {
        "toImageButtonOptions": {
            "filename": filename.stem,
            "width": 1500,
            "height": 1200,
        }
    }
    if save_config is not None:
        config.update(save_config)

    path = outdir.joinpath(filename)
    plotly.offline.plot(fig, filename=path.as_posix(), auto_open=False, config=config)


def _get_vertex_values(function):

    space = function.function_space()
    mesh = space.mesh()
    shape = function.ufl_shape

    if isinstance(space.ufl_element(), fe.FiniteElement):
        return function.compute_vertex_values()
    else:
        size_n = shape[0]
        if isinstance(space.ufl_element(), fe.VectorElement):
            res = np.zeros((mesh.num_entities(0), shape[0]))
            for i in range(size_n):
                res[:, i] = function.sub(i, deepcopy=True).compute_vertex_values()

        if isinstance(space.ufl_element(), fe.TensorElement):
            res = np.zeros((mesh.num_entities(0), shape[0], shape[1]))
            count = 0
            for i in range(size_n):
                for j in range(size_n):
                    res[:, i, j] = function.sub(
                        count + j, deepcopy=True
                    ).compute_vertex_values()
                count += size_n
        return res


def _get_triangles(mesh):
    num_faces = mesh.num_faces()
    if num_faces == 0:
        num_faces = len(list(fe.faces(mesh)))
    triangle = np.zeros((3, num_faces), dtype=int)

    for ind, face in enumerate(fe.faces(mesh)):
        triangle[:, ind] = face.entities(0)
    return triangle


def _cone_plot(function, size=10, showscale=True, normalize=False, **kwargs):

    mesh = function.function_space().mesh()
    points = mesh.coordinates()
    vectors = _get_vertex_values(function)

    if len(points[0, :]) == 2:
        points = np.c_[points, np.zeros(len(points[:, 0]))]

    if vectors.shape[1] == 2:
        vectors = np.c_[vectors, np.zeros(len(vectors[:, 0]))]

    if normalize:
        vectors = np.divide(vectors.T, np.linalg.norm(vectors, axis=1)).T

    cones = go.Cone(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        u=vectors[:, 0],
        v=vectors[:, 1],
        w=vectors[:, 2],
        sizemode="absolute",
        sizeref=size,
        showscale=showscale,
    )

    return cones


def _wireframe_plot_mesh(mesh, **kwargs):
    coord = mesh.coordinates()
    triangle = mesh.cells()
    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    tri_points = coord[triangle]
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k % 3][0] for k in range(4)] + [None])
        Ye.extend([T[k % 3][1] for k in range(4)] + [None])
        Ze.extend([T[k % 3][2] for k in range(4)] + [None])

    # define the trace for triangle sides
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode="lines",
        name="",
        line=dict(color="rgb(70,70,70)", width=2),
        hoverinfo="none",
    )

    return lines


def _surface_plot_function(
    function, colorscale, showscale=True, intensitymode="vertex", **kwargs
):
    mesh = function.function_space().mesh()
    if intensitymode == "vertex":
        val = function.compute_vertex_values()
        triangle = _get_triangles(mesh)
    else:
        # Spooky!
        val = function.vector().get_local()
        triangle = mesh.cells().T
    coord = mesh.coordinates()

    hoverinfo = ["val:" + "%.5f" % item for item in val]

    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    surface = go.Mesh3d(
        x=coord[:, 0],
        y=coord[:, 1],
        z=coord[:, 2],
        i=triangle[0, :],
        j=triangle[1, :],
        k=triangle[2, :],
        flatshading=True,
        intensitymode=intensitymode,
        intensity=val,
        colorscale=colorscale,
        lighting=dict(ambient=1),
        name="",
        hoverinfo="all",
        text=hoverinfo,
        showscale=showscale,
    )

    return surface


def _scatter_plot_function(function, colorscale, showscale=True, size=10, **kwargs):
    dofs_coord = function.function_space().tabulate_dof_coordinates()
    if len(dofs_coord[0, :]) == 2:
        dofs_coord = np.c_[dofs_coord, np.zeros(len(dofs_coord[:, 0]))]

    mesh = function.function_space().mesh()
    val = function.vector().get_local()
    coord = mesh.coordinates()
    hoverinfo = ["val:" + "%.5f" % item for item in val]

    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    points = go.Scatter3d(
        x=dofs_coord[:, 0],
        y=dofs_coord[:, 1],
        z=dofs_coord[:, 2],
        mode="markers",
        marker=dict(size=size, color=val, colorscale=colorscale),
        hoverinfo="all",
        text=hoverinfo,
    )

    return points


def _surface_plot_mesh(mesh, color, opacity=1.0, **kwargs):
    coord = mesh.coordinates()
    triangle = _get_triangles(mesh)
    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    surface = go.Mesh3d(
        x=coord[:, 0],
        y=coord[:, 1],
        z=coord[:, 2],
        i=triangle[0, :],
        j=triangle[1, :],
        k=triangle[2, :],
        flatshading=True,
        color=color,
        opacity=opacity,
        lighting=dict(ambient=1),
    )

    return surface


def _plot_dofs(functionspace, size, **kwargs):
    dofs_coord = functionspace.tabulate_dof_coordinates()
    if len(dofs_coord[0, :]) == 2:
        dofs_coord = np.c_[dofs_coord, np.zeros(len(dofs_coord[:, 0]))]

    points = go.Scatter3d(
        x=dofs_coord[:, 0],
        y=dofs_coord[:, 1],
        z=dofs_coord[:, 2],
        mode="markers",
        name=kwargs.get("name", None),
        marker=dict(size=size),
    )

    return points


def _plot_dirichlet_bc(diri, size, colorscale="inferno", **kwargs):
    values = diri.get_boundary_values()
    dofs = diri.function_space().tabulate_dof_coordinates()
    if len(dofs[0, :]) == 2:
        dofs = np.c_[dofs, np.zeros(len(dofs[:, 0]))]
    coords_diri = np.array([dofs[i] for i in values.keys()])
    vals_diri = np.array(list(values.values()))

    points = go.Scatter3d(
        x=coords_diri[:, 0],
        y=coords_diri[:, 1],
        z=coords_diri[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=vals_diri,
            colorscale=colorscale,
            colorbar=dict(thickness=20),
        ),
    )

    return points


def _surface_plot_meshfunc(meshfunc, colorscale, **kwargs):
    assert meshfunc.dim() == 2
    mesh = meshfunc.mesh()
    array = meshfunc.array()
    coord = mesh.coordinates()
    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    triangle = _get_triangles(mesh)
    hoverinfo = ["val:" + "%d" % item for item in array]

    surface = go.Mesh3d(
        x=coord[:, 0],
        y=coord[:, 1],
        z=coord[:, 2],
        i=triangle[0, :],
        j=triangle[1, :],
        k=triangle[2, :],
        flatshading=True,
        intensity=array,
        colorscale=colorscale,
        lighting=dict(ambient=1),
        name="",
        hoverinfo="all",
        text=hoverinfo,
        intensitymode="cell",
    )

    return surface


def _handle_mesh(obj, **kwargs):
    data = []
    wireframe = kwargs.get("wireframe", True)
    if wireframe:
        surf = _surface_plot_mesh(obj, **kwargs)
        data.append(surf)

    data.append(_wireframe_plot_mesh(obj))

    return data


def _handle_function(
    obj,
    **kwargs,
):
    data = []
    scatter = kwargs.get("scatter", False)
    norm = kwargs.get("norm", False)
    component = kwargs.get("component", None)

    if len(obj.ufl_shape) == 0:
        if scatter:
            surface = _scatter_plot_function(obj, **kwargs)
        else:
            surface = _surface_plot_function(obj, **kwargs)
        data.append(surface)

    elif len(obj.ufl_shape) == 1:
        if norm or component == "magnitude":
            V = obj.function_space().split()[0].collapse()
            magnitude = fe.project(fe.sqrt(fe.inner(obj, obj)), V)
        else:
            magnitude = None

        if component is None:
            if norm:
                surface = _surface_plot_function(magnitude, **kwargs)
                data.append(surface)

            cones = _cone_plot(obj, **kwargs)
            data.append(cones)
        else:
            if component == "magnitude":
                surface = _surface_plot_function(magnitude, **kwargs)
                data.append(surface)
            else:
                for i, comp in enumerate(["x", "y", "z"]):

                    if component not in [comp, comp.upper()]:
                        continue
                    surface = _surface_plot_function(
                        obj.sub(i, deepcopy=True), **kwargs
                    )
                    data.append(surface)

    if kwargs.get("wireframe", True):
        lines = _wireframe_plot_mesh(obj.function_space().mesh())
        data.append(lines)

    return data


def _handle_meshfunction(obj, **kwargs):
    data = []

    if obj.dim() == 3:
        # This is a cell function of dimension 3
        V = fe.FunctionSpace(obj.mesh(), "DG", 0)
        f = fe.Function(V)
        f.vector()[:] = obj.array()
        if kwargs.get("scatter", True):
            surf = _scatter_plot_function(f, **kwargs)
        else:
            surf = _surface_plot_function(f, intensitymode="cell", **kwargs)
    else:
        surf = _surface_plot_meshfunc(obj, **kwargs)
    data.append(surf)

    if kwargs.get("wireframe", True):
        lines = _wireframe_plot_mesh(obj.mesh(), **kwargs)
        data.append(lines)

    return data


def _handle_function_space(obj, **kwargs):
    data = []
    points = _plot_dofs(obj, **kwargs)
    data.append(points)

    if kwargs.get("wireframe", True):
        lines = _wireframe_plot_mesh(obj.mesh(), **kwargs)
        data.append(lines)
    return data


def _handle_dirichlet_bc(obj, **kwargs):
    data = []
    points = _plot_dirichlet_bc(obj, **kwargs)
    data.append(points)

    if kwargs.get("wireframe", True):
        lines = _wireframe_plot_mesh(obj.function_space().mesh(), **kwargs)
        data.append(lines)
    return data


def plot(
    obj,
    colorscale="inferno",
    wireframe=True,
    scatter=False,
    size=10,
    norm=False,
    name="f",
    color="gray",
    opacity=1.0,
    show_grid=False,
    size_frame=None,
    background=(242, 242, 242),
    normalize=False,
    component=None,
    showscale=True,
    show=True,
    filename=None,
):
    """Plot FEnICS object

    Parameters
    ----------
    obj : Mesh, Function. FunctionoSpace, MeshFunction, DirichleyBC
        FEnicS object to be plotted
    colorscale : str, optional
        The colorscale, by default "inferno"
    wireframe : bool, optional
        Whether you want to show the mesh in wirteframe, by default True
    scatter : bool, optional
        Plot function as scatter plot, by default False
    size : int, optional
        Size of scatter points, by default 10
    norm : bool, optional
        For vectors plot the norm as a surface, by default False
    name : str, optional
        Name to show up in legend, by default "f"
    color : str, optional
        Color to be plotted on the mesh, by default "gray"
    opacity : float, optional
        opacity of surface, by default 1.0
    show_grid : bool, optional
        Show x, y (and z) axis grid, by default False
    size_frame : [type], optional
        Size of plot, by default None
    background : tuple, optional
        Background of plot, by default (242, 242, 242)
    normalize : bool, optional
        For vectors, normalize then to have unit length, by default False
    component : [type], optional
        Plot a componenent (["Magnitude", "x", "y", "z"]) for vector, by default None
    showscale : bool, optional
        Show colorbar, by default True
    show : bool, optional
        Show figure, by default True
    filename : [type], optional
        Path to file where you want to save the figure, by default None

    Raises
    ------
    TypeError
        If object to be plotted is not recognized.
    """

    if isinstance(obj, fe.Mesh):
        handle = _handle_mesh

    elif isinstance(obj, fe.Function):
        handle = _handle_function

    elif isinstance(obj, fe.cpp.mesh.MeshFunctionSizet):
        handle = _handle_meshfunction

    elif isinstance(obj, fe.FunctionSpace):
        handle = _handle_function_space

    elif isinstance(obj, fe.DirichletBC):
        handle = _handle_dirichlet_bc

    else:
        raise TypeError(f"Cannot plot object of type {type(obj)}")

    data = handle(
        obj,
        scatter=scatter,
        colorscale=colorscale,
        norm=norm,
        normalize=normalize,
        size=size,
        size_frame=size_frame,
        component=component,
        opacity=opacity,
        show_grid=show_grid,
        color=color,
        wireframe=wireframe,
        showscale=showscale,
        name=name,
    )

    layout = go.Layout(
        scene_xaxis_visible=show_grid,
        scene_yaxis_visible=show_grid,
        scene_zaxis_visible=show_grid,
        paper_bgcolor="rgb" + str(background),
        margin=dict(l=80, r=80, t=50, b=50),
        scene=dict(aspectmode="data"),
    )

    if size_frame is not None:
        layout.update(width=size_frame[0], height=size_frame[1])

    fig = go.FigureWidget(data=data, layout=layout)
    fig.update_layout(hovermode="closest")

    if filename is not None:
        savefig(fig, filename)
    if show and _SHOW_PLOT:
        fig.show()
    return FEniCSPlotFig(fig)


class FEniCSPlotFig:
    def __init__(self, fig):
        self.figure = fig

    def add_plot(self, fig):
        data = list(self.figure.data) + list(fig.figure.data)
        self.figure = go.FigureWidget(data=data, layout=self.figure.layout)

    def show(self):
        if _SHOW_PLOT:
            self.figure.show()

    def save(self, filename):
        savefig(self.figure, filename)
