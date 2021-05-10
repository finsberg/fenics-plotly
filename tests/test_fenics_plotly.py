import itertools as it
from pathlib import Path

import dolfin as df
import pytest

from fenics_plotly import plot


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def get_mesh(dim):
    if dim == 2:
        return df.UnitSquareMesh(2, 2)
    elif dim == 3:
        return df.UnitCubeMesh(2, 2, 2)


@pytest.mark.parametrize("dim, wireframe", it.product([2, 3], [True, False]))
def test_plot_mesh(dim, wireframe):
    mesh = get_mesh(dim)
    plot(mesh, wireframe=wireframe, show=False)


def test_plot_two_mesh():
    mesh1 = df.UnitCubeMesh(2, 2, 2)
    mesh2 = df.BoxMesh(
        df.MPI.comm_world, df.Point(0.0, 0.0, 0.0), df.Point(1.2, 0.5, 1.3), 3, 3, 3
    )
    fig = plot(mesh1, show=False)
    fig.add_plot(plot(mesh2, color="red", show=False))


@pytest.mark.parametrize(
    "dim, wireframe, scatter, degree",
    it.product([2, 3], [True, False], [True, False], [1, 2]),
)
def test_plot_scalar_cg_function(dim, wireframe, scatter, degree):
    mesh = get_mesh(dim)
    V = df.FunctionSpace(mesh, "CG", degree)
    p = df.Function(V)
    p.interpolate(df.Expression("sin(x[0])", degree=1))
    plot(p, scatter=scatter, wireframe=wireframe, show=False)


@pytest.mark.parametrize(
    "dim, wireframe, degree",
    it.product([2, 3], [True, False], [1, 2, 3]),
)
def test_plot_scalar_cg_function_space(dim, wireframe, degree):
    mesh = get_mesh(dim)
    V = df.FunctionSpace(mesh, "CG", degree)
    plot(V, wireframe=wireframe, show=False)


@pytest.mark.parametrize(
    "dim, wireframe, norm, normalize, degree, component",
    it.product(
        [2, 3],
        [True, False],
        [True, False],
        [True, False],
        [1, 2],
        [None, "magnitude", "x", "y", "z"],
    ),
)
def test_plot_vector_cg_function(dim, wireframe, norm, normalize, degree, component):
    mesh = get_mesh(dim)
    V = df.VectorFunctionSpace(mesh, "CG", degree)
    u = df.Function(V)
    if dim == 2:
        u.interpolate(df.Expression(("1 + x[0]*x[0]", "x[1]*x[1]"), degree=1))
        if component == "z":
            return
    else:
        u.interpolate(
            df.Expression(("1 + x[0]*x[0]", "x[1]*x[1]", "x[2]*x[2]"), degree=1)
        )

    plot(
        u,
        norm=norm,
        wireframe=wireframe,
        show=False,
        component=component,
        normalize=normalize,
    )


@pytest.mark.parametrize(
    "dim, wireframe, degree",
    it.product([2, 3], [True, False], [1, 2, 3]),
)
def test_plot_vector_cg_function_space(dim, wireframe, degree):
    mesh = get_mesh(dim)
    V = df.VectorFunctionSpace(mesh, "CG", degree)
    plot(V, wireframe=wireframe, show=False)


def test_plot_cellfunction():
    mesh = get_mesh(3)
    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    # Mark the first subdomain with value 1
    fixed = df.CompiledSubDomain("near(x[0], 0) && on_boundary")
    fixed_marker = 1
    fixed.mark(ffun, fixed_marker)

    # Mark the second subdomain with value 2
    free = df.CompiledSubDomain("near(x[0], 1) && on_boundary")
    free_marker = 2
    free.mark(ffun, free_marker)
    plot(ffun, show=False)


@pytest.mark.parametrize(
    "dim, wireframe",
    it.product([2, 3], [True, False]),
)
def test_plot_facetfunction(dim, wireframe):
    mesh = get_mesh(3)
    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    fixed = df.CompiledSubDomain("near(x[0], 0) && on_boundary")
    free = df.CompiledSubDomain("near(x[0], 1) && on_boundary")

    fixed_marker = 1
    fixed.mark(ffun, fixed_marker)

    # Mark the second subdomain with value 2

    free_marker = 2
    free.mark(ffun, free_marker)
    plot(ffun, show=False)


@pytest.mark.parametrize(
    "dim, wireframe",
    it.product([2, 3], [True, False]),
)
def test_plot_cell_function(dim, wireframe):
    mesh = get_mesh(3)
    ffun = df.MeshFunction("size_t", mesh, dim)
    ffun.set_all(1)

    # left = df.CompiledSubDomain("x[0] < 0.5")
    # left_marker = 1
    # Not needed since they are allready set to 1
    # left.mark(ffun, left_marker)

    # Mark the second subdomain with value 2
    right = df.CompiledSubDomain("x[0] >= 0.5")
    right_marker = 2
    right.mark(ffun, right_marker)
    plot(ffun, show=False)


@pytest.mark.parametrize(
    "dim, wireframe",
    it.product([2, 3], [True, False]),
)
def test_plot_dirichlet_bc(dim, wireframe):
    mesh = get_mesh(dim)

    # Mark the first subdomain with value 1
    fixed = df.CompiledSubDomain("near(x[0], 0) && on_boundary")

    V = df.VectorFunctionSpace(mesh, "CG", 2)
    zero = df.Constant((0.0, 0.0)) if dim == 2 else df.Constant((0.0, 0.0, 0.0))
    bc = df.DirichletBC(V, zero, fixed)

    plot(bc, wireframe=wireframe, show=False)


def test_save():
    mesh = get_mesh(3)
    filename = Path("mesh.html")
    if filename.is_file():
        filename.unlink()

    plot(mesh, filename=filename, show=False)
    assert filename.is_file()
    filename.unlink()


if __name__ == "__main__":
    test_save()
