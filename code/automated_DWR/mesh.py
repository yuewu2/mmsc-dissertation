"""Mesh construction that is independent from marking and refinement."""

from __future__ import annotations

from firedrake import Mesh, UnitSquareMesh
from firedrake.petsc import PETSc


def create_adaptive_unit_square_mesh(nx: int = 8, ny: int = 8) -> Mesh:
    r"""Create a Netgen unit-square mesh suitable for marked h-refinement.

    ``UnitSquareMesh`` is sufficient for a fixed-grid solve but this Firedrake
    installation only permits ``mesh.refine_marked_elements`` on a Netgen
    mesh.  Generic adaptive input files should therefore call this routine.
    """
    try:
        from netgen.geom2d import SplineGeometry

        geometry = SplineGeometry()
        geometry.AddRectangle(
            p1=(0.0, 0.0), p2=(1.0, 1.0), bc="boundary", leftdomain=1, rightdomain=0
        )
        geometry.SetMaterial(1, "domain")
        return Mesh(geometry.GenerateMesh(maxh=1.0 / max(nx, ny)))
    except ImportError:
        PETSc.Sys.Print(
            "WARNING: Netgen/ngsPETSc is unavailable; adaptive h-refinement will not work."
        )
        return UnitSquareMesh(int(nx), int(ny), diagonal="left")


def create_adaptive_l_shaped_mesh(n: int = 8) -> Mesh:
    r"""Create the Netgen L-domain ``(-1,1)^2 \setminus [0,1]\times[-1,0]``.

    The polygon is counter-clockwise, hence Netgen assigns material domain 1
    to its left-hand side.  The vertex ``(0,0)`` has interior angle
    ``3*pi/2`` and is the re-entrant corner used in the singular benchmark.
    Netgen is required because Firedrake's marked h-refiner operates on its
    meshes in this installation.
    """
    try:
        from netgen.geom2d import SplineGeometry

        geometry = SplineGeometry()
        points = [
            geometry.AppendPoint(-1.0, -1.0),
            geometry.AppendPoint(0.0, -1.0),
            geometry.AppendPoint(0.0, 0.0),
            geometry.AppendPoint(1.0, 0.0),
            geometry.AppendPoint(1.0, 1.0),
            geometry.AppendPoint(-1.0, 1.0),
        ]
        for start, end in zip(points, points[1:] + points[:1]):
            geometry.Append(["line", start, end], leftdomain=1, rightdomain=0, bc="boundary")
        geometry.SetMaterial(1, "l_domain")
        return Mesh(geometry.GenerateMesh(maxh=2.0 / max(1, int(n))))
    except ImportError:
        raise RuntimeError(
            "The L-shaped adaptive benchmark requires Netgen. "
            "Activate the Firedrake environment that provides netgen.geom2d."
        )


def create_box_fitted_mesh(
    nx: int = 8,
    ny: int = 8,
    box_left: float = 0.25,
    box_right: float = 0.75,
) -> Mesh:
    """Create a triangular unit-square mesh fitted to ``[a,b]^2``.

    Fitting the terminal-goal discontinuity to facets avoids an avoidable
    quadrature error in ``J(u)=integral_[a,b]^2 u(T) dx``.  The regular
    ``UnitSquareMesh`` fallback keeps the example executable without Netgen.
    """
    try:
        from netgen.geom2d import SplineGeometry

        geometry = SplineGeometry()
        geometry.AddRectangle(
            p1=(0.0, 0.0), p2=(1.0, 1.0), bc="boundary", leftdomain=1, rightdomain=0
        )
        geometry.AddRectangle(
            p1=(box_left, box_left), p2=(box_right, box_right),
            bc="box_interface", leftdomain=2, rightdomain=1,
        )
        geometry.SetMaterial(1, "outer")
        geometry.SetMaterial(2, "inner")
        return Mesh(geometry.GenerateMesh(maxh=1.0 / max(nx, ny)))
    except ImportError:
        PETSc.Sys.Print(
            "WARNING: Netgen/ngsPETSc is unavailable; falling back to UnitSquareMesh."
        )
        if nx % 4 or ny % 4:
            PETSc.Sys.Print("WARNING: nx and ny divisible by four fit the default goal box exactly.")
        return UnitSquareMesh(int(nx), int(ny), diagonal="left")
