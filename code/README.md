# Code accompanying the MSc dissertation

This directory contains the Firedrake/Irksome implementations used for the
scalar numerical experiments in the dissertation.  The Navier--Stokes code is
intentionally excluded from this snapshot.

## Contents

- `moving_hill_dwr_experiments/`: shared nonstationary DWR solver,
  hierarchical bubble--cone recovery, localisation, marking, refinement, and
  output routines used by the rotating-hill and parabolic p-Laplace studies.
- `automated_DWR/`: automated space--time DWR implementation used by the BBM
  study and by the heat-equation comparison tools.
- `nonstationary_dwr/`: supporting problem and solver interfaces imported by
  the scalar benchmark definitions.
- `problem_heat_moving_hill_experiment.py`: rotating-hill heat experiment.
- `advection_diffusion_1d_automated.py`: one-dimensional
  advection--diffusion experiment.
- `automated_DWR/bbm_example.py`: periodic solitary-wave BBM experiment.
- `problem_lshape_*.py`: L-shaped parabolic p-Laplace experiments.
- `heat_dwr_irksome.py` and `heat_dwr_mesh.py`: shared time-stepping and
  one-dimensional mesh helpers imported by the advection--diffusion driver.

## Environment

The experiments require a Firedrake installation with Irksome and the Python
packages installed by Firedrake.  Set `FIREDRAKE_PYTHON` to the Firedrake
Python executable before using the runner scripts, for example

```bash
export FIREDRAKE_PYTHON=/path/to/firedrake/bin/python
```

Representative entry-point commands are

```bash
"$FIREDRAKE_PYTHON" problem_heat_moving_hill_experiment.py --help
"$FIREDRAKE_PYTHON" advection_diffusion_1d_automated.py --help
"$FIREDRAKE_PYTHON" -m automated_DWR.bbm_example --help
"$FIREDRAKE_PYTHON" problem_lshape_j1_bubble_experiment.py --help
"$FIREDRAKE_PYTHON" problem_lshape_j2_terminal_average_experiment.py --help
```

Each entry point exposes the mesh, time-slab, marking, recovery, output and
stopping parameters through its command-line interface.  Generated CSV,
checkpoint, figure and VTK files are deliberately not included here.

## Status

This is the code used for the current dissertation results.  File layout and
documentation may be refined and reorganised further before the final release.
