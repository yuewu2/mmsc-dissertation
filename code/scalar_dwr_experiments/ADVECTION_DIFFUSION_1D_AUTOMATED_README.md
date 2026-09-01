# 1D moving-pulse automated bubble-cone example

## What this program solves

The program `advection_diffusion_1d_automated.py` solves

\[
u_t-\varepsilon u_{xx}+b u_x=f,
\qquad (x,t)\in(0,1)\times(0,T),
\]

with

\[
u(0,t)=u(1,t)=0.
\]

The manufactured exact solution is

\[
u(x,t)=x(1-x)\exp\!\left[-\beta(x-x_0-bt)^2\right].
\]

With the defaults

\[
x_0=0.2,\qquad b=0.6,\qquad \beta=100,\qquad T=1,
\]

the centre of the pulse follows

\[
x_{\mathrm{centre}}(t)=0.2+0.6t.
\]

It therefore travels diagonally across an \(x\)-\(t\) picture.

The terminal quantity of interest is

\[
J(u)=\int_{0.75}^{0.90}u(x,T)\,dx.
\]

## Discrete spaces

```text
primal:                  CG1 in x x Irksome DG0 in t
numerical dual:          CG1 in x x Irksome DG0 in t
enriched dual:           CG2 in x x Irksome DG1 in t
cell residual recovery:  DG1 in x x P1 in reference time
endpoint recovery:       broken DG1 endpoint traces x P1 in time
temporal-interface:      DG1 in x
```

The DWR weight is

\[
z^\star=z_{\mathrm{enriched}}-z_{\mathrm{numerical}}.
\]

The terminal dual data are computed by the discrete \(L^2\) projection

\[
(z_h(T),v_h)=(\chi_{[0.75,0.90]},v_h)
\qquad\forall v_h,
\]

rather than by simply assigning zero or one at finite-element nodes.

## Why there is no Firedrake FB element in this 1D code

In two spatial dimensions a facet is an edge, so the heat example uses a
facet-bubble/cone element. In one spatial dimension a facet is only a point.
There is no direction along that point and therefore no nonconstant
polynomial on it.

On an interval \(K=[x_i,x_{i+1}]\), the two endpoint cone functions are just

\[
C_L(x)=\frac{x_{i+1}-x}{x_{i+1}-x_i},
\qquad
C_R(x)=\frac{x-x_i}{x_{i+1}-x_i}.
\]

They satisfy

\[
C_L(x_i)=1,\quad C_L(x_{i+1})=0,
\]

\[
C_R(x_i)=0,\quad C_R(x_{i+1})=1.
\]

A broken DG1 space has exactly these two independent endpoint traces on each
interval. It therefore supplies the one-dimensional cone tests directly.

## Run the example

Inside the Firedrake environment:

```bash
python advection_diffusion_1d_automated.py
```

A shorter demonstration is:

```bash
python advection_diffusion_1d_automated.py \
  --nx 16 \
  --nt 8 \
  --max-it 3 \
  --tolerance 1e-6 \
  --output-prefix output/advection_diffusion_1d/automated
```

Use `--no-vtk` when only numerical diagnostics and CSV history are needed.

## Main output files

```text
output/advection_diffusion_1d/automated_history.csv
output/advection_diffusion_1d/automated_iterations.pvd
output/advection_diffusion_1d/automated_spacetime_iterations.pvd
output/advection_diffusion_1d/automated_spacetime_iter_0.vtu
output/advection_diffusion_1d/automated_spacetime_iter_0.csv
...
```

`automated_iterations.pvd` shows the ordinary one-dimensional spatial mesh.

`automated_spacetime_iterations.pvd` shows the full two-dimensional
space-time grid. Its horizontal coordinate is \(x\), its vertical coordinate
is \(t\), and every quadrilateral is one slab-local tensor-product cell

\[
K\times I_n.
\]

## Genuine slabwise spatial refinement

This driver does **not** take a union of marked spatial cells over all time
slabs.  It stores one spatial partition \(\mathcal T_n\) for each interval
\(I_n\).  After Dörfler marking, it performs

\[
\mathcal T_n^{\mathrm{new}}
=\operatorname{bisect}\bigl(
  \mathcal T_n,\{K:(K,I_n)\in\mathcal M\}
 \bigr).
\]

Consequently, a marked cell near the end of time produces a small rectangle
only near the end of the \(x\)-\(t\) picture; it does not create a narrow
vertical strip through earlier slabs.  If a slab also meets the time-marking
threshold, both child slabs inherit its newly refined spatial partition.

At each interface the forward state is transferred by an assembled nodal
interpolation matrix,

\[
\boldsymbol u_{n+1}(t_n^+)=P_n\boldsymbol u_n(t_n^-),
\]

and the reversed numerical dual uses its exact discrete \(L^2\)-adjoint,

\[
\boldsymbol z_n(t_n^-)
=M_n^{-1}P_n^T M_{n+1}\boldsymbol z_{n+1}(t_n^+).
\]

The left temporal residual on slab \(I_{n+1}\) contains the same forward
transfer through \(u_{n+1}(t_n^+)-P_nu_n(t_n^-)\).  Thus the primal
interface residual and the dual mesh-transfer operation are consistent.  The
implementation is deliberately serial 1D; extending it to parallel 2D/3D
requires distributed transfer matrices and scalable mass solves.

## ParaView instructions for the x-t picture

1. Open `automated_spacetime_iterations.pvd` and click `Apply`.
2. Select the `2D` interaction button.
3. Set `Representation` to `Surface With Edges`.
4. Select `eta_abs` in `Coloring`.
5. Use the animation next/previous buttons to move between adaptive
   iterations.

The collection time values `0,1,2,...` are adaptive iteration numbers, not
physical time. Physical time is already the vertical coordinate of the plot.

The available cell arrays are:

| Array | Meaning |
|---|---|
| `eta_signed` | signed local \(\eta_{K,n}\) |
| `eta_abs` | \(|\eta_{K,n}|\), recommended colour field |
| `eta_volume` | recovered cell-interior contribution |
| `eta_endpoint` | recovered spatial endpoint/cone contribution |
| `eta_temporal` | recovered temporal-interface contribution |
| `marked` | 1 if \(K\times I_n\) is selected, otherwise 0 |
| `h_x` | spatial interval length of the current slab-local mesh \(\mathcal T_n\) |
| `k_t` | time-slab length |

To show only marked space-time cells:

1. select the space-time source;
2. choose `Filters -> Threshold`;
3. choose the cell array `marked`;
4. retain values above `0.5`;
5. colour the threshold result red;
6. leave the original grid visible in grey with lower opacity.

The marked cells in adaptive iteration \(i\) generate the refined grid in
iteration \(i+1\). The final recorded iteration has `marked=0`, because no
subsequent refinement is performed after the final solve.

## Numerical checks printed by the program

The signed global effectivity index is

\[
I_{\mathrm{eff}}
=\frac{\eta_{\mathrm{global}}}{J(u)-J(U_h)}.
\]

The localisation consistency index is

\[
I_{\mathrm{loc}}
=\frac{\sum_{n,K}\eta_{K,n}}{\eta_{\mathrm{global}}}.
\]

The first measures the quality of the enriched dual DWR estimate. The second
measures whether automated recovery has successfully redistributed that same
global residual into space-time cells. A value close to one is desirable for
both, but they test different things.
