# Mixed nonlinear DWR formulation

This note records the conventions implemented by Stages C and D.  It is
independent of the continuous partition-of-unity localisation used in R5.

## 1. Primal residual and DAE structure

Let `U=(u,p)` and `Phi=(v,q)`.  On one open time slab the weak operator is

```text
A^V(U)(Phi) =
    (d_t u,v)
  + ((grad u)u,v)
  + nu(grad u,grad v)
  - (p,div v)
  - (div u,q).
```

At the left dG interface it also contains

```text
A^T_n(U)(Phi) = (u(t_{n-1}+) - P_n u(t_{n-1}-), v(t_{n-1}+)).
```

There is no `d_t p`, pressure mass, or pressure jump.  Thus the time mass is
`M=diag(M_u,0)`.  On a changing mesh `P_n` and its mass adjoint `P_n*` act on
velocity only.

The residual convention used below is `rho(U)(Phi)=-A(U)(Phi)`.

## 2. Goal and adjoint

The production goal is the R5 surface-traction mean drag.  If `n_f` denotes
Firedrake's normal pointing out of the fluid (into the circular hole), then
R5's obstacle-outward normal is `-n_f` and

```text
J(U) = -(20/T) integral_0^T integral_Gamma_circle
       (-p I + nu grad u) n_f . e_1 ds dt.
```

For `delta U=(delta u,delta p)`,

```text
J'(U)(delta U) =
  -(20/T) integral_0^T integral_Gamma_circle
  (-delta p I + nu grad delta u) n_f . e_1 ds dt.
```

This functional is linear and has no terminal or temporal-endpoint term.
The 2026 variational volume drag remains available under an explicitly
different function name for diagnostics only; it is never used to build the
R5 adjoint.

The linearisation of the slab operator is

```text
A'^V(U)(delta U,Z) =
    (d_t delta u,z_u)
  + ((grad delta u)u + (grad u)delta u,z_u)
  + nu(grad delta u,grad z_u)
  - (delta p,div z_u)
  - (div delta u,z_p).
```

After integration by parts in time, the physical adjoint runs backward.  The
code sets `tau=T-t`, so Irksome again advances forward and `Dt(Z)` represents
`-d_t Z`.  The adjoint equation is

```text
A'(U_h)(delta U,Z_h) - J'(U_h)(delta U) = 0.
```

The time-averaged surface functional has no terminal value, so the reverse
adjoint starts from zero at `T`.

## 3. Enrichment

The numerical pair is the verified AS(2)/Alfeld-CG(1) pair with the primal dG
degree.  The enriched primal and adjoint use CG4/CG2 and one higher dG degree
on the same spatial mesh and time partition.  The enriched primal is only an
estimator reconstruction; it does not replace the verified primal trajectory.

No nested-space injection is assumed.  For the goal-weight sensitivity used
in Stage D,

```text
z_star = Z+ - Pi_h Z+.
```

The AS velocity element has derivative nodes and Firedrake cannot construct a
nodal CG4-to-AS interpolant.  `Pi_h` is therefore a componentwise L2
projection into AS/Alfeld, followed by the homogeneous adjoint velocity trace.
The actual low adjoint `Z_h` remains in the adjoint-residual and correction
terms, matching the existing solver's `enriched_minus_interpolant` convention.

## 4. Symmetric nonlinear identity

With the Lagrangian `L(U,Z)=J(U)-A(U)(Z)`, define

```text
e_U = U+ - U_h,
e_Z = Z+ - Pi_h Z+,
rho*(U_h,Z_h)(e_U) = J'(U_h)(e_U) - A'(U_h)(e_U,Z_h).
```

The requested computable estimator is

```text
eta = 0.5 rho(U_h)(e_Z)
    + 0.5 rho*(U_h,Z_h)(e_U)
    + rho(U_h)(Z_h).
```

The correction has a plus sign under the convention `rho=-A`.  It vanishes
up to algebraic and quadrature error when the low primal and low adjoint obey
Galerkin orthogonality.  The code reports

```text
R3_observed = J(U+) - J(U_h) - eta.
```

Because the selected `e_Z` uses `Pi_h Z+` while the other two terms use the
independently solved `Z_h`, this is the solver's hierarchical weight-
sensitivity approximation, not the literal endpoint difference
`Z+-Z_h`.  The reported enriched goal gap is therefore a convergence
diagnostic, not a strict cubic remainder unless these low representations
become asymptotically equivalent.

## 5. Non-PU localisation

On every slab, let `s in [0,1]` be physical reference time.  The primal
residual is reconstructed once using:

1. one mixed block of cell-bubble momentum and continuity densities in
   `DG(2) x P2(s)`;
2. broken `FB(4) x P2(s)` facet cones for the spatial traction complement,
   after subtracting the mixed volume block;
3. a velocity-only temporal-facet density obtained with the temporal cone
   `1-s`, after subtracting the volume density;
4. a velocity-only spatial-facet x left-time-interface ridge, after
   subtracting volume, spatial-facet, and temporal-facet densities.

That recovered residual is evaluated both on `Z+-Pi_h Z+` and on `Z_h`, so
the Galerkin correction is included without DG0 weak-cell bookkeeping.

The reverse-adjoint residual uses the same tensor construction in reverse
reference time `r=1-s`.  Its temporal cone is `1-r`, so its time facet is the
physical-right trace of the slab.  The cylinder drag derivative is retained
as an exact known boundary residual entity because it acts on the normal
derivative of the no-slip primal error.

For dG time stepping, the polynomial endpoint and the propagated state must
not be identified.  On slab `n`, write `e_n^R` for the primal-error polynomial
right trace, `e_n^I` for its propagated incoming trace, and `z_n^R,z_n^L` for
the two physical dual polynomial traces.  Temporal integration by parts and
the original primal dG jump give the exact slab-local endpoint split

```text
-(e_n^R, z_n^R) + (e_n^I, z_n^L).
```

Both endpoint densities are recovered with cell bubbles.  This form is also
valid when neighbouring slabs have different meshes: `e_n^I` already contains
the forward `P` transfer, while the backward adjoint solve contains `P*`; no
cross-mesh subtraction is needed inside the local indicator.

Both equations also contain the codimension-two mixed ridge left after the
volume, spatial-facet, and temporal-cell projections are subtracted.  For the
primal it lies on `spatial facet x t_(n-1)+`; for the reverse adjoint it lies
on `spatial facet x t_n-`.  It is velocity-valued because pressure has no time
jump.  The primal ridge is evaluated both on `Z+-Pi_h Z+` and on `Z_h`.

This ridge statement applies only to the nonlinear symmetric tensor-recovery
path.  The strict-linear production path instead integrates the primal weak
residual by parts elementwise and uses exactly three local entities: strong
cell volume, spatial stress facet (shared equally by adjacent cells), and dG
temporal jump.  That algebraic split contains no mixed space-time ridge and
requires no recovery projection solves.

The hierarchical marking value is therefore

```text
eta_Kn = 0.5 eta_primal,bubble-cone_Kn
       + 0.5 eta_adjoint,bubble-cone_Kn
       +     eta_correction,bubble-cone_Kn.
```

An additional exact DG0 weak-cell split of the global terms is retained only
as a closure test.  Slab-level audits also compare the recovered primal and
correction actions with their direct weak actions, and the recovered adjoint
action with the independently assembled reverse-time weak action.  None of
these audit partitions contributes to marking.

Finally one Dörfler operation is applied to all `abs(eta_Kn)` at once.  Its
output consists of slab-local cell masks, their spatial union, and the slabs
whose marked-cell fraction passes the requested time-bisection threshold.

## 6. Changing-mesh mixed interfaces

Let `V_n` be the velocity space on slab `I_n`.  The forward differential
trace is

```text
u_n^I = P_n u_(n-1)^R,       P_n = Q_n I_n : V_(n-1) -> V_n,
```

where `I_n` embeds the old velocity on the causally nested target mesh and
`Q_n` is the target Stokes-L2 projection.  `Q_n` restores the discrete
incompressibility constraint and imposes the physical inlet/wall trace.  The
pressure has no differential trace and is set only to a zero algebraic
initial guess for the nonlinear slab solve.

If `V_(n-1)=V_n` is literally the same discrete space, the interface operator
is the exact identity and no Stokes projection is solved.  Applying `Q_n`
repeatedly on an unchanged mesh is algebraically unnecessary and its small
solver/projection perturbations accumulate over long trajectories.  Pressure
may be copied in this identity branch as an algebraic solver initial guess;
it remains absent from the mass and dG jump forms.

The reverse interface uses the mass adjoint of exactly this linearized
homogeneous-trace transfer:

```text
(P_n v, z)_V_n = (v, P_n_star z)_V_(n-1),
P_n_star = I_n_star Q_n.
```

On separately marked meshes, `I_n` is assembled into a broken target DG
embedding space and `I_n_star` is the literal transpose of that matrix.
Consequently the adjoint does not use an independently chosen reverse
interpolant.  The same construction is used for enriched CG4 velocity with a
DG4 embedding and target CG2 Stokes multiplier.  Both transfers act on
velocity only.

## 7. Adaptive outer iteration

For adaptive iteration `ell`, the implementation performs

```text
(U_h, U_plus, Z_h, Z_plus)
  -> global symmetric eta
  -> recovered signed eta_Kn
  -> one global Doerfler set M
  -> causal local h-refinement and selected time bisection.
```

If `C_n` is the spatial marker used on slab `n`, causal nesting is maintained
by

```text
C_1 = M_1,
C_n = M_n union inherit_(n-1 -> n)(C_(n-1)).
```

The inheritance maps the already marked physical parent regions onto every
later nested mesh.  This gives only coarse-to-fine forward interfaces; the
verified `P/P_star` construction is therefore sufficient.  A marked time
slab is bisected and both children use its newly refined mesh.  All spaces,
interface operators, four trajectories, the global estimator, and the
tensor recovery are rebuilt on the resulting grid before the next marking
decision.  Marking is rejected if the recovered local sum differs from the
global estimator by more than the configured reliability threshold.

## 8. Reference error and Stage G diagnostics

For the full DFG 2D-3 horizon `T=8`, let

```text
J_ref = 1.6031368,
e_J   = J_ref - J(U_h).
```

The production history reports

```text
I_eff_global = eta_global / e_J,
I_eff_local  = eta_local_sum / e_J,
J_corrected  = J(U_h) + eta_global,
J_enriched   = J(U_h) + (J(U_plus)-J(U_h)).
```

The reference is not automatically applied to a shorter horizon, because
the time-mean drag is then a different quantity of interest.  These
effectivities are diagnostics, not unconditional reliability statements:
the estimator uses recovered enriched sensitivities and its interpretation
still depends on enrichment and saturation quality.

For a spatial mixed dimension `dim(X_n)` and dG degree `r`, the reported
primal space-time algebraic size is

```text
N_st = sum_n (r+1) dim(X_n).
```

This counts every time coefficient on every independently adapted slab.  It
is distinct from both the number of spatial cells and the sum of the spatial
dimensions with the temporal multiplicity omitted.
