# DFG 2D-3 cylinder mixed DWR

This package is the isolated mixed-system path for the nonlinear 2D-3
cylinder benchmark.  It deliberately does **not** modify
`nonstationary_dwr/` or make that scalar solver understand mixed systems.
The implemented residual, adjoint, nonlinear identity, and localisation
conventions are derived in [`MATHEMATICS.md`](MATHEMATICS.md).

## Completed boundary

### Stage A: benchmark and reusable primal

- Reuses `solve_static_primal` from the verified AS(2)/Alfeld-CG(1), dG(r)
  cylinder implementation.
- Encodes the R5 surface-traction mean-drag functional and its `T=8`
  reference value `1.6031368`.
- Evaluates the running goal from the saved mixed dG slab polynomials.
- Keeps the 2026 variational volume drag under an explicitly different name
  as a diagnostic; it is not the production goal.

### Stage B: mixed-DAE adapter

- Declares velocity as the only differential field and pressure as algebraic.
- Supplies velocity-only time mass and temporal jump actions.
- Reuses the verified Irksome mixed-stage repacking.
- Reuses the divergence-preserving velocity `P` and its L2 adjoint `P_star`.
- Constructs interface states with transferred velocity and zero pressure
  initial guess.

### Stage C: nonlinear mixed adjoint and symmetric DWR identity

- Solves the low AS/Alfeld adjoint and the CG4/CG2, time-enriched adjoint
  backward in physical time.
- Solves a CG4/CG2, time-enriched primal reconstruction without replacing the
  verified AS/Alfeld primal.
- Assembles all three nonlinear terms
  `0.5*rho(e_z) + 0.5*rho_star(e_u) + rho(z_h)`.
- Treats pressure as algebraic in both primal and adjoint temporal traces.
- Reports the enriched goal difference and the observed cubic remainder.

### Stage D: non-PU localisation and marking

For the production strict-linear/primal-residual path, localisation is the
direct three-part strong-residual split `cell volume + spatial stress facet +
dG temporal jump`.  Interior-facet values are shared equally by their two
cells.  This path has no space-time ridge entity and performs no auxiliary
recovery solves.  The tensor bubble/cone recovery described below is retained
only for the nonlinear symmetric estimator paths.

- Recovers the primal momentum, incompressibility, interior traction, outlet,
  and velocity-only dG-jump entities on each complete space-time slab using a
  common temporal polynomial basis.
- Uses the same recovered primal entities once with `Z+ - Pi_h Z+` and once
  with `Z_h`, so both the primal term and Galerkin correction are localised.
- Separately recovers the reverse-adjoint momentum, constraint, facet-flux,
  and velocity-only reverse-dG-jump entities, then weights them by `U+ - U_h`.
- Includes the additional velocity-only dG propagation-defect entity required
  because an Irksome propagated interface state is not the same object as the
  dG polynomial endpoint.
- Recovers the spatial-facet x temporal-interface mixed ridge for both the
  primal and reverse-adjoint equations.  The primal ridge is reused for the
  Galerkin correction.
- Uses the sequential tensor-product partition `volume -> spatial facet ->
  temporal facet -> mixed ridge`; each later solve subtracts every preceding
  recovered entity.
- Keeps the known cylinder drag derivative as an explicit boundary residual
  entity; this preserves its normal-derivative action, which a trace-only cone
  cannot represent.
- For AS derivative-node elements, `Pi_h` is a constrained componentwise L2
  projection rather than unavailable nodal interpolation.
- Retains the exact DG0 weak-cell decomposition only as an independent global
  closure diagnostic.  No weak-cell term enters the marking indicator.
- Applies one global space-time Doerfler operation and returns both the
  spatial union marker and the set of time slabs selected for bisection.
- Temporal marking supports three score sources
  (`--time-score-source`).  The original `marked_fraction` trigger counts
  spatially marked cells per slab.  `combined_indicator` reuses the existing
  production localisation and assigns the slab score
  `E_n=sum_K abs(eta_Kn)` without any extra adjoint, directional split, or
  localisation assembly.  `directional_time` instead ranks slabs by the
  localised temporal component `|rho(U_h)(z+ - I_k z+)|` of the R5 split and
  therefore requires `--directional-split-diagnostic`.  The two indicator
  sources require a score-driven strategy (`fixed_rate` or
  `slab_bulk_capped`); cell-fraction strategies ignore slab scores.

### Stage E: mixed slab-interface transfer

- Allows every time slab to own a different, causally nested Netgen mesh.
- Uses the verified AS/Alfeld velocity transfer `P=QI` and its literal
  velocity-L2 adjoint `P_star=I_star Q`; pressure is never transferred.
- Uses the exact identity on an unchanged slab mesh.  The Stokes projection is
  applied only at a genuine mesh change, preventing long-time accumulation of
  same-space projection perturbations.
- Supplies the same construction for the CG4/CG2 enriched trajectories, with
  a target-DG4 marked-mesh embedding and its assembled matrix transpose.
- Solves primal interfaces forward with `P`, adjoint interfaces backward with
  `P_star`, and resets the algebraic pressure component to zero.
- Enforces causal spatial nesting, so the implementation never guesses a
  fine-to-coarse AS interpolation.

### Stage F: complete adaptive loop

- Implements `SOLVE -> ESTIMATE -> LOCALISE -> MARK -> REFINE` in the
  problem-specific `CylinderAdaptiveSolver` without changing
  `nonstationary_dwr`.
- Applies one global Doerfler selection to all recovered nonlinear
  `abs(eta_Kn)`, then locally refines each causal slab mesh.
- Bisects a time slab when its selected-cell fraction exceeds the configured
  threshold; both children inherit the refined parent mesh.
- Rebuilds low/enriched spaces and `P/P_star` operators, then resolves the
  primal, enriched primal, and both adjoints on the new slab grid.
- Rejects a marking step if the tensor-recovery gap exceeds the configured
  global reliability gate (5% by default).

### Stage G: reproducible benchmark experiments

- Provides a production CLI around the Stage F solver without introducing a
  second discretisation or changing the mixed-system implementation.
- Writes the convergence history atomically after every completed outer
  iteration, together with a JSON run manifest and compressed slab indicator
  arrays.  Optional slab-wise VTK contains velocity, pressure, divergence,
  and signed/absolute recovered indicators.
- Reports the true space-time DoF counts, including all dG time modes, rather
  than only adding the spatial dimensions once per slab.
- Automatically uses the R5 surface mean-drag reference `1.6031368` only
  when `T=8`; shorter tests report no effectivity unless an explicit reference
  is supplied.
- Records `eta/true_error`, the localised effectivity, corrected goal, enriched
  goal, recovery acceptance, wall time, cell count, and time-step range.
- If a completed recovery fails the reliability gate, all diagnostics and
  indicators are saved before refinement is rejected and the manifest records
  the failure.

## Verification

Run from the repository root in the Firedrake environment:

```bash
OMP_NUM_THREADS=1 python -m cylinder_mixed_dwr.phase12_smoke \
  --levels 1 --nt 2 --T 0.125 --time-degree 1
```

The check requires every saved dG coefficient to remain mixed, verifies that
a pressure-only direction gives zero temporal-jump action, and checks that the
R5 and legacy drag conventions agree.

Stages C and D have their own short-horizon check:

```bash
OMP_NUM_THREADS=1 python -m cylinder_mixed_dwr.phase34_smoke \
  --levels 1 --nt 1 --T 0.0625 \
  --primal-time-degree 1 --enriched-time-degree 2 --theta 0.30
```

Stages E and F are checked together with:

```bash
OMP_NUM_THREADS=1 python -m cylinder_mixed_dwr.phase56_smoke \
  --levels 0 --nt 1 --T 0.0625 --max-it 2 --time-fraction 0.01
```

The Stage E gate constructs a genuinely marked child mesh, checks the low
and enriched `P/P_star` pairing, checks the transferred AS divergence, and
runs a coarse-to-fine primal plus fine-to-coarse reverse adjoints.  The Stage
F gate requires a nonempty global Doerfler set and a larger second-iteration
spatial grid.

The Stage G short production check is:

```bash
OMP_NUM_THREADS=1 python -m cylinder_mixed_dwr.production \
  --levels 0 --nt 1 --T 0.0625 --max-it 1 \
  --output-prefix /tmp/cylinder_stage_g_smoke --write-vtk --report-every 0
```

The first full R5 baseline should be run without VTK:

```bash
OMP_NUM_THREADS=1 python -m cylinder_mixed_dwr.production \
  --levels 1 --nt 128 --T 8 --max-it 1 \
  --output-prefix output/cylinder_mixed_dwr/r5_baseline
```

After accepting its recovery and resource diagnostics, the adaptive run is:

```bash
OMP_NUM_THREADS=1 python -m cylinder_mixed_dwr.production \
  --levels 1 --nt 128 --T 8 --max-it 3 --theta 0.30 \
  --time-fraction 0.05 \
  --output-prefix output/cylinder_mixed_dwr/r5_adaptive
```

Every production run now writes an immutable outer-grid checkpoint before
each solve.  By default these live in `OUTPUT_PREFIX_checkpoints/iter_NNNN`.
Each directory contains HDF5 audit copies of the unique slab meshes, the
completed convergence history, and a compressed marked-refinement lineage.
The lineage is necessary because a mesh loaded only from Firedrake HDF5 no
longer owns the Netgen object required for another local refinement.

To roll back to a saved grid, use a new output prefix so the resumed future is
a separate branch:

```bash
OMP_NUM_THREADS=1 python -m cylinder_mixed_dwr.production \
  --resume-from output/cylinder_mixed_dwr/r5_adaptive_checkpoints/iter_0001 \
  --max-it 3 --theta 0.30 --time-fraction 0.05 \
  --output-prefix output/cylinder_mixed_dwr/r5_adaptive_branch_01
```

`--max-it` is the absolute outer-iteration limit: resuming grid 1 with
`--max-it 3` solves iterations 1 and 2.  Structural data (geometry, initial
grid, viscosity, and primal/enriched polynomial degrees) come from the
checkpoint.  The current CLI controls only the future stopping and marking
policy.  Use `--checkpoint-root PATH` to choose another checkpoint location,
or `--no-checkpoint` for disposable smoke tests.

`--space-mode causal` retains slab-dependent spatial meshes.  This is useful
for transfer research but is not currently accepted for the long nonlinear
cylinder trajectory.  `--space-mode common` takes the union of the globally
marked spatial cells, refines one mesh, and shares it across every slab while
retaining the independently marked/bisected time grid.  On resume, an
explicit `--space-mode` rebranches the saved refinement marks with that policy.

## Verified boundary after Stage G

The genuine mixed tensor-product recovery is now the default.  On the
documented one-slab audit, DG2 volume densities, FB4 facet densities, and P2
time recovery give a combined recovery gap of `0.4766%`, below the required
`5%` gate.  The primal and Galerkin-correction gaps are approximately
`3.45e-5` and `3.44e-5`; the reverse-adjoint gap is approximately
`-3.60e-4`.  The independently assembled reverse-adjoint identity closes to
`2.22e-16`.

A two-slab fixed-mesh audit also passes: the combined gap is `0.523%` and the reverse
identity closes to `1.18e-16`.  This test exercises the internal temporal
interface and guards against replacing adjacent polynomial traces by
Irksome's propagated states.

The marked-mesh Stage E gate gives low-order relative `P/P_star` pairing
error `9.16e-15`, enriched pairing error `0`, and transferred AS divergence
`4.28e-12`.  The actual coarse-to-fine primal and both reverse adjoints also
solve successfully.

In the two-iteration Stage F audit, global Doerfler marking selects four of
223 initial space-time cells and local h-refinement produces 295 cells.  With
the 1% temporal threshold the marked slab is also bisected, producing two
slabs and 590 cell-slab pairs.  The second-iteration recovery gap is `1.27%`,
the weak closure gap is `1.32e-16`, and the reverse-adjoint identity gap is
`-2.15e-16`.

The Stage G success-path smoke writes a complete manifest, one-row history,
one compressed indicator archive, and a slab VTK dataset.  Its measured
space-time dimensions are 3042 for the AS/Alfeld dG(1) primal and 12957 for
the CG4/CG2 dG(2) enrichment.  An intentional recovery-gate failure also
retains its completed history and indicators and changes the manifest state
to `failed`.  These production-I/O checks were followed by the full `T=8`,
128-slab effectivity baseline reported below.

The first attempted full Stage G baseline exposed and fixed a long-time
interface defect: the initial slabwise implementation reapplied the Stokes
projection on all 127 interfaces even though the mesh was unchanged.  That
run produced a nonphysical mean drag `3.9741` and is explicitly marked
invalid.  After switching unchanged meshes to the exact identity, a complete
low-primal `T=8`, 128-slab audit gives `1.41667431104`, versus
`1.41667431694` from the verified persistent-stepper baseline.  Their
difference is `5.9e-9`; the polynomial drag extrema also agree with the
propagated endpoint extrema.

Historical variational-drag DWR histories are not reused after the goal
switch.  An R5-goal run must recompute the adjoint, estimator, localisation
and marking, although an existing primal trajectory can be reevaluated with
the R5 surface functional.

The checkpoint recovery test continuously solved a 223-cell grid, marked four
cells, refined to 295 cells, and then solved the next outer iteration.  A
separate process resumed exactly from `iter_0001`; all 42 non-timing history
fields in the resumed iteration matched the continuous run byte-for-byte.

The first checkpointed full adaptive continuation is deliberately retained as
a rejected diagnostic.  Iteration zero reproduced the accepted baseline and
globally marked 1065 space-time cells; nine slabs crossed the 5% time-marking
threshold.  The resulting grid has 137 slabs and between 892 and 1397 cells
per slab.  Iteration one completed all four primal/dual trajectories, but its
mean drag was the nonphysical value `3.99017`, its effectivity was `0.0360`,
and its recovery gap was `8.092%`, so the 5% gate correctly rejected it.  The
weak closure (`7.63e-16`) and reverse identity (`-8.49e-15`) remained closed.

The failure is upstream of localisation: the grid held 119 distinct mesh
objects and therefore applied 118 changed-mesh `QI` velocity projections,
although the causal refinement lineage contained only 24 distinct cumulative
mark sets.  Reconstructing an equivalent new Netgen mesh for every slab
prevented the exact-identity interface path from being used and accumulated a
large primal trace perturbation.  The next branch must reuse one mesh object
for identical cumulative causal marks (or use a common spatial mesh) before
the full adaptive solve is accepted.  Do not bypass the recovery gate on the
rejected result.

The causal-refinement correction now caches the result by parent-mesh identity
and the complete cumulative mark mask.  Its dedicated gate requires different
cumulative masks to produce different meshes and identical cumulative masks
to reuse the same Python object.  Replaying the rejected iteration-one lineage
with this policy gives 24 unique meshes and 23 true mesh-change interfaces,
instead of 119 meshes and 118 projections; all replayed meshes retain their
Netgen refinement data.  The rejected history remains unchanged as an audit
record, and the corrected computation is run as a separate branch from the
iteration-zero checkpoint.

A low-primal interface audit then showed why mesh-object reuse alone was not
enough: each of the 23 genuine coarse-to-fine `QI` transfers changed the
embedded velocity by roughly 14.6--15.5% in relative L2 norm, despite the
projected divergence remaining around `1e-14`.  The resulting causal branch
was again rejected (`J_h=1.88783`, recovery gap `91.96%`).  Thus exact
divergence preservation did not provide a dynamically faithful long-time
state transfer.

Rebranching the same Dörfler marks with `--space-mode common` removes all
cross-mesh traces while keeping the nine marked time-slab bisections.  The
137-slab grid uses one 1397-cell spatial mesh.  The full accepted iteration
gives `J_h=1.49987615`, enriched goal `J_plus=1.60382341`, global estimate
`eta=0.12741067`, global effectivity `1.23387`, and localised effectivity
`1.21605`.  The recovery gap is `1.4449%`, weak closure is `5.55e-17`, and
the reverse identity gap is `1.67e-16`.  Its endpoint drag range
`[-0.18504, 2.74519]` is physical and consistent with the fixed-grid audit.
This common-grid branch is the accepted production continuation; the causal
branches remain diagnostic failures.

Continuing that accepted branch to iteration 2 gives a common 3389-cell
spatial mesh and 162 time slabs (`dt` in `[0.015625, 0.0625]`).  The result is
`J_h=1.56444673`, enriched goal `J_plus=1.60448261`, global estimate
`eta=0.06705431`, global effectivity `1.73311`, and localised effectivity
`1.73723`.  The recovery gap decreases to `0.2377%`, weak closure is
`6.94e-17`, and the reverse identity gap is `3.05e-16`, so the iteration
passes the 5% recovery gate.  Final-iteration marking records 24,467 proposed
space--time cells and 53 time slabs without applying another refinement; the
saved indicators and immutable `iter_0002` checkpoint can therefore seed a
later continuation.

### Taylor--Hood slabwise continuation

The earlier causal AS/Alfeld branch used a cross-mesh operator described as
`P=QI`.  Here `mat_type="aij"` is only PETSc's sparse-matrix storage name; it
is unrelated to the Runge--Kutta/weak-time coefficients `a_ij`.  The operator
first embedded the old velocity in a target discontinuous space (`I`) and
then applied a target Stokes projection (`Q`) to impose both discrete
incompressibility and the new boundary trace.  The latter constraints do not
belong to the dG interface mass term.  On the 23 genuine mesh changes this
extra projection perturbed the physical velocity by 14.6--15.5% in relative
L2 norm, which accumulated in the nonlinear trajectory and produced the
rejected `J_h=1.88783` branch.

For the mixed DAE, pressure has no temporal mass.  The corrected interface is
therefore the target-space Riesz representation

``(P u_minus, v_plus)_L2 = (u_minus, v_plus)_L2``

for every target velocity test, with no pressure transfer and no additional
Stokes constraint on the trace representative.  The new-slab stages still
satisfy their Taylor--Hood divergence equation and strong boundary
conditions.  The reverse interface uses the exact L2 adjoint of this same
mass operator.  The legacy divergence-preserving transfer remains available
as `--interface-transfer divfree` for AS-only audit runs.  A physical
137-slab causal audit reduced the maximum interface correction from about
15% to `0.0769%`; its low-primal goal was `1.49673526`, close to the
corresponding common-grid result `1.49987615`.

The production Taylor--Hood branch uses low CG2/CG1 with temporal dG(1), and
CG4/CG2 with temporal dG(2) for enrichment.  It starts again from the
unrefined `iter_0000`, so its marks are not inherited from the AS estimator.
Iteration zero gives `J_h=1.48122411`, `J_plus=1.59710176`, and a `0.3362%`
localisation gap.  Its 622 global Dörfler cell--slab marks generate a causal
varying grid whose cell count grows from 892 to 1224 over the early slabs.
Iteration one gives `J_h=1.49288402`, `J_plus=1.59551738`, and a `0.1430%`
localisation gap; weak closure and the reverse identity are respectively
`4.88e-15` and `-2.22e-15`.  The primal goal error decreases from
`0.12191269` to `0.11025278`.  However the symmetric global estimator is
negative (`eta=-0.53452293`, effectivity `-4.84816`), so this branch validates
the slabwise transfer and localisation but not the accuracy of the current
nonlinear correction.  Its saved 1729 cell--slab and 15 time-slab candidates
must not seed another production iteration until that global-estimator sign
and remainder issue is resolved.
# Current Taylor--Hood estimator contract

The production path uses Taylor--Hood ``CG2/CG1 x dG(1)`` and the enriched
``CG3/CG2 x dG(2)`` problem.  Every adaptive iteration recomputes the low and
enriched primal and dual trajectories.  Its working estimator is the explicit
three-term nonlinear DWR approximation

``1/2 rho(Uh)(Zh+ - Zh) + 1/2 rho*(Uh,Zh)(Uh+ - Uh) + rho(Uh)(Zh)``.

The cubic Lagrangian remainder is currently excluded from both the global
estimator and its cell localization.  The dual weight is the direct enriched
minus numerical difference, not an enriched-minus-projection weight.  The
slabwise discrete-transpose adjoint and its stationarity gate must pass before
bubble recovery, global bulk Doerfler marking, or marked-cell-fraction time
bisection is allowed to run.
