# Equations and method

## Equations

```
d/dt ∫ U dV + ∮ (F(U) - F_v(U, ∇U)) · n dS = 0,     U = (ρ, ρu, ρv, E)
F · n   = ( ρVn,  ρu Vn + p nx,  ρv Vn + p ny,  (E + p) Vn ),   Vn = u nx + v ny
F_v · n = ( 0,  (τ·n)x,  (τ·n)y,  (τ·V)·n + k ∇T·n )
τ = μ(∇V + ∇Vᵀ - ⅔(∇·V) I),   k = μ / (Pr (γ-1)),   T = a² = γp/ρ
```

Variables are scaled by the freestream density, the freestream speed of
sound and the chord: the freestream is `ρ = 1, (u, v) = M∞ (cos α, sin α), p = 1/γ`, and
`μ = M∞/Re`. Viscosity is constant or follows Sutherland's law,
`μ/μ∞ = T^{3/2} (1+s)/(T+s)`, `s = S/T∞`.

With Spalart-Allmaras (SA-noft2 with SA-neg, as defined by NASA's Turbulence
Modeling Resource) a fifth equation is solved for `ρν̃`, stored over `ν∞` so
that it is O(1) beside the flow variables; the stresses and heat flux use
`μ + μ_t` and `μ/Pr + μ_t/Pr_t`, `μ_t = ρν̃ f_v1`, `Pr_t = 0.9`:

```
∂(ρν̃)/∂t + ∇·(ρVν̃) = ρ c_b1 S̃ ν̃ − ρ c_w1 f_w (ν̃/d)² + (1/σ)[∇·((μ + ρν̃ f_n)∇ν̃) + c_b2 ρ|∇ν̃|²]
```

## Discretization

- **Cell-centred finite volume** on structured blocks. Every grid, single- or
  multi-block, is packed into one array (the *atlas*) with two ghost layers
  per block; ghost cells hold the neighbouring block's cells across an
  interface and the boundary condition's state on a patch.
- **Convective flux**: MUSCL reconstruction of primitive variables from raw
  index differences, limited by `minmod`, `van_albada` or none; then HLLC
  (default) or Roe at each face.
- **Viscous flux**: face gradients by the diamond construction, exact for a
  linear field on any quadrilateral.
- **Boundary conditions** by patch: slip and adiabatic no-slip walls,
  symmetry, a characteristic far field (Riemann invariants, optionally
  against the freestream plus the compressible point vortex of the current
  lift), total-condition inflow and static-pressure outflow.

## Iteration to steady state

- **Explicit**: Jameson's multistage Runge-Kutta with local time steps.
- **Implicit**: pseudo-transient Newton-Krylov. Each step solves
  `(V/Δt + ∂R/∂U) ΔU = -R` by right-preconditioned restarted GMRES on
  Jacobian-vector products from `jax.linearize`; by default the operator is
  the first-order residual's Jacobian (defect correction). The CFL grows by
  the residual reduction when the linear solve succeeded and is cut back after
  a rejected step. The preconditioner solves block-tridiagonal systems exactly
  along grid lines in `j`, through a C-grid's wake cut, with a first-order
  upwind approximate Jacobian (Mavriplis, AIAA J. 37 (1999)).

Both paths drive the same residual through the same on-device loop, so they
share their converged solutions. A polar maps the step over the freestream
and runs every angle at once, split over GPUs.

## Grids

Generated round one airfoil by hyperbolic marching (Steger and Chaussee):
O-grids and C-grids, Vinokur spacing along the surface and geometric growth
off it. Or read from Plot3D or CGNS (HDF5), with block interfaces found from
coincident faces and boundary patches named from the file or from geometric
rules. Every grid passes a quality report before it runs.
