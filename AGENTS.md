# Project Agent Rules

These rules apply to all work on this project.

## Non-Negotiable Physics Rule

All implementations and tests must be faithful to known physics. Do not introduce artificial numerical hacks to make results look stable, smooth, or quiet.

Prohibited examples include:

- Locally or globally increasing dissipation only to suppress noise or instability.
- Changing viscosity, relaxation time, forcing, density scaling, or flow speed away from the stated physical configuration without explicitly framing it as a diagnostic and not a solution.
- Filtering, smoothing, clipping, or damping flow variables to hide grid-scale artifacts.
- Calling a run physically valid when it uses nonphysical parameter substitutions.

## Boundary-Condition Rule

Before implementing or changing a boundary condition, identify the physical conditions it must satisfy.

For stationary solid boundaries, preserve:

- No-penetration / kinematic condition.
- Correct mass flux condition.
- Appropriate no-slip or slip/shear condition for the modeled wall.

For moving solid boundaries, also preserve:

- Wall kinematics and velocity consistency.
- Momentum exchange with the moving wall.
- Dynamic consistency of the boundary forcing.

These conditions may not be compromised to reduce numerical noise.

## Geometry Interpretation Rule

Grid-scale interpretation of boundary geometry is allowed when it is physically meaningful and documented.

Allowed examples:

- Interpreting a grid-scale corner as a locally angled surface.
- Using SDF normals, cut-link fractions, or grid-scale geometric reconstruction to represent sub-cell wall location.
- Smoothing only the geometry interpretation at grid scale, not the flow field, and not as added dissipation.

## Diagnosis Rule

Assume noise and stability problems indicate incorrect physics implementation, numerical boundary-condition errors, or inaccurate local approximations until proven otherwise.

When a problem appears:

- First inspect physical consistency of boundary conditions, fluxes, kinematics, and momentum exchange.
- Then inspect numerical realization of those laws.
- Treat low-Re, high-viscosity, or damped tests only as diagnostics, never as fixes for a high-Re water configuration.

## Research Rule

For each configuration studied, review and apply the relevant physical laws and established numerical methods before implementation. Prefer documented LBM boundary methods and validation benchmarks over ad hoc changes.

## Review Habit

Review this file before making project changes, especially before modifying physics, boundary conditions, collision operators, forcing, viscosity, or test interpretation.
