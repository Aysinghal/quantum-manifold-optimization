"""
Riemannian manifold abstractions for variational-parameter spaces.

Each ``Manifold`` defines:

  - ``project(theta, vec)``: project an ambient vector ``vec in R^d`` onto the
    tangent space ``T_theta M``.
  - ``inner(theta, u, v)``: manifold inner product on ``T_theta M``. Currently
    informational only -- the QNG-family training path uses the Fubini-Study
    metric externally; this method is here for completeness and Phase 2 use.
  - ``retract(theta, vec)``: take an ambient step ``vec`` from ``theta`` and
    return a new point that lies on ``M``. For a flat manifold this is plain
    addition; for the torus it wraps mod ``2*pi``; for the sphere it is the
    geodesic exponential map.

Phase 1 ships ``Euclidean`` and ``Torus``; ``Sphere`` lands in Phase 2;
``Hyperbolic`` (Lorentz model) lands in Phase 2.5b. Phase 2.5b's R-QNG and
projected baselines on the hyperboloid mirror the Phase-2 sphere lineup so
sphere vs. hyperbolic vs. flat is an apples-to-apples cross-manifold
comparison; see docs/HYPERBOLIC_SCOPING.md for the mechanism story.
"""

from abc import ABC, abstractmethod

import numpy as np


class Manifold(ABC):
    """Abstract base class for parameter-space manifolds."""

    name = "manifold"

    @abstractmethod
    def project(self, theta, vec):
        """Project ``vec`` into the tangent space ``T_theta M``."""

    @abstractmethod
    def retract(self, theta, vec):
        """Take a step ``vec`` from ``theta``, returning a new point on ``M``."""

    def inner(self, theta, u, v):
        """Default Euclidean inner product on the tangent space."""
        return float(np.dot(np.asarray(u).flatten(), np.asarray(v).flatten()))

    def parallel_transport(self, theta_old, theta_new, vec):
        """Map tangent ``vec`` at ``theta_old`` into ``T_{theta_new} M``.

        Flat manifolds are translation-invariant, so the identity transport
        is exact there. Curved manifolds should override this with a closed
        form or numerically stable approximation.
        """
        return np.asarray(vec)


class Euclidean(Manifold):
    """Flat ``R^d`` -- the default for standard optimizers."""

    name = "euclidean"

    def project(self, theta, vec):
        return np.asarray(vec)

    def retract(self, theta, vec):
        return np.asarray(theta) + np.asarray(vec)


class Torus(Manifold):
    """``T^d = (S^1)^d``. Locally flat, so projection is the identity; the
    only difference from ``Euclidean`` is that ``retract`` wraps the result
    into ``[0, 2*pi)^d``.

    Quantum rotation gates satisfy ``R(theta) = R(theta + 2*pi)``, so wrapping
    is physically a no-op -- but it keeps ``||theta||`` bounded and prevents
    momentum / metric-tensor estimation from drifting in overparameterized
    circuits where the optimizer otherwise has nothing constraining ``theta``.
    """

    name = "torus"
    period = 2.0 * np.pi

    def project(self, theta, vec):
        return np.asarray(vec)

    def retract(self, theta, vec):
        return np.mod(np.asarray(theta) + np.asarray(vec), self.period)


class Sphere(Manifold):
    """Sphere ``{ theta in R^d : ||theta|| = R }`` of *implicit* radius R.

    Design choices:

    1. **One global sphere over all flattened params** (vs per-layer / per-row
       spheres). Cleanest math, and matches the Phase-2 rationale ("force
       dials to compete globally"). Per-layer sphere is a follow-up.

    2. **Radius is implicit**: instead of storing R, we read it off the
       current point as ``R = ||theta||``. Both ``project`` and ``retract``
       compute R lazily, so as long as ``init_on_manifold`` normalizes
       ``theta_init`` to whatever R the experiment wants, every subsequent
       step will preserve it. This avoids needing a per-task Sphere
       instance keyed on parameter dimension.

    3. **Natural-scale init**: the canonical R for a VQA circuit with
       parameters drawn from ``U(0, 2*pi)`` is the expected init norm,
       ``R* = 2*pi * sqrt(d / 3)`` (since ``E[theta_i^2] = 4*pi^2/3``).
       This is the "shell where flat optimizers naturally drift to" and
       gives sphere optimizers a fighting chance against flat baselines
       (the unit sphere is much smaller than any usable VQA configuration
       and forces sphere optimizers to whisper while flat ones shout).
       The default is set in ``models.init_on_manifold``.

    Geometry on a radius-R sphere (generalizing the unit-sphere formulas):
      Tangent at ``theta``:   ``T_theta S = { v : <v, theta> = 0 }``
      Project ambient ``g``:  ``Proj(g) = g - <g, theta>/R^2 * theta``
      Geodesic exp:           ``exp_theta(v) = cos(|v|/R) theta
                                              + R sin(|v|/R) v/|v|``

    The unit-sphere formulas (R=1) drop out as a special case. Returns are
    reshaped back to the input's shape so the rest of the training loop
    (which expects ``params.shape``) stays unchanged.
    """

    name = "sphere"

    def project(self, theta, vec):
        theta_flat = np.asarray(theta).flatten()
        vec_flat = np.asarray(vec).flatten()
        R2 = float(np.dot(theta_flat, theta_flat))
        if R2 < 1e-24:
            return np.zeros_like(vec_flat).reshape(np.shape(vec))
        proj = vec_flat - (np.dot(vec_flat, theta_flat) / R2) * theta_flat
        return proj.reshape(np.shape(vec))

    def retract(self, theta, vec):
        theta_flat = np.asarray(theta).flatten()
        vec_flat = np.asarray(vec).flatten()
        R = float(np.linalg.norm(theta_flat))
        v_norm = float(np.linalg.norm(vec_flat))
        if R < 1e-12 or v_norm < 1e-12:
            return np.array(theta)
        # Geodesic exp on a radius-R sphere; cos(|v|/R) keeps the radial
        # component, sin(|v|/R) traverses the tangent. Re-normalize to
        # exactly R after the trig to kill float drift across many steps.
        ang = v_norm / R
        new = np.cos(ang) * theta_flat + R * np.sin(ang) * vec_flat / v_norm
        new = new * (R / np.linalg.norm(new))
        return new.reshape(np.shape(theta))


class Hyperbolic(Manifold):
    """Hyperbolic ``H^d`` via the Lorentz / hyperboloid model with constant
    negative curvature ``c``.

    Coordinate convention -- "implicit ``x_0``":

    The d-dimensional flat parameter array stores the **spatial coordinates**
    ``x_s = (x_1, ..., x_d)`` of a Lorentz point ``x = (x_0, x_s) in R^{d+1}``.
    The temporal coordinate ``x_0 = sqrt(1/c + ||x_s||^2)`` is recomputed
    lazily inside ``project`` / ``retract`` from the constraint
    ``<x, x>_M = -1/c`` where the Minkowski inner product is

        ``<x, y>_M = -x_0 y_0 + sum_i x_i y_i``.

    The constraint defines the upper sheet of the two-sheeted hyperboloid;
    we always pick the future-pointing branch ``x_0 > 0`` via the positive
    sqrt. The advantage of this convention is that the d-dim parameter
    array stays drop-in-compatible with the rest of the training pipeline
    (which expects ``params.shape`` to match the circuit ansatz) -- the
    Lorentz embedding is a private bookkeeping device of this class.

    Why Lorentz over Poincare ball: Poincare's parameters compress near
    the ball boundary so 64-bit gradients vanish on long runs (arXiv
    2211.00181). Lorentz stays well-conditioned for the
    ~ O(1)-magnitude parameter scales VQE circuits live in.

    Curvature: ``c = 0.01`` (Phase 2.5b-v2 default). The earlier ``c = 1``
    run revealed that at U(0, 2*pi)-natural PQC scale the projection
    inflates the input displacement by ``c * ||x_s||^2 ~ 900x``, so a
    fixed-magnitude leash had to clip every step and the optimizer never
    saw Adam's per-coordinate scaling or QFIM's curvature preconditioning.
    Dropping ``c`` by 100x cuts the inflation factor to ``~9`` at the
    same parameter scale, putting Adam's natural step magnitude
    (~lr per coord) in the regime where the adaptive leash below is
    actually meaningful (i.e. mostly inactive).

    Geometry on a curvature-``c`` hyperboloid (with ``L = ||v||_M`` and
    ``v_0`` induced from the tangency constraint ``<x, v>_M = 0``):

        Tangent at ``x``:        ``T_x H = { v in R^{d+1} : <x, v>_M = 0 }``
        Project ambient ``g``:   ``Proj(g) = g - <x, g>_M / <x, x>_M * x``
                                            ``= g + c * <x, g>_M * x``
        Geodesic exp:            ``exp_x(v) = cosh(L) x + sinh(L) v / L``

    Both ``project`` and ``retract`` accept and return d-dim arrays; the
    caller passes a d-dim ambient direction ``vec`` (treated as the
    spatial part of an ambient vector with ``v_0 = 0`` for projection;
    ``v_0`` is induced from tangency for retract). The full ``(x_0, x_s)``
    invariant is satisfied by construction at every call.

    Adaptive growth-bounded leash: ``retract`` clips ``L = ||v||_M`` to
    ``self._L_cap = arccosh(1 + max_growth)`` (default ``max_growth=0.05``,
    so ``L_cap ~ 0.31``) before applying the cosh / sinh formulas.

    Why a *growth-bounded* (rather than magnitude-bounded) leash:

        (a) the geodesic exp ``new_x_s = cosh(L) x_s + sinh(L)/L v_s`` is
            radial-dominated once ``L >~ 2 / ||x_s||``: at large radius
            the cosh term scales ``x_s`` outward by a factor ``cosh(L)``
            while the tangential correction stays ``O(1)``. Capping
            ``cosh(L) - 1`` at ``max_growth`` directly bounds the radial
            inflation per step, which is the failure mode that matters;
        (b) at the new ``c = 0.01`` the projection inflation ``c||x_s||^2``
            is ``~9`` instead of ``~900``, so the cap is *usually inactive*:
            an Adam ``lr=0.05`` step on a 24-param VQE produces a
            post-projection ``L`` of order ``0.05 * 9 ~ 0.45``... still
            above ``L_cap = 0.31``, but only modestly, so most of Adam's
            adaptive-magnitude information now flows through. Smaller
            steps (Adam after a few iterations, or QNG with curvature
            normalization) skip the cap entirely.

    The clipped step still lies on the geodesic through ``x`` in the
    direction of ``v``, just at a bounded geodesic distance, so the
    Lorentz invariant ``<x, x>_M = -1/c`` continues to hold exactly.

    Tradeoff: at large ``||x_s||`` and when the cap *does* fire, the
    per-step descent is bounded by ``L_cap`` (not by ``lr * ||grad||``).
    But unlike the prior fixed-magnitude leash, this only happens when
    the requested step would inflate the radius by more than
    ``max_growth``; well-conditioned steps (small Adam updates,
    QFIM-rescaled moves) are not throttled.
    """

    name = "hyperbolic"

    def __init__(self, c=0.01, max_growth=0.05):
        self.c = float(c)
        self.max_growth = float(max_growth)
        # L_cap derived from cosh(L_cap) = 1 + max_growth so that radial
        # inflation per step stays at most max_growth. With the default
        # max_growth=0.05 this gives L_cap ~ 0.31, ~6x looser than the
        # prior fixed cap of 0.05, *and* it auto-skips when the Adam /
        # QFIM-supplied step is already smaller than the cap.
        self._L_cap = float(np.arccosh(1.0 + self.max_growth))

    def _x0(self, x_s_flat):
        # Future-pointing temporal coordinate from the hyperboloid
        # constraint <x, x>_M = -1/c, i.e. -x_0^2 + ||x_s||^2 = -1/c.
        return float(np.sqrt(1.0 / self.c + float(np.dot(x_s_flat, x_s_flat))))

    def project(self, theta, vec):
        # Embed ambient vec as (0, vec) in R^{d+1} and project onto T_x H:
        #   Proj((0, vec)) = (0, vec) + c * <x, (0, vec)>_M * x
        # The Minkowski inner product <x, (0, vec)>_M = sum_i x_i * vec_i
        # (the temporal contribution -x_0 * 0 vanishes), so the spatial
        # part of the projected tangent is
        #   v_T_s = vec + c * (x_s . vec) * x_s.
        theta_flat = np.asarray(theta).flatten()
        vec_flat = np.asarray(vec).flatten()
        ip = float(np.dot(theta_flat, vec_flat))
        proj_s = vec_flat + self.c * ip * theta_flat
        return proj_s.reshape(np.shape(vec))

    def retract(self, theta, vec):
        # Lorentz exponential map. We assume vec is a *spatial part* of a
        # tangent vector at x; the temporal part v_0 is induced from the
        # tangency constraint <x, v>_M = 0:
        #   -x_0 v_0 + x_s . v_s = 0  =>  v_0 = (x_s . v_s) / x_0
        # The Minkowski norm of a tangent is real:
        #   ||v||_M^2 = -v_0^2 + ||v_s||^2 >= 0
        # (clamped at 0 to absorb roundoff). We update only the spatial
        # part since the convention recomputes x_0 lazily on the next
        # project/retract:
        #   new_x_s = cosh(L) x_s + sinh(L) v_s / L,  L = ||v||_M.
        # L is clipped to self._L_cap (the geodesic distance whose cosh
        # is 1 + max_growth) before cosh/sinh; vec is rescaled by the
        # same factor so the geodesic direction is preserved (we walk
        # the same geodesic, just less far). See class docstring for
        # the growth-bounded-leash rationale.
        theta_flat = np.asarray(theta).flatten()
        vec_flat = np.asarray(vec).flatten()
        x0 = self._x0(theta_flat)
        v0 = float(np.dot(theta_flat, vec_flat)) / x0
        v_norm_sq = float(np.dot(vec_flat, vec_flat) - v0 * v0)
        if v_norm_sq < 0.0:
            v_norm_sq = 0.0
        v_norm = float(np.sqrt(v_norm_sq))
        if v_norm < 1e-12:
            return np.array(theta)
        if v_norm > self._L_cap:
            scale = self._L_cap / v_norm
            vec_flat = vec_flat * scale
            v_norm = self._L_cap
        ch = np.cosh(v_norm)
        sh = np.sinh(v_norm)
        new_s = ch * theta_flat + sh * vec_flat / v_norm
        return new_s.reshape(np.shape(theta))

    def inner(self, theta, u, v):
        # Minkowski inner product on the tangent space, induced from the
        # spatial parts of u, v via the tangency constraint (matches the
        # v_0 derivation in retract).
        theta_flat = np.asarray(theta).flatten()
        u_flat = np.asarray(u).flatten()
        v_flat = np.asarray(v).flatten()
        x0 = self._x0(theta_flat)
        u0 = float(np.dot(theta_flat, u_flat)) / x0
        v0 = float(np.dot(theta_flat, v_flat)) / x0
        return float(np.dot(u_flat, v_flat) - u0 * v0)

    def parallel_transport(self, theta_old, theta_new, vec):
        # Closed-form Lorentz parallel transport along the geodesic from
        # p -> q on the curvature-c hyperboloid (Becigneul & Ganea 2019):
        #   PT_{p->q}(v) = v - <q, v>_M / (1/c + <p, q>_M) * (p + q)
        # We store only spatial coordinates; temporal coords are reconstructed
        # lazily from the manifold constraint.
        p_s = np.asarray(theta_old).flatten()
        q_s = np.asarray(theta_new).flatten()
        v_s = np.asarray(vec).flatten()
        p0 = self._x0(p_s)
        q0 = self._x0(q_s)
        v0 = float(np.dot(p_s, v_s)) / p0
        q_dot_v = float(np.dot(q_s, v_s) - q0 * v0)
        p_dot_q = float(np.dot(p_s, q_s) - p0 * q0)
        denom = float(1.0 / self.c + p_dot_q)
        if abs(denom) < 1e-12:
            return np.asarray(vec)
        out_s = v_s - (q_dot_v / denom) * (p_s + q_s)
        return out_s.reshape(np.shape(vec))


# Registry mapping optimizer-name -> Manifold instance. Optimizers not listed
# here use ``Euclidean`` (i.e. the standard ``params - lr * update`` step).
# Phase 1: torus. Phase 2: three sphere optimizers, all sharing one Sphere
# instance because Sphere has no per-optimizer state (project/retract are
# pure functions of (theta, vec)). Phase 2.5b-v2: five hyperbolic
# optimizers sharing one Hyperbolic() instance (c=0.01, max_growth=0.05).
_SPHERE = Sphere()
_HYPERBOLIC = Hyperbolic()  # c=0.01, max_growth=0.05
_OPTIMIZER_MANIFOLDS = {
    "RQNG_torus_block":  Torus(),
    "RQNG_sphere_block": _SPHERE,
    "ProjQNG_sphere":    _SPHERE,
    "ProjAdam_sphere":   _SPHERE,
    # Phase 2.5a: QNGAdam_v1 on the sphere. Two variants for the same
    # update rule -- "Proj" projects the post-solve direction (mirrors
    # ProjQNG_sphere wrt QNG_block); "R" pre-projects the gradient before
    # the FS solve (mirrors RQNG_sphere_block wrt QNG_block). Both share
    # the geodesic retraction in _apply_manifold_step.
    "ProjQNGAdam_v1_sphere": _SPHERE,
    "RQNGAdam_v1_sphere":    _SPHERE,
    # Phase 2.5b: five hyperbolic optimizers mirroring the sphere lineup.
    # ProjAdam_hyperbolic stays in the Adam path; the four QNG-flavored
    # ones go through the manual QNG-family loop. R*_hyperbolic variants
    # pre-project the gradient onto T_x H before the FS solve (matching
    # RQNG_sphere_block / RQNGAdam_v1_sphere); Proj*_hyperbolic skip the
    # pre-projection and rely on the manifold step in _apply_manifold_step.
    "ProjAdam_hyperbolic":         _HYPERBOLIC,
    "ProjQNG_hyperbolic":          _HYPERBOLIC,
    "RQNG_hyperbolic_block":       _HYPERBOLIC,
    "ProjQNGAdam_v1_hyperbolic":   _HYPERBOLIC,
    "RQNGAdam_v1_hyperbolic":      _HYPERBOLIC,
}


def manifold_for(opt_name):
    """Return the ``Manifold`` instance the named optimizer wants its
    parameters to live on. Defaults to ``Euclidean``."""
    return _OPTIMIZER_MANIFOLDS.get(opt_name, Euclidean())
