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

Phase 1 ships ``Euclidean`` and ``Torus`` only. ``Sphere`` is stubbed out for
Phase 2 -- not wired into any optimizer until then.
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


# Registry mapping optimizer-name -> Manifold instance. Optimizers not listed
# here use ``Euclidean`` (i.e. the standard ``params - lr * update`` step).
# Phase 1: torus. Phase 2: three sphere optimizers, all sharing one Sphere
# instance because Sphere has no per-optimizer state (project/retract are
# pure functions of (theta, vec)).
_SPHERE = Sphere()
_OPTIMIZER_MANIFOLDS = {
    "RQNG_torus_block":  Torus(),
    "RQNG_sphere_block": _SPHERE,
    "ProjQNG_sphere":    _SPHERE,
    "ProjAdam_sphere":   _SPHERE,
}


def manifold_for(opt_name):
    """Return the ``Manifold`` instance the named optimizer wants its
    parameters to live on. Defaults to ``Euclidean``."""
    return _OPTIMIZER_MANIFOLDS.get(opt_name, Euclidean())
