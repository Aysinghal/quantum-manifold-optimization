"""
Generic training loops for the QNG-family and baseline optimizers.

All QNG-family variants (QNG_block, QNG_full, MomentumQNG_block,
QNGAdam_v1_block, QNGAdam_v2_block) share the same manual update path,
which applies Fix 1 (adaptive trace-scaled Tikhonov damping) and Fix 5
(EMA on the metric tensor) to stabilize natural-gradient steps under
shot noise. They differ in:
  - the metric-tensor approximation passed to qml.metric_tensor
    ("block-diag" for *_block variants, None for QNG_full);
  - how the natural-gradient direction is built from (G, grad), which is
    dispatched in _natgrad_direction().

For regression/classification (data-dependent circuits), the metric tensor
is averaged over a subsample of training inputs; the gradient is computed
for the aggregate cost; the per-step direction is built by
_natgrad_direction() and applied as theta -= lr * direction.
"""

import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.manifolds import manifold_for
from src.models import init_on_manifold


# Approximate circuit evaluations per optimisation step. Momentum / Adam
# state for the new variants is purely classical, so per-step circuit cost
# matches the underlying QNG_block. R-QNG variants pay the same circuit cost
# as block-diag QNG; the manifold retraction is a free classical post-step.
EVAL_COST = {
    "GD": lambda d, L: 2 * d,
    "Adam": lambda d, L: 2 * d,
    "QNG_block": lambda d, L: 2 * d + L,
    "QNG_full": lambda d, L: 2 * d + d * (d + 1) // 2,
    "MomentumQNG_block": lambda d, L: 2 * d + L,
    "QNGAdam_v1_block":  lambda d, L: 2 * d + L,
    "QNGAdam_v2_block":  lambda d, L: 2 * d + L,
    "RQNG_torus_block":  lambda d, L: 2 * d + L,
    # Phase 2 sphere optimizers. Per-step circuit-evaluation count matches
    # the underlying update rule; the sphere project / retract is classical
    # and free.
    "RQNG_sphere_block": lambda d, L: 2 * d + L,
    "ProjQNG_sphere":    lambda d, L: 2 * d + L,
    "ProjAdam_sphere":   lambda d, L: 2 * d,
}


# Optimizer names handled by the manual QNG-family loop.
_QNG_FAMILY = {
    "QNG_block",
    "QNG_full",
    "MomentumQNG_block",
    "QNGAdam_v1_block",
    "QNGAdam_v2_block",
    "RQNG_torus_block",
    # Phase 2: both QNG-side sphere optimizers go through the manual
    # natural-gradient loop. ProjAdam_sphere stays in the Adam path.
    "RQNG_sphere_block",
    "ProjQNG_sphere",
}


def _apply_manifold_step(opt_name, params, update, lr):
    """Apply ``params <- retract(params, -lr * update)`` on the optimizer's
    chosen manifold and re-wrap the result as a PennyLane numpy array with
    ``requires_grad=True`` so the next ``qml.grad`` call still differentiates.

    For Euclidean optimizers this reduces to ``params - lr * update``. For
    ``RQNG_torus_block`` the result is wrapped into ``[0, 2*pi)^d`` via
    ``mod 2*pi``. Phase 2 sphere optimizers (``RQNG_sphere_block``,
    ``ProjQNG_sphere``, ``ProjAdam_sphere``) project the displacement onto
    the tangent space at ``theta`` *before* retracting via the geodesic
    exponential map -- this guarantees the resulting point sits exactly on
    the unit sphere.
    """
    manifold = manifold_for(opt_name)
    arr = np.array(params)
    step = (-lr * np.asarray(update)).reshape(arr.shape)
    if getattr(manifold, "name", None) == "sphere":
        step = manifold.project(arr, step)
    new_arr = manifold.retract(arr, step)
    return pnp.array(new_arr, requires_grad=True)


def _retract_adam_step(opt_name, params_old, params_new):
    """Adam path post-step hook: snap PennyLane-Adam's flat-space update
    onto the optimizer's manifold.

    ``params_new = params_old - lr * adam_dir`` was already computed by
    PennyLane's optimizer. We treat the displacement ``(params_new -
    params_old)`` as the Euclidean step and retract it on the manifold.
    For ``Euclidean`` this is a no-op (``retract`` is plain addition); for
    ``Sphere`` this projects the displacement to the tangent and applies
    the geodesic exp map. Returns a PennyLane numpy array with
    ``requires_grad=True``.
    """
    manifold = manifold_for(opt_name)
    if getattr(manifold, "name", None) == "euclidean":
        return params_new
    arr_old = np.array(params_old)
    arr_new = np.array(params_new)
    step = arr_new - arr_old
    if getattr(manifold, "name", None) == "sphere":
        step = manifold.project(arr_old, step)
    retracted = manifold.retract(arr_old, step)
    return pnp.array(retracted, requires_grad=True)


def _mt_approx_for(opt_name):
    """Return the metric_tensor approximation argument for a QNG-family opt."""
    return None if opt_name == "QNG_full" else "block-diag"


def make_optimizer(name, lr):
    """Instantiate a PennyLane optimizer (GD or Adam only)."""
    if name == "GD":
        return qml.GradientDescentOptimizer(stepsize=lr)
    if name == "Adam":
        return qml.AdamOptimizer(stepsize=lr)
    raise ValueError(f"Use the QNG-family training path for: {name}")


# ---------------------------------------------------------------------------
# Natural-gradient step-direction dispatch
# ---------------------------------------------------------------------------

# Default hyperparameters for the new variants. Match the literature:
#   Borysenko et al. 2024 (MomentumQNG): rho = 0.9
#   Kingma & Ba 2014 (Adam):             beta1=0.9, beta2=0.999, eps=1e-8
_MOMENTUM_RHO = 0.9
_ADAM_BETA1   = 0.9
_ADAM_BETA2   = 0.999
_ADAM_EPS     = 1e-8


def _natgrad_direction(opt_name, G, grad_flat, opt_state):
    """Build the per-step parameter-update direction for QNG-family optimizers.

    The caller applies ``theta -= lr * direction``.

    G:          stabilised metric tensor (G_ema + reg * I), shape (d, d)
    grad_flat:  flat gradient vector, shape (d,)
    opt_state:  dict mutated in place; holds {m, v, vel, t} as needed
    """
    if opt_name in (
        "QNG_block", "QNG_full", "RQNG_torus_block",
        "RQNG_sphere_block", "ProjQNG_sphere",
    ):
        # R-QNG on the torus: projection onto T_theta T^d is the identity
        # (the torus is locally flat), so the natural-gradient direction is
        # algebraically the same as block-diag QNG. The manifold structure
        # only enters at the retraction step (see _apply_manifold_step).
        #
        # Sphere variants: the FS-preconditioned direction is computed
        # identically to QNG_block here. The sphere geometry enters in
        # _apply_manifold_step, which projects this direction to the
        # tangent space at theta and applies the geodesic retraction.
        # RQNG_sphere_block and ProjQNG_sphere differ only in the per-task
        # init / convergence behavior driven by the sphere constraint --
        # the per-step dispatch shares this branch.
        return np.linalg.solve(G, grad_flat)

    if opt_name == "MomentumQNG_block":
        nat_grad = np.linalg.solve(G, grad_flat)
        if opt_state.get("vel") is None:
            opt_state["vel"] = np.zeros_like(grad_flat)
        opt_state["vel"] = _MOMENTUM_RHO * opt_state["vel"] + nat_grad
        return opt_state["vel"]

    if opt_name in ("QNGAdam_v1_block", "QNGAdam_v2_block"):
        if opt_state.get("m") is None:
            opt_state["m"] = np.zeros_like(grad_flat)
            opt_state["v"] = np.zeros_like(grad_flat)
            opt_state["t"] = 0

        b1, b2, eps = _ADAM_BETA1, _ADAM_BETA2, _ADAM_EPS

        if opt_name == "QNGAdam_v1_block":
            # Adam smoothing on the raw gradient, then preconditioner.
            g = grad_flat
            opt_state["m"] = b1 * opt_state["m"] + (1.0 - b1) * g
            opt_state["v"] = b2 * opt_state["v"] + (1.0 - b2) * g * g
            opt_state["t"] += 1
            t = opt_state["t"]
            m_hat = opt_state["m"] / (1.0 - b1 ** t)
            v_hat = opt_state["v"] / (1.0 - b2 ** t)
            adam_dir = m_hat / (np.sqrt(v_hat) + eps)
            return np.linalg.solve(G, adam_dir)

        # QNGAdam_v2: preconditioner first, then Adam on the natural gradient.
        ng = np.linalg.solve(G, grad_flat)
        opt_state["m"] = b1 * opt_state["m"] + (1.0 - b1) * ng
        opt_state["v"] = b2 * opt_state["v"] + (1.0 - b2) * ng * ng
        opt_state["t"] += 1
        t = opt_state["t"]
        m_hat = opt_state["m"] / (1.0 - b1 ** t)
        v_hat = opt_state["v"] / (1.0 - b2 ** t)
        return m_hat / (np.sqrt(v_hat) + eps)

    raise ValueError(f"Unknown QNG-family optimizer: {opt_name}")


# ---------------------------------------------------------------------------
# Cost-function factories
# ---------------------------------------------------------------------------

def mse_cost(circuit, x_train, y_train):
    """Return a cost function: MSE over dataset, closing over data."""
    def cost(params):
        total = pnp.array(0.0)
        for x, y in zip(x_train, y_train):
            pred = circuit(params, x)
            total = total + (pred - y) ** 2
        return total / len(x_train)
    return cost


def hinge_cost(circuit, x_train, y_train):
    """Return a cost function: squared hinge loss for binary classification."""
    def cost(params):
        total = pnp.array(0.0)
        for x, y in zip(x_train, y_train):
            pred = circuit(params, x)
            total = total + (1.0 - y * pred) ** 2
        return total / len(x_train)
    return cost


# ---------------------------------------------------------------------------
# Manual QNG step (for circuits with data arguments)
# ---------------------------------------------------------------------------

_MT_SUBSAMPLE = 10


def _qng_step(mt_fn, cost_fn, params, x_train, lr, lam=1e-3, rng=None,
              ema_state=None, opt_state=None, opt_name="QNG_block"):
    """One QNG-family update using pre-built metric-tensor function.

    mt_fn:     output of qml.metric_tensor(circuit, approx=...)
    cost_fn:   scalar cost that closes over data
    x_train:   list of training inputs; metric tensor is averaged over a subsample
    ema_state: dict holding {"G": array_or_None, "alpha": float} -- updated in place
               to enable EMA smoothing of the metric tensor (Fix 5).
    opt_state: dict holding optimizer-specific state (m, v, vel, t) -- mutated in
               place by _natgrad_direction(). Required for MomentumQNG / QNGAdam.
    opt_name:  one of QNG_block, QNG_full, MomentumQNG_block,
               QNGAdam_v1_block, QNGAdam_v2_block. Selects both the metric-tensor
               approximation that mt_fn was built with and the natural-gradient
               direction recipe in _natgrad_direction.
    """
    if rng is None:
        rng = np.random.default_rng()
    if opt_state is None:
        opt_state = {}

    n = len(x_train)
    k = min(n, _MT_SUBSAMPLE)
    indices = rng.choice(n, size=k, replace=False)

    mt0 = mt_fn(params, x_train[indices[0]])
    shape = mt0.shape
    d = int(np.prod(shape[: len(shape) // 2]))
    G_sum = np.array(mt0).reshape(d, d)

    for idx in indices[1:]:
        mt_i = mt_fn(params, x_train[idx])
        G_sum += np.array(mt_i).reshape(d, d)

    G_new = G_sum / k

    # Fix 5: exponential moving average of G across steps.
    if ema_state is not None:
        if ema_state["G"] is None:
            ema_state["G"] = G_new
        else:
            a = ema_state["alpha"]
            ema_state["G"] = a * ema_state["G"] + (1.0 - a) * G_new
        G = ema_state["G"]
    else:
        G = G_new
    # Fix 1: adaptive trace-scaled Tikhonov damping.
    reg = max(lam, 0.05 * np.mean(np.abs(np.diag(G))))

    G = G + reg * np.eye(d)

    grad = qml.grad(cost_fn)(params)
    grad_flat = np.array(grad).flatten()

    # RQNG_sphere_block: project grad onto T_theta S BEFORE the FS solve, so
    # the natural-gradient direction comes out tangent. ProjQNG_sphere skips
    # this; it relies on _apply_manifold_step to project the post-solve
    # direction. The two recipes coincide for ||theta||=1 at first order but
    # differ once curvature kicks in across many steps.
    if opt_name == "RQNG_sphere_block":
        manifold = manifold_for(opt_name)
        grad_flat = np.asarray(manifold.project(np.array(params), grad_flat)).flatten()

    update = _natgrad_direction(opt_name, G, grad_flat, opt_state)
    return _apply_manifold_step(opt_name, params, update, lr)


# ---------------------------------------------------------------------------
# Training: data-dependent circuits (regression / classification)
# ---------------------------------------------------------------------------

def train_with_data(circuit, params, x_train, y_train,
                    opt_name, lr, n_steps, n_layers,
                    loss_type="mse", lam=1e-3, verbose=True,
                    progress_cb=None, track_theta_norm=False):
    """Train a circuit whose cost aggregates over (x, y) pairs.

    Works for GD, Adam (via PennyLane optimizers) and QNG_block / QNG_full
    (via manual metric-tensor step).

    progress_cb: optional callable(step, n_steps, loss) invoked every step
                 for the first 10 steps, then every 10 steps thereafter.
    """
    if loss_type == "mse":
        cost_fn = mse_cost(circuit, x_train, y_train)
    else:
        cost_fn = hinge_cost(circuit, x_train, y_train)

    # Sphere optimizers need ||theta_init|| = 1; Euclidean / Torus get the
    # raw uniform-on-[0, 2pi) draw passed in by the caller.
    params = init_on_manifold(params, manifold_for(opt_name))

    d = int(np.prod(params.shape))
    L = n_layers
    evals_per_step = EVAL_COST[opt_name](d, L)

    losses, wall_times, cum_evals = [], [], []
    theta_norms = [] if track_theta_norm else None
    total_evals = 0
    t0 = time.perf_counter()

    if opt_name in ("GD", "Adam") or opt_name == "ProjAdam_sphere":
        # ProjAdam_sphere reuses PennyLane's Adam to compute the flat-space
        # update, then snaps the resulting params back onto the unit sphere
        # via the manifold's geodesic retraction (see _retract_adam_step).
        opt = make_optimizer("Adam" if opt_name == "ProjAdam_sphere" else opt_name, lr)
        for step in range(n_steps):
            params_old = params
            params, loss = opt.step_and_cost(cost_fn, params)
            params = _retract_adam_step(opt_name, params_old, params)
            total_evals += evals_per_step
            losses.append(float(loss))
            wall_times.append(time.perf_counter() - t0)
            cum_evals.append(total_evals)
            if track_theta_norm:
                theta_norms.append(float(np.linalg.norm(np.array(params).flatten())))
            if verbose and step % 25 == 0:
                print(f"    step {step:4d}  loss={float(loss):.6f}")
            if progress_cb and (step < 10 or step % 10 == 0):
                progress_cb(step, n_steps, float(loss))
    elif opt_name in _QNG_FAMILY:
        approx = _mt_approx_for(opt_name)
        mt_fn = qml.metric_tensor(circuit, approx=approx)
        rng = np.random.default_rng(42)
        # Fix 5: EMA state shared by all QNG-family variants.
        ema_state = {"G": None, "alpha": 0.9}
        # Optimizer-specific state (momentum buffer / Adam moments). Mutated
        # in place by _natgrad_direction.
        opt_state = {}
        for step in range(n_steps):
            loss = float(cost_fn(params))
            params = _qng_step(
                mt_fn, cost_fn, params, x_train, lr, lam, rng,
                ema_state=ema_state, opt_state=opt_state, opt_name=opt_name,
            )
            total_evals += evals_per_step
            losses.append(loss)
            wall_times.append(time.perf_counter() - t0)
            cum_evals.append(total_evals)
            if track_theta_norm:
                theta_norms.append(float(np.linalg.norm(np.array(params).flatten())))
            if verbose and step % 25 == 0:
                print(f"    step {step:4d}  loss={loss:.6f}")
            if progress_cb and (step < 10 or step % 10 == 0):
                progress_cb(step, n_steps, loss)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    out = {
        "params": params,
        "losses": losses,
        "wall_times": wall_times,
        "circuit_evals": cum_evals,
    }
    if track_theta_norm:
        out["theta_norms"] = theta_norms
    return out


# ---------------------------------------------------------------------------
# Training: VQE (no data -- QNGOptimizer works directly)
# ---------------------------------------------------------------------------

def train_vqe(circuit, params, opt_name, lr, n_steps, n_layers,
              lam=1e-3, verbose=True, progress_cb=None,
              track_theta_norm=False):
    """Train a VQE circuit where cost = circuit(params) = <H>.

    progress_cb: optional callable(step, n_steps, loss) invoked every step
                 for the first 10 steps, then every 10 steps thereafter.
    """
    # Sphere optimizers need ||theta_init|| = 1; Euclidean / Torus get the
    # raw uniform-on-[0, 2pi) draw passed in by the caller.
    params = init_on_manifold(params, manifold_for(opt_name))

    d = int(np.prod(params.shape))
    L = n_layers
    evals_per_step = EVAL_COST[opt_name](d, L)

    losses, wall_times, cum_evals = [], [], []
    theta_norms = [] if track_theta_norm else None
    total_evals = 0
    t0 = time.perf_counter()

    if opt_name in ("GD", "Adam") or opt_name == "ProjAdam_sphere":
        # ProjAdam_sphere: build a vanilla PennyLane Adam optimizer; the
        # post-step retract hook (in the loop below) handles the sphere
        # constraint after each Adam update.
        opt = make_optimizer("Adam" if opt_name == "ProjAdam_sphere" else opt_name, lr)
    elif opt_name in _QNG_FAMILY:
        # Shared manual loop: applies Fix 1 (adaptive Tikhonov) + Fix 5 (EMA on G)
        # to all QNG-family variants. The only differences are (a) which
        # metric_tensor approximation mt_fn was built with, and (b) which
        # natural-gradient direction recipe is dispatched in _natgrad_direction.
        opt = None
        approx = _mt_approx_for(opt_name)
        mt_fn = qml.metric_tensor(circuit, approx=approx)
        G_ema = None
        ema_alpha = 0.9
        opt_state = {}
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    for step in range(n_steps):
        if opt is not None:
            params_old = params
            params, loss = opt.step_and_cost(circuit, params)
            # No-op for vanilla Adam/GD (Euclidean manifold); for
            # ProjAdam_sphere this snaps back onto the unit sphere.
            params = _retract_adam_step(opt_name, params_old, params)
        else:
            loss = float(circuit(params))
            mt_raw = mt_fn(params)
            G_new = np.array(mt_raw).reshape(d, d)
            if G_ema is None:
                G_ema = G_new
            else:
                G_ema = ema_alpha * G_ema + (1.0 - ema_alpha) * G_new
            reg = max(lam, 0.05 * np.mean(np.abs(np.diag(G_ema))))
            G = G_ema + reg * np.eye(d)
            grad = qml.grad(circuit)(params)
            grad_flat = np.array(grad).flatten()
            # See _qng_step: RQNG_sphere_block pre-projects the gradient so
            # the FS solve runs on a tangent vector.
            if opt_name == "RQNG_sphere_block":
                manifold = manifold_for(opt_name)
                grad_flat = np.asarray(
                    manifold.project(np.array(params), grad_flat)
                ).flatten()
            update = _natgrad_direction(opt_name, G, grad_flat, opt_state)
            params = _apply_manifold_step(opt_name, params, update, lr)
        total_evals += evals_per_step
        losses.append(float(loss))
        wall_times.append(time.perf_counter() - t0)
        cum_evals.append(total_evals)
        if track_theta_norm:
            theta_norms.append(float(np.linalg.norm(np.array(params).flatten())))
        if verbose and step % 25 == 0:
            print(f"    step {step:4d}  energy={float(loss):.6f}")
        if progress_cb and (step < 10 or step % 10 == 0):
            progress_cb(step, n_steps, float(loss))

    out = {
        "params": params,
        "losses": losses,
        "wall_times": wall_times,
        "circuit_evals": cum_evals,
    }
    if track_theta_norm:
        out["theta_norms"] = theta_norms
    return out
