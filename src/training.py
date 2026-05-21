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
    # Phase 2.5a: QNGAdam_v1 on the sphere. Per-step circuit cost is the
    # same as the underlying QNGAdam_v1_block update (Adam moments are
    # purely classical and free); the project/retract is also classical.
    "ProjQNGAdam_v1_sphere": lambda d, L: 2 * d + L,
    "RQNGAdam_v1_sphere":    lambda d, L: 2 * d + L,
    # Phase 2.5b: hyperbolic-on-PQC. Per-step circuit cost mirrors the
    # underlying update rule -- the Lorentz project/retract is classical
    # and free. ProjAdam_hyperbolic stays in the flat-Adam path (2*d);
    # the four QNG-flavored ones touch the FS metric tensor (2*d + L).
    "ProjAdam_hyperbolic":         lambda d, L: 2 * d,
    "ProjQNG_hyperbolic":          lambda d, L: 2 * d + L,
    "RQNG_hyperbolic_block":       lambda d, L: 2 * d + L,
    "ProjQNGAdam_v1_hyperbolic":   lambda d, L: 2 * d + L,
    "RQNGAdam_v1_hyperbolic":      lambda d, L: 2 * d + L,
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
    # Phase 2.5a: QNGAdam_v1 on the sphere. Both variants touch the FS
    # metric tensor (Adam smoothing is on the raw grad, then FS-solve),
    # so they live in the manual QNG loop alongside the QNG_block sphere
    # optimizers above. ProjQNGAdam_v1_sphere matches ProjQNG_sphere's
    # post-solve-only projection; RQNGAdam_v1_sphere matches
    # RQNG_sphere_block's pre-solve gradient projection.
    "ProjQNGAdam_v1_sphere",
    "RQNGAdam_v1_sphere",
    # Phase 2.5b: hyperbolic QNG-flavored optimizers. Same dispatch logic
    # as their sphere counterparts (R* pre-projects, Proj* post-projects).
    # ProjAdam_hyperbolic stays in the flat-Adam path and is NOT listed
    # here.
    "ProjQNG_hyperbolic",
    "RQNG_hyperbolic_block",
    "ProjQNGAdam_v1_hyperbolic",
    "RQNGAdam_v1_hyperbolic",
}


def _apply_manifold_step(opt_name, params, update, lr):
    """Apply ``params <- retract(params, -lr * update)`` on the optimizer's
    chosen manifold and re-wrap the result as a PennyLane numpy array with
    ``requires_grad=True`` so the next ``qml.grad`` call still differentiates.

    For Euclidean optimizers this reduces to ``params - lr * update``. For
    ``RQNG_torus_block`` the result is wrapped into ``[0, 2*pi)^d`` via
    ``mod 2*pi``. Phase 2 sphere optimizers and Phase 2.5b hyperbolic
    optimizers project the displacement onto the tangent space at
    ``theta`` *before* retracting via the manifold's geodesic exponential
    map -- this guarantees the resulting point sits exactly on the
    sphere / hyperboloid (modulo float precision; sphere also re-normalizes
    inside its retract, hyperbolic recomputes ``x_0`` lazily on every
    project / retract so drift is structurally impossible).
    """
    manifold = manifold_for(opt_name)
    arr = np.array(params)
    step = (-lr * np.asarray(update)).reshape(arr.shape)
    if getattr(manifold, "name", None) in ("sphere", "hyperbolic"):
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
    ``Sphere`` / ``Hyperbolic`` this projects the displacement to the
    tangent at ``params_old`` and applies the geodesic exp map. Returns
    a PennyLane numpy array with ``requires_grad=True``.
    """
    manifold = manifold_for(opt_name)
    if getattr(manifold, "name", None) == "euclidean":
        return params_new
    arr_old = np.array(params_old)
    arr_new = np.array(params_new)
    step = arr_new - arr_old
    if getattr(manifold, "name", None) in ("sphere", "hyperbolic"):
        step = manifold.project(arr_old, step)
    retracted = manifold.retract(arr_old, step)
    return pnp.array(retracted, requires_grad=True)


def _transport_opt_state_m(opt_name, theta_old, theta_new, opt_state):
    """Parallel-transport QNG-Adam first-moment state across one manifold step."""
    if opt_state.get("m") is None:
        return
    manifold = manifold_for(opt_name)
    if getattr(manifold, "name", None) != "hyperbolic":
        return
    m_arr = np.asarray(opt_state["m"]).reshape(np.shape(theta_old))
    m_new = manifold.parallel_transport(
        np.asarray(theta_old), np.asarray(theta_new), m_arr
    )
    opt_state["m"] = np.asarray(m_new).flatten()


def _transport_pl_adam_fm(opt_name, theta_old, theta_new, opt):
    """Parallel-transport PennyLane Adam first-moment accumulator in-place."""
    manifold = manifold_for(opt_name)
    if getattr(manifold, "name", None) != "hyperbolic":
        return
    if not hasattr(opt, "fm") or opt.fm is None:
        return
    # PennyLane stores Adam moments as one entry per optimized argument.
    # Our training loops optimize a single parameter tensor, so transport
    # the first-moment entry in place while preserving list structure.
    if not isinstance(getattr(opt, "accumulation", None), dict):
        return
    fm_list = opt.accumulation.get("fm")
    if not isinstance(fm_list, list):
        return
    for idx, fm in enumerate(fm_list):
        fm_arr = np.asarray(fm).reshape(np.shape(theta_old))
        fm_new = manifold.parallel_transport(
            np.asarray(theta_old), np.asarray(theta_new), fm_arr
        )
        fm_list[idx] = np.asarray(fm_new).reshape(np.shape(fm))


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
        "RQNG_hyperbolic_block", "ProjQNG_hyperbolic",
    ):
        # R-QNG on the torus: projection onto T_theta T^d is the identity
        # (the torus is locally flat), so the natural-gradient direction is
        # algebraically the same as block-diag QNG. The manifold structure
        # only enters at the retraction step (see _apply_manifold_step).
        #
        # Sphere / hyperbolic variants: the FS-preconditioned direction is
        # computed identically to QNG_block here. The manifold geometry
        # enters in _apply_manifold_step, which projects this direction
        # onto the tangent space at theta and applies the manifold's
        # geodesic retraction. R*_block (sphere/hyperbolic) and
        # Proj*_sphere/hyperbolic differ only in whether grad_flat was
        # pre-projected onto the tangent before reaching this dispatch
        # (see _qng_step / train_vqe pre-projection branches) -- the FS
        # solve recipe is shared.
        return np.linalg.solve(G, grad_flat)

    if opt_name == "MomentumQNG_block":
        nat_grad = np.linalg.solve(G, grad_flat)
        if opt_state.get("vel") is None:
            opt_state["vel"] = np.zeros_like(grad_flat)
        opt_state["vel"] = _MOMENTUM_RHO * opt_state["vel"] + nat_grad
        return opt_state["vel"]

    if opt_name in (
        "QNGAdam_v1_block", "QNGAdam_v2_block",
        "ProjQNGAdam_v1_sphere", "RQNGAdam_v1_sphere",
        "ProjQNGAdam_v1_hyperbolic", "RQNGAdam_v1_hyperbolic",
    ):
        if opt_state.get("m") is None:
            opt_state["m"] = np.zeros_like(grad_flat)
            opt_state["v"] = np.zeros_like(grad_flat)
            opt_state["t"] = 0

        b1, b2, eps = _ADAM_BETA1, _ADAM_BETA2, _ADAM_EPS

        if opt_name in (
            "QNGAdam_v1_block",
            "ProjQNGAdam_v1_sphere", "RQNGAdam_v1_sphere",
            "ProjQNGAdam_v1_hyperbolic", "RQNGAdam_v1_hyperbolic",
        ):
            # Adam smoothing on the raw gradient, then preconditioner.
            # All four manifold-aware variants share this body. They
            # differ only in whether grad_flat was pre-projected onto
            # T_theta M upstream (R* variants do, Proj* variants do not),
            # and the manifold's geodesic retraction lives in
            # _apply_manifold_step.
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

    # R*_{sphere,hyperbolic} variants: project grad onto T_theta M BEFORE
    # the FS solve, so the natural-gradient direction comes out tangent.
    # The Proj*_{sphere,hyperbolic} variants skip this; they rely on
    # _apply_manifold_step to project the post-solve direction. The two
    # recipes coincide at the manifold's "origin" (||theta||=1 on the
    # sphere; the spatial origin x_s=0 on the hyperboloid) at first order
    # but differ once curvature kicks in across many steps.
    if opt_name in (
        "RQNG_sphere_block", "RQNGAdam_v1_sphere",
        "RQNG_hyperbolic_block", "RQNGAdam_v1_hyperbolic",
    ):
        manifold = manifold_for(opt_name)
        grad_flat = np.asarray(manifold.project(np.array(params), grad_flat)).flatten()

    update = _natgrad_direction(opt_name, G, grad_flat, opt_state)
    new_params = _apply_manifold_step(opt_name, params, update, lr)
    _transport_opt_state_m(opt_name, params, new_params, opt_state)
    return new_params


# ---------------------------------------------------------------------------
# Training: data-dependent circuits (regression / classification)
# ---------------------------------------------------------------------------

def train_with_data(circuit, params, x_train, y_train,
                    opt_name, lr, n_steps, n_layers,
                    loss_type="mse", lam=1e-3, verbose=True,
                    progress_cb=None):
    """Train a circuit whose cost aggregates over (x, y) pairs.

    Works for GD, Adam (via PennyLane optimizers) and QNG_block / QNG_full
    (via manual metric-tensor step).

    progress_cb: optional callable(step, n_steps, loss) invoked every step
                 for the first 10 steps, then every 10 steps thereafter.

    Always records ``theta_norms`` (one ||theta|| per step) in the result
    dict. The cost is one np.linalg.norm per step, negligible compared to
    the optimizer step itself, and the trajectory is needed for manifold
    diagnostics in the post-processing pipeline. ``theta_norms`` is the
    Euclidean ``np.linalg.norm`` of the flat parameter array; on the
    hyperbolic manifold this is the Euclidean norm of the **spatial
    coordinates** ``x_s`` (the d-dim parameter array), not the Minkowski
    norm of the (implicit-x_0) ambient point. The Minkowski norm is
    identically ``-1/c`` and would be uninformative; ``||x_s||`` tracks
    how far the spatial parameters drift, which is the analog of the
    sphere's ``||theta||`` shell.
    """
    if loss_type == "mse":
        cost_fn = mse_cost(circuit, x_train, y_train)
    else:
        cost_fn = hinge_cost(circuit, x_train, y_train)

    # Sphere optimizers project ||theta_init|| onto the natural-shell
    # radius; Euclidean / Torus / Hyperbolic get the raw uniform-on-[0,
    # 2pi) draw passed in by the caller (see init_on_manifold for the
    # per-manifold rationale).
    params = init_on_manifold(params, manifold_for(opt_name))

    d = int(np.prod(params.shape))
    L = n_layers
    evals_per_step = EVAL_COST[opt_name](d, L)

    losses, wall_times, cum_evals, theta_norms = [], [], [], []
    total_evals = 0
    t0 = time.perf_counter()

    if opt_name in ("GD", "Adam", "ProjAdam_sphere", "ProjAdam_hyperbolic"):
        # ProjAdam_{sphere,hyperbolic} reuse PennyLane's Adam to compute
        # the flat-space update, then snap the resulting params back onto
        # the manifold via the manifold's geodesic retraction (see
        # _retract_adam_step).
        adam_aliases = ("ProjAdam_sphere", "ProjAdam_hyperbolic")
        opt = make_optimizer("Adam" if opt_name in adam_aliases else opt_name, lr)
        for step in range(n_steps):
            params_old = params
            params, loss = opt.step_and_cost(cost_fn, params)
            params = _retract_adam_step(opt_name, params_old, params)
            _transport_pl_adam_fm(opt_name, params_old, params, opt)
            total_evals += evals_per_step
            losses.append(float(loss))
            wall_times.append(time.perf_counter() - t0)
            cum_evals.append(total_evals)
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
            theta_norms.append(float(np.linalg.norm(np.array(params).flatten())))
            if verbose and step % 25 == 0:
                print(f"    step {step:4d}  loss={loss:.6f}")
            if progress_cb and (step < 10 or step % 10 == 0):
                progress_cb(step, n_steps, loss)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return {
        "params": params,
        "losses": losses,
        "wall_times": wall_times,
        "circuit_evals": cum_evals,
        "theta_norms": theta_norms,
    }


# ---------------------------------------------------------------------------
# Training: VQE (no data -- QNGOptimizer works directly)
# ---------------------------------------------------------------------------

def train_vqe(circuit, params, opt_name, lr, n_steps, n_layers,
              lam=1e-3, verbose=True, progress_cb=None):
    """Train a VQE circuit where cost = circuit(params) = <H>.

    progress_cb: optional callable(step, n_steps, loss) invoked every step
                 for the first 10 steps, then every 10 steps thereafter.

    Always records ``theta_norms`` (one ||theta|| per step) in the result
    dict; see ``train_with_data`` for the rationale (incl. the hyperbolic
    "Euclidean norm of spatial coords" decision).
    """
    # Sphere optimizers project ||theta_init|| onto the natural-shell
    # radius; Euclidean / Torus / Hyperbolic get the raw uniform-on-[0,
    # 2pi) draw passed in by the caller (see init_on_manifold for the
    # per-manifold rationale).
    params = init_on_manifold(params, manifold_for(opt_name))

    d = int(np.prod(params.shape))
    L = n_layers
    evals_per_step = EVAL_COST[opt_name](d, L)

    losses, wall_times, cum_evals, theta_norms = [], [], [], []
    total_evals = 0
    t0 = time.perf_counter()

    if opt_name in ("GD", "Adam", "ProjAdam_sphere", "ProjAdam_hyperbolic"):
        # ProjAdam_{sphere,hyperbolic}: build a vanilla PennyLane Adam
        # optimizer; the post-step retract hook (in the loop below)
        # handles the manifold constraint after each Adam update.
        adam_aliases = ("ProjAdam_sphere", "ProjAdam_hyperbolic")
        opt = make_optimizer("Adam" if opt_name in adam_aliases else opt_name, lr)
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
            # ProjAdam_{sphere,hyperbolic} this snaps the post-Adam
            # params back onto the manifold via geodesic retraction.
            params = _retract_adam_step(opt_name, params_old, params)
            _transport_pl_adam_fm(opt_name, params_old, params, opt)
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
            # See _qng_step: R*_{sphere,hyperbolic} variants pre-project
            # the gradient so the FS solve runs on a tangent vector.
            if opt_name in (
                "RQNG_sphere_block", "RQNGAdam_v1_sphere",
                "RQNG_hyperbolic_block", "RQNGAdam_v1_hyperbolic",
            ):
                manifold = manifold_for(opt_name)
                grad_flat = np.asarray(
                    manifold.project(np.array(params), grad_flat)
                ).flatten()
            update = _natgrad_direction(opt_name, G, grad_flat, opt_state)
            params_new = _apply_manifold_step(opt_name, params, update, lr)
            _transport_opt_state_m(opt_name, params, params_new, opt_state)
            params = params_new
        total_evals += evals_per_step
        losses.append(float(loss))
        wall_times.append(time.perf_counter() - t0)
        cum_evals.append(total_evals)
        theta_norms.append(float(np.linalg.norm(np.array(params).flatten())))
        if verbose and step % 25 == 0:
            print(f"    step {step:4d}  energy={float(loss):.6f}")
        if progress_cb and (step < 10 or step % 10 == 0):
            progress_cb(step, n_steps, float(loss))

    return {
        "params": params,
        "losses": losses,
        "wall_times": wall_times,
        "circuit_evals": cum_evals,
        "theta_norms": theta_norms,
    }
