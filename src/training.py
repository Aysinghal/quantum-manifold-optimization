"""
Generic training loops for all four optimizers.

For VQE (no data), PennyLane's QNGOptimizer works directly on the QNode.
For regression/classification (data-dependent circuits), QNG is implemented
manually: metric tensor is averaged over a subsample of training inputs,
gradient is computed for the aggregate cost, and the update is G^{-1} @ grad.
"""

import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# Approximate circuit evaluations per optimisation step.
EVAL_COST = {
    "GD": lambda d, L: 2 * d,
    "Adam": lambda d, L: 2 * d,
    "QNG_block": lambda d, L: 2 * d + L,
    "QNG_full": lambda d, L: 2 * d + d * (d + 1) // 2,
}


def make_optimizer(name, lr):
    """Instantiate a PennyLane optimizer (GD or Adam only)."""
    if name == "GD":
        return qml.GradientDescentOptimizer(stepsize=lr)
    if name == "Adam":
        return qml.AdamOptimizer(stepsize=lr)
    raise ValueError(f"Use train_vqe for QNG optimizers, got: {name}")


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


def _qng_step(mt_fn, cost_fn, params, x_train, lr, lam=1e-3, rng=None):
    """One QNG update using pre-built metric-tensor function.

    mt_fn:   output of qml.metric_tensor(circuit, approx=...)
    cost_fn: scalar cost that closes over data
    x_train: list of training inputs; metric tensor is averaged over a subsample
    """
    if rng is None:
        rng = np.random.default_rng()

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

    G = G_sum / k
    reg = max(lam, 1e-4 * np.max(np.abs(np.diag(G))))
    G = G + reg * np.eye(d)

    grad = qml.grad(cost_fn)(params)
    grad_flat = np.array(grad).flatten()

    update = np.linalg.solve(G, grad_flat)
    new_params = pnp.array(
        np.array(params) - lr * update.reshape(params.shape),
        requires_grad=True,
    )
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
    """
    if loss_type == "mse":
        cost_fn = mse_cost(circuit, x_train, y_train)
    else:
        cost_fn = hinge_cost(circuit, x_train, y_train)

    d = int(np.prod(params.shape))
    L = n_layers
    evals_per_step = EVAL_COST[opt_name](d, L)

    losses, wall_times, cum_evals = [], [], []
    total_evals = 0
    t0 = time.perf_counter()

    if opt_name in ("GD", "Adam"):
        opt = make_optimizer(opt_name, lr)
        for step in range(n_steps):
            params, loss = opt.step_and_cost(cost_fn, params)
            total_evals += evals_per_step
            losses.append(float(loss))
            wall_times.append(time.perf_counter() - t0)
            cum_evals.append(total_evals)
            if verbose and step % 25 == 0:
                print(f"    step {step:4d}  loss={float(loss):.6f}")
            if progress_cb and (step < 10 or step % 10 == 0):
                progress_cb(step, n_steps, float(loss))
    else:
        approx = "block-diag" if opt_name == "QNG_block" else None
        mt_fn = qml.metric_tensor(circuit, approx=approx)
        rng = np.random.default_rng(42)
        for step in range(n_steps):
            loss = float(cost_fn(params))
            params = _qng_step(mt_fn, cost_fn, params, x_train, lr, lam, rng)
            total_evals += evals_per_step
            losses.append(loss)
            wall_times.append(time.perf_counter() - t0)
            cum_evals.append(total_evals)
            if verbose and step % 25 == 0:
                print(f"    step {step:4d}  loss={loss:.6f}")
            if progress_cb and (step < 10 or step % 10 == 0):
                progress_cb(step, n_steps, loss)

    return {
        "params": params,
        "losses": losses,
        "wall_times": wall_times,
        "circuit_evals": cum_evals,
    }


# ---------------------------------------------------------------------------
# Training: VQE (no data -- QNGOptimizer works directly)
# ---------------------------------------------------------------------------

def train_vqe(circuit, params, opt_name, lr, n_steps, n_layers,
              lam=1e-3, verbose=True, progress_cb=None):
    """Train a VQE circuit where cost = circuit(params) = <H>.

    progress_cb: optional callable(step, n_steps, loss) invoked every step
                 for the first 10 steps, then every 10 steps thereafter.
    """
    d = int(np.prod(params.shape))
    L = n_layers
    evals_per_step = EVAL_COST[opt_name](d, L)

    losses, wall_times, cum_evals = [], [], []
    total_evals = 0
    t0 = time.perf_counter()

    if opt_name in ("GD", "Adam"):
        opt = make_optimizer(opt_name, lr)
    else:
        approx = "block-diag" if opt_name == "QNG_block" else None
        opt = qml.QNGOptimizer(stepsize=lr, approx=approx, lam=lam)

    for step in range(n_steps):
        params, loss = opt.step_and_cost(circuit, params)
        total_evals += evals_per_step
        losses.append(float(loss))
        wall_times.append(time.perf_counter() - t0)
        cum_evals.append(total_evals)
        if verbose and step % 25 == 0:
            print(f"    step {step:4d}  energy={float(loss):.6f}")
        if progress_cb and (step < 10 or step % 10 == 0):
            progress_cb(step, n_steps, float(loss))

    return {
        "params": params,
        "losses": losses,
        "wall_times": wall_times,
        "circuit_evals": cum_evals,
    }
