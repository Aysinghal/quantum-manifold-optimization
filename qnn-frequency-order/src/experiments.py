"""Experiments 1-3. Each returns a dict of numpy arrays (saved as .npz by run_all.py)."""
import numpy as np
import jax
import jax.numpy as jnp

from . import core

ETA = 0.2
T = 300
SEEDS = np.arange(100)
L_TRAIN = 1


# ----------------------------------------------------------------- metrics
def first_below(curve, thr=0.5):
    """First step t at which curve[t, k] < thr, per column. NaN if it never happens."""
    below = curve < thr
    t = below.argmax(0).astype(float)
    t[~below.any(0)] = np.nan
    return t


def projected_residual(r):
    """rho_w(t) = Re(r_w(t) conj(r_w(0))) / |r_w(0)|^2, per seed. r (S, T, K)."""
    r0 = r[:, :1, :]
    return np.real(r * np.conj(r0)) / np.abs(r0) ** 2


def relative_error(r):
    return np.abs(r) / np.abs(r[:, :1, :])


def summarise(r):
    rho_mean = projected_residual(r).mean(0)          # arithmetic mean over seeds
    rel_median = np.median(relative_error(r), axis=0)  # median over seeds
    return dict(rho_mean=rho_mean, half_life=first_below(rho_mean),
                rel_median=rel_median, rel_cross=first_below(rel_median))


# ------------------------------------------------- experiment 1: theory check
THEORY_CONFIGS = [
    ("pauli1_L3", "pauli(1), L = 3", core.eig_pauli(1), 3, 3000),
    ("pauli2_L2", "pauli(2), L = 2", core.eig_pauli(2), 2, 1000),
    ("ternary3_L1", "ternary(3), L = 1", core.eig_ternary(3), 1, 1000),
]


def theory_check():
    out = {}
    for i, (key, label, lam, L, S) in enumerate(THEORY_CONFIGS):
        rng = np.random.default_rng(1000 + i)
        d = len(lam)
        K = int(L * (lam.max() - lam.min()))
        N = 2 * K + 3
        f_single, _ = core.make_model(lam, L, N)
        theta0 = jnp.zeros((L + 1, d * d - 1))

        def one(V):
            f = f_single(theta0, V)
            J = jax.jacrev(f_single)(theta0, V).reshape(N, -1)
            c = jnp.fft.fft(f) / N
            Jc = jnp.fft.fft(J, axis=0) / N
            return jnp.abs(c) ** 2, jnp.sum(jnp.abs(Jc) ** 2, axis=1)

        batch = jax.jit(jax.vmap(one))
        Vs = np.stack([np.stack([core.haar(d, rng) for _ in range(L + 1)]) for _ in range(S)])
        c2, g = [], []
        for chunk in np.array_split(Vs, max(1, S // 250)):
            a, b = batch(jnp.array(chunk))
            c2.append(np.asarray(a))
            g.append(np.asarray(b))
        c2 = np.concatenate(c2).mean(0)[1:K + 1]
        g = np.concatenate(g).mean(0)[1:K + 1]

        _, R, V = core.predictions(lam, L)
        G_pred = core.kernel_diag_prediction(lam, L)
        out.update({
            f"{key}_omega": np.arange(1, K + 1),
            f"{key}_var_mc": c2,
            f"{key}_var_pred": V[1:],
            f"{key}_R": R[1:],
            f"{key}_kdiag_mc": g,
            f"{key}_kdiag_pred": G_pred[1:],
        })
    return out


# ------------------------------------------- experiments 2 and 3: training
def run_training(lam, amps_of_K, with_frozen=False):
    """Train seeds 0-99 at L = 1. amps_of_K(K) -> amplitudes of length K+1 (index 0 unused)."""
    L = L_TRAIN
    d = len(lam)
    K = int(lam.max() - lam.min())
    N = 2 * K + 3
    amps = amps_of_K(K)
    tr = core.make_trainer(lam, L, N, ETA, T)

    drawn = [core.draw_seed(int(s), L, d, K) for s in SEEDS]
    Vs = np.stack([v for v, _ in drawn])
    ys = np.stack([core.make_target(tr.xs, amps, phi) for _, phi in drawn])

    fs = np.asarray(tr.train_batch(jnp.array(Vs), jnp.array(ys)))      # (S, T, N)
    yhat = (np.fft.fft(ys, axis=-1) / N)[:, 1:K + 1]
    r = (np.fft.fft(fs, axis=-1) / N)[:, :, 1:K + 1] - yhat[:, None, :]  # (S, T, K)

    omega = np.arange(1, K + 1)
    G_pred = core.kernel_diag_prediction(lam, L)[1:]
    res = dict(omega=omega, amps=amps[1:], r=r, r1=core.r1(lam)[K + 1:],
               rate_pred=ETA * G_pred, half_life_pred=np.log(2) / (ETA * G_pred),
               f0_max=np.abs(fs[:, 0]).max())
    res.update(summarise(r))

    if with_frozen:
        J = np.asarray(tr.jac0_batch(jnp.array(Vs)))                    # (S, N, m)
        e = core.frozen_kernel_trajectory(J, ys, ETA, T)
        res["r_frozen"] = (np.fft.fft(e, axis=-1) / N)[:, :, 1:K + 1]
        G = core.fourier_kernel(J)                                      # (S, N, N)
        res["G_diag"] = np.real(np.einsum("sii->si", G))[:, 1:K + 1]    # per seed
        res["G_mean_abs"] = np.abs(G.mean(0))[1:K + 1, 1:K + 1]
    return res


def equal_amps(a):
    return lambda K: np.concatenate([[0.0], np.full(K, a)])


def dominant_amps(a, omega, factor):
    def make(K):
        amps = np.concatenate([[0.0], np.full(K, a)])
        amps[omega] *= factor
        return amps
    return make


def learning_order():
    binary = run_training(core.eig_binary(3), equal_amps(0.005))
    ternary = run_training(core.eig_ternary(3), equal_amps(0.005), with_frozen=True)
    return binary, ternary


def amplitude():
    return run_training(core.eig_ternary(3), dominant_amps(0.002, 5, 9.0))
