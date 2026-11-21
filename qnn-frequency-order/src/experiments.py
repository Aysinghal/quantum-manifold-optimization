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


def crossing_time(curve, thr=0.5):
    """Continuous time at which curve[:, k] first falls below thr, interpolating log(curve)
    linearly between the two bracketing steps. NaN if it never happens."""
    t1 = first_below(curve, thr)
    out = np.full(curve.shape[1], np.nan)
    for k, t in enumerate(t1):
        if np.isnan(t) or t == 0:
            out[k] = t
            continue
        a, b = np.log(curve[int(t) - 1, k]), np.log(curve[int(t), k])
        out[k] = t - 1 + (a - np.log(thr)) / (a - b)
    return out


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
def run_training(lam, amps_of_K, with_frozen=False, L=L_TRAIN, steps=T, keep_G=False):
    """Train seeds 0-99 with L encoding layers for `steps` GD steps (defaults: L = 1, T).
    amps_of_K(K) -> amplitudes of length K+1 (index 0 unused)."""
    d = len(lam)
    W = int(lam.max() - lam.min())
    K = L * W
    N = 2 * K + 3
    amps = amps_of_K(K)
    tr = core.make_trainer(lam, L, N, ETA, steps)

    drawn = [core.draw_seed(int(s), L, d, K) for s in SEEDS]
    Vs = np.stack([v for v, _ in drawn])
    ys = np.stack([core.make_target(tr.xs, amps, phi) for _, phi in drawn])

    fs = np.asarray(tr.train_batch(jnp.array(Vs), jnp.array(ys)))      # (S, T, N)
    yhat = (np.fft.fft(ys, axis=-1) / N)[:, 1:K + 1]
    r = (np.fft.fft(fs, axis=-1) / N)[:, :, 1:K + 1] - yhat[:, None, :]  # (S, T, K)

    omega = np.arange(1, K + 1)
    G_pred = core.kernel_diag_prediction(lam, L)[1:]
    r1 = np.zeros(K)
    r1[:W] = core.r1(lam)[W + 1:]                                       # zero above W when L > 1
    res = dict(omega=omega, amps=amps[1:], r=r, r1=r1, R=core.predictions(lam, L)[1][1:],
               rate_pred=ETA * G_pred, half_life_pred=np.log(2) / (ETA * G_pred),
               f0_max=np.abs(fs[:, 0]).max())
    res.update(summarise(r))

    if with_frozen:
        J = np.asarray(tr.jac0_batch(jnp.array(Vs)))                    # (S, N, m)
        e = core.frozen_kernel_trajectory(J, ys, ETA, steps)
        res["r_frozen"] = (np.fft.fft(e, axis=-1) / N)[:, :, 1:K + 1]
        G = core.fourier_kernel(J)                                      # (S, N, N)
        res["G_diag"] = np.real(np.einsum("sii->si", G))[:, 1:K + 1]    # per seed
        res["G_mean_abs"] = np.abs(G.mean(0))[1:K + 1, 1:K + 1]
        if keep_G:
            res["G"] = G
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


# ------------------------------------------------ experiment 4: kernel gap
def _kernel_gap_one(lam, trained):
    """Half-lives under successively finer kernel approximations, for one encoding.

    Uses the same seeds (hence the same V_l) as run_training; needs no training of its own.
    """
    L = L_TRAIN
    d = len(lam)
    K = int(lam.max() - lam.min())
    N = 2 * K + 3
    tr = core.make_trainer(lam, L, N, ETA, T)
    Vs = np.stack([core.draw_seed(int(s), L, d, K)[0] for s in SEEDS])
    G = core.fourier_kernel(np.asarray(tr.jac0_batch(jnp.array(Vs))))   # (S, N, N)
    idx = np.arange(1, K + 1)

    Gd = np.real(G[:, idx, idx])                                         # (S, K)
    rho_diag = ((1 - ETA * Gd)[:, None, :] ** np.arange(T)[None, :, None]).mean(0)
    rho_spec, m1, m2 = core.local_spectral_rho(G, idx, ETA, T)
    g_pred = trained["rate_pred"] / ETA                                 # theory E[G_ww]
    # All half-lives here are continuous crossing times of discrete-time GD curves, so that
    # whole-step rounding (up to 20% at 5 steps) does not masquerade as a slowdown.
    return dict(
        omega=idx, r1=trained["r1"],
        half_life=crossing_time(trained["rho_mean"]),               # measured (training)
        half_life_avg=np.log(0.5) / np.log(1 - ETA * g_pred),       # averaged kernel, theory
        half_life_diag=crossing_time(rho_diag),                     # each seed's own diagonal
        half_life_spec=crossing_time(rho_spec),                     # each seed's full kernel
        half_life_2mom=core.two_moment_half_life(m1, m2, ETA),
        rho_diag=rho_diag, rho_spec=rho_spec, m1=m1, m2=m2,
        k_diag=Gd.mean(0) ** 2 / Gd.var(0),                 # Gamma shape of seed-to-seed spread
    )


def kernel_gap(binary, ternary):
    """Experiment 4. Splits the gap between averaged-kernel and measured half-lives into
    seed-to-seed spread of G_ww and mixing between frequencies within each seed's kernel."""
    out = {}
    for name, lam, trained in [("binary", core.eig_binary(3), binary),
                               ("ternary", core.eig_ternary(3), ternary)]:
        out.update({f"{name}_{k}": v for k, v in _kernel_gap_one(lam, trained).items()})
    return out


# ------------------------------------------------------ experiment 5: depth
# (name, encoding, L, GD steps). Steps cover the slowest learnable frequency; the top frequency
# of ternary(3) at L = 2 (predicted half-life ~1300, measured several times that) is not reached.
DEPTH_CONFIGS = [
    ("pauli1_L3", "pauli(1), L = 3", core.eig_pauli(1), 3, 350),
    ("binary2_L2", "binary(2), L = 2", core.eig_binary(2), 2, 1600),
    ("ternary3_L2", "ternary(3), L = 2", core.eig_ternary(3), 2, 2000),
]
LEAK_FACTOR = 10   # a mode counts as "fast" for frequency w if its eigenvalue exceeds 10 E[G_ww]


def depth():
    """Experiment 5. Train with L >= 2 and compare half-lives to the corrected averaged kernel,
    the naive count R(omega), and each run's own frozen kernel; measure the fast-mode leak."""
    out = {}
    for key, _, lam, L, steps in DEPTH_CONFIGS:
        res = run_training(lam, equal_amps(0.002), with_frozen=True, L=L, steps=steps, keep_G=True)
        G, K = res["G"], len(res["omega"])
        idx = np.arange(1, K + 1)
        g_pred = res["rate_pred"] / ETA
        rho_spec, m1, m2 = core.local_spectral_rho(G, idx, ETA, steps)
        Gd = np.real(G[:, idx, idx])
        rho_diag = ((1 - ETA * Gd)[:, None, :] ** np.arange(steps)[None, :, None]).mean(0)
        leak_w, leak_m1 = core.fast_mode_share(G, idx, LEAK_FACTOR)
        a = res["amps"][0]
        out.update({f"{key}_{k}": v for k, v in dict(
            omega=res["omega"], R=res["R"], steps=steps,
            g_pred=g_pred, g_naive=g_pred[0] * res["R"] / res["R"][0],   # naive, matched at w = 1
            m1=m1, m2=m2,
            rho_mean=res["rho_mean"], rho_spec=rho_spec,
            half_life=crossing_time(res["rho_mean"]),
            half_life_avg=np.log(0.5) / np.log(1 - ETA * g_pred),
            half_life_diag=crossing_time(rho_diag),
            half_life_spec=crossing_time(rho_spec),
            half_life_2mom=core.two_moment_half_life(m1, m2, ETA),
            leak_weight=leak_w, leak_m1_share=leak_m1,
            frozen_err=(np.abs(res["r"] - res["r_frozen"]).mean(0) / a).max(),
            f0_max=res["f0_max"],
        ).items()})
    return out
