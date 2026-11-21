import numpy as np
import jax
import jax.numpy as jnp

from src import core


def test_r1_ternary_and_binary():
    r = core.r1(core.eig_ternary(3))
    assert r[13:].tolist() == [8, 4, 2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 2, 1]
    r = core.r1(core.eig_binary(3))
    assert r[7:].tolist() == [8 - w for w in range(8)]


def test_predictions_pauli1_L3():
    om, R, V = core.predictions(core.eig_pauli(1), 3)
    assert om.tolist() == [0, 1, 2, 3]
    assert R.tolist() == [20, 15, 6, 1]
    np.testing.assert_allclose(V[1:], [6 / 81, 3 / 81, 1 / 81], rtol=1e-13)


def _setup(seed=0):
    lam = core.eig_ternary(3)
    K, N = 13, 29
    tr = core.make_trainer(lam, 1, N, 0.2, 300)
    Vs, phi = core.draw_seed(seed, 1, 8, K)
    y = core.make_target(tr.xs, np.concatenate([[0.0], np.full(K, 0.005)]), phi)
    return tr, jnp.array(Vs), jnp.array(y), K, N


def test_antisymmetric_model_is_zero_at_init():
    tr, Vs, _, _, _ = _setup()
    assert np.all(np.asarray(tr.f(tr.p0, Vs)) == 0.0)


def test_no_fourier_weight_above_K():
    tr, Vs, _, K, N = _setup()
    rng = np.random.default_rng(1)
    params = jnp.array(rng.normal(size=tr.p0.shape))
    c = np.fft.fft(np.asarray(tr.f(params, Vs))) / N
    assert np.abs(c[1:K + 1]).max() > 1e-4          # the model is not trivially zero
    assert np.abs(c[K + 1]) < 1e-12                 # omega = +(K+1)
    assert np.abs(c[K + 2]) < 1e-12                 # omega = -(K+1)


def test_grad_matches_finite_difference():
    tr, Vs, y, _, _ = _setup()
    rng = np.random.default_rng(2)
    params = jnp.array(0.3 * rng.normal(size=tr.p0.shape))
    g = np.asarray(jax.grad(tr.loss)(params, Vs, y))
    h = 1e-5
    for _ in range(5):
        idx = tuple(rng.integers(0, s) for s in params.shape)
        e = np.zeros(params.shape)
        e[idx] = h
        fd = (tr.loss(params + e, Vs, y) - tr.loss(params - e, Vs, y)) / (2 * h)
        np.testing.assert_allclose(g[idx], float(fd), rtol=1e-6, atol=1e-10)


def test_local_spectral_rho_matches_matrix_power():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 7, 5)) + 1j * rng.normal(size=(4, 7, 5))
    G = 0.1 * A @ np.conj(np.swapaxes(A, -1, -2))                # Hermitian PSD, (S, N, N)
    idx, eta, T = np.array([1, 2, 3]), 0.2, 6
    rho, m1, m2 = core.local_spectral_rho(G, idx, eta, T)
    M = np.eye(7) - eta * G
    for t in range(T):
        Mt = np.linalg.matrix_power(M, t)
        np.testing.assert_allclose(rho[t], np.real(Mt[:, idx, idx]).mean(0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(m1, np.real(G[:, idx, idx]).mean(0), rtol=1e-10)
    np.testing.assert_allclose(m2, np.real((G @ G)[:, idx, idx]).mean(0), rtol=1e-10)


def test_two_moment_half_life_limits():
    eta, g = 0.2, 0.3
    # zero spread (k -> infinity) recovers the single-rate discrete half-life
    exact = np.log(0.5) / np.log(1 - eta * g)
    np.testing.assert_allclose(core.two_moment_half_life(g, g**2 * (1 + 1e-9), eta), exact, rtol=1e-6)
    # exponentially distributed rates (k = 1): (1 + kappa t)^-1 = 1/2 at t = 1/kappa, 1/ln2 slower
    np.testing.assert_allclose(core.two_moment_half_life(g, 2 * g**2, eta), -1 / np.log1p(-eta * g))


def test_crossing_time_interpolates_exponential():
    from src import experiments as ex
    t = np.arange(40)
    rates = np.array([0.05, 0.13])
    curve = np.exp(-rates[None, :] * t[:, None])
    np.testing.assert_allclose(ex.crossing_time(curve), np.log(2) / rates, rtol=1e-12)


def test_fast_mode_share_two_level():
    a = np.arcsin(np.sqrt(0.02))                           # 2% of index 0's weight on the fast mode
    U = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    G = (U @ np.diag([1.0, 100.0]) @ U.T)[None].astype(complex)
    weight, share = core.fast_mode_share(G, np.array([0]), factor=10)
    np.testing.assert_allclose(weight, [0.02], rtol=1e-12)
    np.testing.assert_allclose(share, [0.02 * 100 / (0.98 + 0.02 * 100)], rtol=1e-12)
