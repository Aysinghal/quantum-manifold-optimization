"""Core model: re-uploading QNN as a dense statevector simulation (numpy + JAX, CPU)."""
import itertools
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------- encodings
def eig_pauli(n):
    return np.array([bin(j).count("1") for j in range(2**n)])


def eig_binary(n):
    return np.arange(2**n)


def eig_ternary(n):
    return np.array([sum(3**i * ((j >> i) & 1) for i in range(n)) for j in range(2**n)])


# ------------------------------------------------------- redundancy, theory
def r1(lam):
    """Single-layer redundancy over omega = -W..W; index W is omega = 0."""
    W = lam.max() - lam.min()
    r = np.zeros(2 * W + 1)
    for a in lam:
        for b in lam:
            r[a - b + W] += 1
    return r


def conv_pow(r, L):
    out = np.array([1.0])
    for _ in range(L):
        out = np.convolve(out, r)
    return out


def predictions(lam, L):
    """omegas 0..K, R(omega), Var(c_omega) (valid for omega != 0, O = Z on qubit 1)."""
    d = len(lam)
    r = r1(lam)
    W = (len(r) - 1) // 2
    R = conv_pow(r, L)
    rm = r.copy()
    rm[W] -= 1
    V = conv_pow(rm, L) / ((d - 1) ** L * (d + 1) ** (L + 1))
    c = (len(R) - 1) // 2
    return np.arange(0, c + 1), R[c:], V[c:]


def kernel_diag_prediction(lam, L):
    """E[G_ww] = (L+1) (d^2/2) Var(c_w), for omega = 0..K (valid for omega != 0)."""
    d = len(lam)
    return (L + 1) * (d**2 / 2) * predictions(lam, L)[2]


# ------------------------------------------------------------------- model
def haar(d, rng):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def pauli_basis(n):
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1])
    out = []
    for s in itertools.product([I, X, Y, Z], repeat=n):
        m = np.array([[1.0 + 0j]])
        for p in s:
            m = np.kron(m, p)
        out.append(m)
    return np.array(out[1:])  # drop identity -> (d^2 - 1, d, d)


_expm = jax.vmap(jax.scipy.linalg.expm)


def make_model(lam, L, N):
    """Returns f_single(theta (L+1, d^2-1), V (L+1, d, d)) -> f on the grid (N,), and the grid."""
    d = len(lam)
    n = int(np.log2(d))
    P = jnp.array(pauli_basis(n))
    xs = 2 * jnp.pi * jnp.arange(N) / N
    z = jnp.array([1.0 if j < d // 2 else -1.0 for j in range(d)])
    phases = jnp.exp(-1j * xs[:, None] * jnp.array(lam, dtype=jnp.float64)[None, :])  # (N, d)

    def f_single(theta, V):
        A = jnp.einsum("lk,kab->lab", theta, P)
        W = jnp.einsum("lab,lbc->lac", V, _expm(-0.5j * A))
        psi = jnp.broadcast_to(W[0][:, 0], (N, d))
        for l in range(1, L + 1):
            psi = (psi * phases) @ W[l].T
        return jnp.sum(jnp.abs(psi) ** 2 * z, axis=1)

    return f_single, xs


# ---------------------------------------------------------------- training
def draw_seed(seed, L, d, K):
    """Seed convention: V_0..V_L first, then phi = uniform(0, 2pi, K+1) (index 0 unused)."""
    rng = np.random.default_rng(seed)
    Vs = np.stack([haar(d, rng) for _ in range(L + 1)])
    phi = rng.uniform(0, 2 * np.pi, K + 1)
    return Vs, phi


def make_target(xs, amps, phi):
    """y(x) = sum_{w=1..K} 2 a_w cos(w x + phi_w); amps and phi have length K+1, index 0 unused."""
    xs = np.asarray(xs)
    om = np.arange(1, len(amps))
    return (2 * amps[1:, None] * np.cos(om[:, None] * xs[None, :] + phi[1:, None])).sum(0)


def make_trainer(lam, L, N, eta, T):
    """Antisymmetric model f = (f_A - f_B)/sqrt(2), loss, GD loop and init Jacobian."""
    d = len(lam)
    f_single, xs = make_model(lam, L, N)
    p0 = jnp.zeros((2, L + 1, d * d - 1))

    def f(params, Vs):
        return (f_single(params[0], Vs) - f_single(params[1], Vs)) / jnp.sqrt(2.0)

    def loss(params, Vs, y):
        return 0.5 * jnp.mean((f(params, Vs) - y) ** 2)

    def _loss_aux(params, Vs, y):
        out = f(params, Vs)
        return 0.5 * jnp.mean((out - y) ** 2), out

    def train(Vs, y):
        """Full-batch GD from theta = 0. Returns f on the grid before each update, (T, N)."""

        def step(p, _):
            g, out = jax.grad(_loss_aux, has_aux=True)(p, Vs, y)
            return p - eta * g, out

        _, fs = jax.lax.scan(step, p0, None, length=T)
        return fs

    def jac0(Vs):
        """Jacobian of f(grid) w.r.t. all parameters at initialisation, (N, m)."""
        return jax.jacrev(f)(p0, Vs).reshape(N, -1)

    return SimpleNamespace(
        f=f, loss=loss, p0=p0, xs=np.asarray(xs),
        train_batch=jax.jit(jax.vmap(train)),
        jac0_batch=jax.jit(jax.vmap(jac0)),
    )


def frozen_kernel_trajectory(J, y, eta, T):
    """Linearised dynamics with the kernel frozen at initialisation.

    J (S, N, m), y (S, N). Iterates e <- e - eta Kx e from e = -y with Kx = J J^T / N.
    Returns e before each update, (S, T, N).
    """
    N = J.shape[1]
    Kx = np.einsum("sim,sjm->sij", J, J) / N
    e = -np.array(y)
    out = np.empty((J.shape[0], T, N))
    for t in range(T):
        out[:, t] = e
        e = e - eta * np.einsum("sij,sj->si", Kx, e)
    return out


def fourier_kernel(J):
    """Fourier tangent kernel G = Jc Jc^dagger with Jc = FFT(J)/N. J (..., N, m) -> (..., N, N)."""
    N = J.shape[-2]
    Jc = np.fft.fft(J, axis=-2) / N
    return Jc @ np.conj(np.swapaxes(Jc, -1, -2))


# ------------------------------------------------- kernel gap (experiment 4)
def local_spectral_rho(G, idx, eta, T):
    """Seed- and phase-averaged projected residual under each seed's frozen kernel.

    In Fourier space the frozen dynamics are r <- (I - eta G) r. With target phases uniform and
    independent across omega, E_phase[rho_w(t)] = [(I - eta G)^t]_ww = sum_n |U_wn|^2 (1 - eta l_n)^t,
    a mixture of decays weighted by the local spectral measure of G at w.

    G (S, N, N) Hermitian, idx (K,) Fourier indices. Returns rho (T, K) averaged over seeds, and the
    first two moments of the seed-averaged local spectral measure, m1 = E[G_ww], m2 = E[(G^2)_ww].
    """
    lam, U = np.linalg.eigh(G)
    wt = np.abs(U[:, idx, :]) ** 2                                       # (S, K, N)
    decay = (1 - eta * lam)[:, :, None] ** np.arange(T)[None, None, :]   # (S, N, T)
    rho = np.einsum("skn,snt->tk", wt, decay) / G.shape[0]
    m1 = np.einsum("skn,sn->k", wt, lam) / G.shape[0]
    m2 = np.einsum("skn,sn->k", wt, lam**2) / G.shape[0]
    return rho, m1, m2


def two_moment_half_life(m1, m2, eta):
    """Half-life of a Gamma mixture of exponential decays whose rates have mean m1 and second
    moment m2: (1 + kappa t / k)^-k = 1/2 with shape k = m1^2 / (m2 - m1^2). The mean per-step
    decay rate is taken in discrete time, kappa = -log(1 - eta m1), to match gradient descent."""
    k = m1**2 / (m2 - m1**2)
    return k * (2 ** (1 / k) - 1) / -np.log1p(-eta * m1)


def fast_mode_share(G, idx, factor=10):
    """For each frequency in idx: the seed-averaged share of its local spectral weight, and of its
    mean rate E[G_ww], that sits on 'fast' eigenmodes, those with eigenvalue > factor * E[G_ww]."""
    lam, U = np.linalg.eigh(G)
    wt = np.abs(U[:, idx, :]) ** 2                                       # (S, K, N)
    m1 = np.einsum("skn,sn->k", wt, lam) / G.shape[0]
    fast = lam[:, None, :] > factor * m1[None, :, None]                  # (S, K, N)
    weight = (wt * fast).sum((0, 2)) / wt.sum((0, 2))
    m1_share = (wt * fast * lam[:, None, :]).sum((0, 2)) / (wt * lam[:, None, :]).sum((0, 2))
    return weight, m1_share
