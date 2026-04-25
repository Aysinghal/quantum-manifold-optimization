"""
Circuit ansatze for the three benchmark tasks:
  1. Function fitting (1D and 2D regression)
  2. VQE (transverse-field Ising model)
  3. Classification (binary, 2D input)
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


def _resolve_diff_method(shots, force=None):
    """Pick diff_method for a QNode.

    `shots` > 0 forces parameter-shift (adjoint can't sample).
    `force` overrides automatic selection (used by QNG-variant workers,
    since qml.metric_tensor builds tapes that lightning.qubit's adjoint
    implementation rejects). Pass force="parameter-shift" for QNG.
    """
    if force is not None:
        return force
    return "parameter-shift" if shots else "adjoint"


def _maybe_with_shots(qnode, shots):
    """Attach a shot count to a QNode via the non-deprecated set_shots transform."""
    if shots:
        return qml.set_shots(qnode, shots=shots)
    return qnode


# ---------------------------------------------------------------------------
# Task 1: Function Fitting
# ---------------------------------------------------------------------------

def make_regression_circuit_1d(n_qubits=2, n_layers=4, shots=None, diff_method=None):
    """Return a QNode that maps a scalar x -> expectation value (1D regression)."""
    # +1 wire for Hadamard-test auxiliary qubit (needed by full metric tensor)
    dev = qml.device("lightning.qubit", wires=n_qubits + 1)

    @qml.qnode(dev, interface="autograd", diff_method=_resolve_diff_method(shots, diff_method))
    def circuit(params, x):
        # params shape: (n_layers+1, n_qubits, 3)  -- StronglyEntanglingLayers
        n_w = n_qubits
        n_rot = 3
        for layer in range(params.shape[0] - 1):
            qml.StronglyEntanglingLayers(
                params[layer].reshape(1, n_w, n_rot), wires=range(n_w)
            )
            # data encoding: embed the same scalar on every qubit via RZ
            for w in range(n_w):
                qml.RZ(x, wires=w)
        # final trainable block (no encoding after it)
        qml.StronglyEntanglingLayers(
            params[-1].reshape(1, n_w, n_rot), wires=range(n_w)
        )
        return qml.expval(qml.PauliZ(0))

    return _maybe_with_shots(circuit, shots)


def make_regression_circuit_2d(n_qubits=2, n_layers=4, shots=None, diff_method=None):
    """Return a QNode that maps (x1, x2) -> expectation value (2D regression)."""
    dev = qml.device("lightning.qubit", wires=n_qubits + 1)

    @qml.qnode(dev, interface="autograd", diff_method=_resolve_diff_method(shots, diff_method))
    def circuit(params, x):
        n_w = n_qubits
        n_rot = 3
        for layer in range(params.shape[0] - 1):
            qml.StronglyEntanglingLayers(
                params[layer].reshape(1, n_w, n_rot), wires=range(n_w)
            )
            qml.AngleEmbedding(x, wires=range(n_w), rotation="Z")
        qml.StronglyEntanglingLayers(
            params[-1].reshape(1, n_w, n_rot), wires=range(n_w)
        )
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return _maybe_with_shots(circuit, shots)


# ---------------------------------------------------------------------------
# Task 2: VQE  --  Transverse-field Ising model
# ---------------------------------------------------------------------------

def make_ising_hamiltonian(n_qubits, J=1.0, h=1.0):
    """H = -J sum_i Z_i Z_{i+1}  -  h sum_i X_i   (open boundary)."""
    coeffs = []
    ops = []
    for i in range(n_qubits - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def exact_ground_energy(n_qubits, J=1.0, h=1.0):
    """Exact TFIM ground-state energy via full diagonalisation. Kept for
    backward compatibility with the original `vqe` task; new tasks should
    use `exact_ground_energy_from_h` with their own Hamiltonian."""
    H = make_ising_hamiltonian(n_qubits, J, h)
    return exact_ground_energy_from_h(H)


def exact_ground_energy_from_h(hamiltonian):
    """Exact ground-state energy for any PennyLane Hamiltonian via dense
    diagonalisation. Only feasible for ~<=12 qubits."""
    mat = qml.matrix(hamiltonian)
    return float(np.min(np.linalg.eigvalsh(mat)))


def make_stokes_hamiltonian(n_qubits=6):
    """H = Z_0 (x) Z_1 only.

    Stokes-style "rigged" Hamiltonian: only 2 of the n_qubits enter the loss,
    so the remaining circuit parameters are "dead weight" that the QNG metric
    tensor can down-weight. This is the regime where vanilla QNG was designed
    to dominate. n_qubits is just the circuit width; the Hamiltonian itself
    only touches wires 0 and 1.
    """
    return qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)])


def make_sk_hamiltonian(n_qubits, seed=42):
    """Sherrington-Kirkpatrick spin-glass Hamiltonian.

    ``H = sum_{i<j} J_{ij} Z_i Z_j`` with ``J_{ij} ~ U(-1, 1)``. All-to-all
    couplings on Z, drawn from a fixed seed so the *Hamiltonian* is identical
    across runs (only the optimizer initialisation varies with `seed` upstream).
    Frustrated and rugged -- a notoriously hard landscape that should stress
    every optimizer; included in Phase 1 as a "hard real-world" check that
    R-QNG-torus does not collapse on a non-toy problem.
    """
    rng = np.random.default_rng(seed)
    coeffs, ops = [], []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coeffs.append(float(rng.uniform(-1.0, 1.0)))
            ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)


def make_xxz_hamiltonian(n_qubits, delta=1.0):
    """XXZ chain (open boundary).

    ``H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + delta * Z_i Z_{i+1})``. Setting
    ``delta = 1`` recovers the isotropic Heisenberg chain; ``delta != 1`` is
    the anisotropic XXZ regime. Smooth-but-rugged condensed-matter benchmark;
    contrasts with the Stokes "rigged" Hamiltonian where most circuit wires
    are dead weight.
    """
    coeffs, ops = [], []
    for i in range(n_qubits - 1):
        j = i + 1
        coeffs.append(1.0); ops.append(qml.PauliX(i) @ qml.PauliX(j))
        coeffs.append(1.0); ops.append(qml.PauliY(i) @ qml.PauliY(j))
        coeffs.append(float(delta)); ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)


def make_heisenberg_ring_hamiltonian(n_qubits, J=1.0, periodic=True):
    """H = J * sum_<i,j> (X_i X_j + Y_i Y_j + Z_i Z_j)  on a ring (periodic) or
    open chain. Antiferromagnetic for J > 0. Ground-state landscape has
    multiple local minima (frustration on odd-length rings, near-degeneracies
    on even-length rings) but is otherwise smooth -- the regime where
    momentum-augmented QNG should beat both vanilla QNG (which gets stuck on
    plateaus) and Adam (which lacks the geometric preconditioner)."""
    coeffs, ops = [], []
    n_terms = n_qubits if periodic else n_qubits - 1
    for i in range(n_terms):
        j = (i + 1) % n_qubits
        coeffs.append(J); ops.append(qml.PauliX(i) @ qml.PauliX(j))
        coeffs.append(J); ops.append(qml.PauliY(i) @ qml.PauliY(j))
        coeffs.append(J); ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)


def make_vqe_circuit(n_qubits=4, n_layers=4, J=1.0, h=1.0,
                     hamiltonian=None, shots=None, diff_method=None):
    """Return (circuit, hamiltonian) for VQE.

    If `hamiltonian` is provided, it is used as-is (the J/h kwargs are ignored).
    Otherwise the default transverse-field Ising Hamiltonian is built from
    n_qubits, J, h. Existing callers that pass (n_qubits, n_layers, J, h)
    continue to work unchanged.
    """
    dev = qml.device("lightning.qubit", wires=n_qubits + 1)
    if hamiltonian is None:
        hamiltonian = make_ising_hamiltonian(n_qubits, J, h)

    @qml.qnode(dev, interface="autograd", diff_method=_resolve_diff_method(shots, diff_method))
    def circuit(params):
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return qml.expval(hamiltonian)

    return _maybe_with_shots(circuit, shots), hamiltonian


# ---------------------------------------------------------------------------
# Task 3: Classification
# ---------------------------------------------------------------------------

def make_classification_circuit(n_qubits=2, n_layers=4, shots=None, diff_method=None):
    """Return a QNode that maps 2D input -> expectation value for binary classification."""
    dev = qml.device("lightning.qubit", wires=n_qubits + 1)

    @qml.qnode(dev, interface="autograd", diff_method=_resolve_diff_method(shots, diff_method))
    def circuit(params, x):
        # params shape: (n_layers+1, n_qubits, 3)
        n_w = n_qubits
        n_rot = 3
        for layer in range(params.shape[0] - 1):
            qml.StronglyEntanglingLayers(
                params[layer].reshape(1, n_w, n_rot), wires=range(n_w)
            )
            qml.AngleEmbedding(x, wires=range(n_w), rotation="Z")
        qml.StronglyEntanglingLayers(
            params[-1].reshape(1, n_w, n_rot), wires=range(n_w)
        )
        return qml.expval(qml.PauliZ(0))

    return _maybe_with_shots(circuit, shots)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_params_regression(n_qubits=2, n_layers=4, seed=42):
    """Random initial parameters for regression circuits: shape (n_layers+1, n_qubits, 3)."""
    rng = np.random.default_rng(seed)
    shape = (n_layers + 1, n_qubits, 3)
    return pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)


def init_params_vqe(n_qubits=4, n_layers=4, seed=42):
    """Random initial parameters for VQE: shape matching StronglyEntanglingLayers."""
    rng = np.random.default_rng(seed)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)


def init_on_manifold(params, manifold, target_radius=None):
    """Project initial parameters onto the optimizer's manifold so the very
    first training step starts from a feasible point.

    Sphere case (the only non-trivial one): rescale the flattened parameter
    vector to ``||theta_flat|| = R``. The default radius is the **natural
    shell** of a uniform-on-``[0, 2*pi)`` initialization,

        ``R* = 2*pi * sqrt(d / 3)``  with ``d = len(theta_flat)``,

    derived from ``E[theta_i^2] = 4*pi^2/3`` for ``theta_i ~ U(0, 2*pi)``.
    Empirically this also tracks where flat optimizers (Adam, QNG) settle
    after a long run on these tasks, which is what makes the sphere
    constraint a fair comparison rather than a strawman: at ``R=1`` the
    sphere optimizers are forced into a vanishingly small parameter range
    and lose by construction. Pass ``target_radius`` to override.

    Euclidean and Torus return the input unchanged (Torus retract handles
    wrapping itself; the initial random uniform draw on ``[0, 2*pi)``
    already lies on the torus).

    The shape and ``requires_grad`` flag of the input are preserved so the
    rest of the training loop sees an identical-looking ``pnp.array``.
    """
    if manifold is None:
        return params
    if getattr(manifold, "name", None) != "sphere":
        return params
    arr = np.array(params)
    flat = arr.flatten()
    d = flat.size
    if target_radius is None:
        target_radius = 2.0 * np.pi * np.sqrt(d / 3.0)
    norm = float(np.linalg.norm(flat))
    if norm < 1e-12:
        return params
    flat = flat * (target_radius / norm)
    return pnp.array(flat.reshape(arr.shape), requires_grad=True)
