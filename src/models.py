"""
Circuit ansatze for the three benchmark tasks:
  1. Function fitting (1D and 2D regression)
  2. VQE (transverse-field Ising model)
  3. Classification (binary, 2D input)
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


# ---------------------------------------------------------------------------
# Task 1: Function Fitting
# ---------------------------------------------------------------------------

def make_regression_circuit_1d(n_qubits=2, n_layers=4):
    """Return a QNode that maps a scalar x -> expectation value (1D regression)."""
    # +1 wire for Hadamard-test auxiliary qubit (needed by full metric tensor)
    dev = qml.device("lightning.qubit", wires=n_qubits + 1)

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
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

    return circuit


def make_regression_circuit_2d(n_qubits=2, n_layers=4):
    """Return a QNode that maps (x1, x2) -> expectation value (2D regression)."""
    dev = qml.device("lightning.qubit", wires=n_qubits + 1)

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
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

    return circuit


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
    """Exact ground-state energy via full diagonalisation."""
    H = make_ising_hamiltonian(n_qubits, J, h)
    mat = qml.matrix(H)
    return float(np.min(np.linalg.eigvalsh(mat)))


def make_vqe_circuit(n_qubits=4, n_layers=4, J=1.0, h=1.0):
    """Return (circuit, hamiltonian) for VQE on the Ising model."""
    dev = qml.device("lightning.qubit", wires=n_qubits + 1)
    H = make_ising_hamiltonian(n_qubits, J, h)

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def circuit(params):
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return qml.expval(H)

    return circuit, H


# ---------------------------------------------------------------------------
# Task 3: Classification
# ---------------------------------------------------------------------------

def make_classification_circuit(n_qubits=2, n_layers=4):
    """Return a QNode that maps 2D input -> expectation value for binary classification."""
    dev = qml.device("lightning.qubit", wires=n_qubits + 1)

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
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

    return circuit


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
