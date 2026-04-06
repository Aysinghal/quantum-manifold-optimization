"""
Metric computation helpers and results serialisation (JSON).
"""

import json
import os
import numpy as np


def mse(predictions, targets):
    return float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))


def accuracy(predictions, targets):
    """Binary accuracy where sign(prediction) is the predicted class (+1 / -1)."""
    preds = np.sign(np.asarray(predictions))
    return float(np.mean(preds == np.asarray(targets)))


def steps_to_threshold(losses, threshold):
    """Number of steps before loss first drops below `threshold`. None if never reached."""
    for i, l in enumerate(losses):
        if l <= threshold:
            return i
    return None


def aggregate_seeds(results_by_seed):
    """Given {seed: {losses: [...], ...}}, compute mean/std across seeds.

    Returns dict with arrays: mean_losses, std_losses, mean_wall, std_wall, etc.
    """
    all_losses = np.array([r["losses"] for r in results_by_seed.values()])
    all_wall = np.array([r["wall_times"] for r in results_by_seed.values()])
    all_evals = np.array([r["circuit_evals"] for r in results_by_seed.values()])

    return {
        "mean_losses": all_losses.mean(axis=0).tolist(),
        "std_losses": all_losses.std(axis=0).tolist(),
        "mean_wall": all_wall.mean(axis=0).tolist(),
        "std_wall": all_wall.std(axis=0).tolist(),
        "mean_evals": all_evals.mean(axis=0).tolist(),
        "std_evals": all_evals.std(axis=0).tolist(),
        "final_loss_mean": float(all_losses[:, -1].mean()),
        "final_loss_std": float(all_losses[:, -1].std()),
    }


def save_results(results, path):
    """Save results dict to JSON (numpy arrays are converted to lists)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _convert(obj):
        if hasattr(obj, "numpy"):  # PennyLane autograd tensor
            return obj.numpy().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"  Results saved -> {path}")


def load_results(path):
    with open(path) as f:
        return json.load(f)
