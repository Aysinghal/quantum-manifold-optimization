"""Runs experiments 1-5, writes results/ and figures/, and fills the acceptance table in README.md."""
import json
import os
import re
import time

import numpy as np

from src import experiments as ex
from src import figures

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


def fmt(v, p=3):
    if isinstance(v, (list, tuple, np.ndarray)):
        return ", ".join(fmt(x, p) for x in v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "never"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.{p}g}"


def acceptance(e1, binary, ternary, dom, gap, dp):
    checks, extra = [], {}

    def add(n, name, expected, measured, ok):
        checks.append(dict(id=n, check=name, expected=expected, measured=measured, passed=bool(ok)))

    # --- Experiment 1
    var_ratio, k_ratio = {}, {}
    for key, label, *_ in ex.THEORY_CONFIGS:
        var_ratio[label] = e1[f"{key}_var_mc"] / e1[f"{key}_var_pred"]
        k_ratio[label] = e1[f"{key}_kdiag_mc"] / e1[f"{key}_kdiag_pred"]
    allv = np.concatenate(list(var_ratio.values()))
    allk = np.concatenate(list(k_ratio.values()))
    add(1, "Exp 1: MC Var / predicted, every ω ≥ 1, all configurations", "0.88–1.08",
        f"{allv.min():.3f}–{allv.max():.3f}", allv.min() >= 0.80 and allv.max() <= 1.20)
    add(2, "Exp 1: MC kernel diagonal / ((L+1)(d²/2)·Var), every ω ≥ 1", "0.95–1.05",
        f"{allk.min():.3f}–{allk.max():.3f}", allk.min() >= 0.90 and allk.max() <= 1.10)
    v = e1["pauli1_L3_var_pred"]
    add(3, "Exp 1, pauli(1), L = 3: predicted Var at ω = 1, 2, 3", "6/81, 3/81, 1/81",
        ", ".join(f"{x * 81:.12g}/81" for x in v), np.allclose(v * 81, [6, 3, 1], rtol=1e-12, atol=0))
    extra["exp1_var_ratio"] = var_ratio
    extra["exp1_kernel_ratio"] = k_ratio

    # --- Experiment 2, binary
    hb = binary["half_life"]
    exp_b = np.array([5, 6, 7, 9, 14, 19, 52])
    add(4, "Exp 2 binary: half-lives for ω = 1…7", fmt(exp_b), fmt(hb),
        np.all(np.diff(hb) >= 0) and np.all(np.abs(hb / exp_b - 1) <= 0.30))

    # --- Experiment 2, ternary
    ht, red, om = ternary["half_life"], ternary["r1"], ternary["omega"]
    grp = {R: ht[red == R] for R in (4, 2, 1)}
    add(5, "Exp 2 ternary: half-lives by redundancy group",
        "R=4: 9; R=2: 19–27; R=1: 61–77",
        "; ".join(f"R={R}: {fmt(g.min())}–{fmt(g.max())}" for R, g in grp.items()),
        np.nanmax(grp[4]) < np.nanmin(grp[2]) and np.nanmax(grp[2]) < np.nanmin(grp[1])
        and not np.isnan(ht).any())
    ratio59 = ht[om == 5][0] / ht[om == 9][0]
    add(6, "Exp 2 ternary: half-life(ω = 5) / half-life(ω = 9)", "8.6", fmt(ratio59), ratio59 > 4)

    rate_meas = ex.ETA * ternary["G_diag"].mean(0)
    dev = rate_meas / ternary["rate_pred"] - 1
    add(7, "Exp 2 ternary: seed-mean kernel diagonal × η vs predicted rate "
           "(0.0903, 0.0451, 0.0226 for R = 4, 2, 1)", "largest deviation 10%",
        "group means " + ", ".join(f"{rate_meas[red == R].mean():.4f}" for R in (4, 2, 1))
        + f"; largest deviation {100 * np.abs(dev).max():.1f}%", np.abs(dev).max() <= 0.15)

    a = ternary["amps"][0]
    frozen_err = (np.abs(ternary["r"] - ternary["r_frozen"]).mean(0) / a).max()
    add(8, "Exp 2 ternary: max over (t, ω) of seed-mean |r − r_frozen| / a", "0.007",
        fmt(frozen_err), frozen_err < 0.03)

    slow = ht / ternary["half_life_pred"]
    slow_g = [float(np.mean(slow[red == R])) for R in (4, 2, 1)]
    add(9, "Exp 2 ternary: measured / predicted half-life, mean per group (R = 4, 2, 1)",
        "1.2, 1.5, 2.2", ", ".join(f"{x:.2f}" for x in slow_g),
        slow_g[0] < slow_g[1] < slow_g[2])

    # --- Experiment 3
    hi = 5
    others = om != hi
    pr = dom["half_life"] / ternary["half_life"]
    add(10, "Exp 3: projected half-life, dominant / equal, every ω", "0.9–1.4",
        f"{np.nanmin(pr):.2f}–{np.nanmax(pr):.2f}",
        not np.isnan(pr).any() and pr.min() >= 0.6 and pr.max() <= 1.6)
    ce, cd = ternary["rel_cross"], dom["rel_cross"]
    # A crossing that never happens within T counts as later than any that does.
    cd_f = np.where(np.isnan(cd), np.inf, cd)
    ce_f = np.where(np.isnan(ce), np.inf, ce)
    rises = int(np.sum(cd_f[others] > ce_f[others]))
    rr = cd[others] / ce[others]
    add(11, "Exp 3: relative-error crossing time rises for the non-dominant ω",
        "12 of 12 rise, by 1.04–5.2×",
        f"{rises} of 12 rise, ratios {np.nanmin(rr):.2f}–{np.nanmax(rr):.2f}×"
        + (f" ({int(np.isnan(rr).sum())} never cross within T)" if np.isnan(rr).any() else ""),
        rises >= 9)
    k = hi - 1
    add(12, "Exp 3: relative-error crossing time of ω = 5", "falls from 94 to 69",
        f"{fmt(ce[k])} → {fmt(cd[k])}", cd_f[k] < ce_f[k])

    # --- Experiment 4
    def mean_rel_err(enc, key):
        return float(np.mean(np.abs(gap[f"{enc}_{key}"] / gap[f"{enc}_half_life"] - 1)))

    err2 = [mean_rel_err(e, "half_life_2mom") for e in ("binary", "ternary")]
    err_avg = [mean_rel_err(e, "half_life_avg") for e in ("binary", "ternary")]
    add(13, "Exp 4: two-moment prediction vs measured half-life, mean |relative error| "
            "(binary, ternary)", "5.5%, 6.1% (averaged kernel: 21%, 35%)",
        f"{100 * err2[0]:.1f}%, {100 * err2[1]:.1f}% (averaged kernel: "
        f"{100 * err_avg[0]:.0f}%, {100 * err_avg[1]:.0f}%)",
        max(err2) <= 0.10 and all(a < b for a, b in zip(err2, err_avg)))

    gred = gap["ternary_r1"]
    def group_ratio(key):
        return [float(np.mean((gap[f"ternary_{key}"] / gap["ternary_half_life_avg"])[gred == R]))
                for R in (4, 2, 1)]
    diag_g, meas_g = group_ratio("half_life_diag"), group_ratio("half_life")
    add(14, "Exp 4 ternary: half-life / averaged-kernel prediction, each seed's own G_ww only "
            "vs measured (R = 4, 2, 1)", "1.04, 1.14, 1.38 vs 1.16, 1.49, 2.19",
        ", ".join(f"{x:.2f}" for x in diag_g) + " vs " + ", ".join(f"{x:.2f}" for x in meas_g),
        all(1 <= a < b for a, b in zip(diag_g, meas_g)))

    err_spec = [mean_rel_err(e, "half_life_spec") for e in ("binary", "ternary")]
    add(15, "Exp 4: phase-averaged per-seed kernel vs measured half-life, mean |relative error| "
            "(binary, ternary)", "4.4%, 6.0%",
        f"{100 * err_spec[0]:.1f}%, {100 * err_spec[1]:.1f}%", max(err_spec) <= 0.10)

    # --- Experiment 5
    keys = [k for k, *_ in ex.DEPTH_CONFIGS]
    m1r = np.concatenate([dp[f"{k}_m1"] / dp[f"{k}_g_pred"] for k in keys])
    naive_r = {k: dp[f"{k}_m1"] / dp[f"{k}_g_naive"] for k in keys}
    naive_max = max(float(np.max(np.maximum(v, 1 / v))) for v in naive_r.values())
    add(16, "Exp 5: seed-mean kernel diagonal / corrected prediction, every ω, all depth configs "
            "(naive R(ω), matched at ω = 1, for contrast)", "0.95–1.28 (naive: off by up to 2.96×)",
        f"{m1r.min():.2f}–{m1r.max():.2f} (naive: off by up to {naive_max:.2f}×)",
        m1r.min() >= 0.8 and m1r.max() <= 1.4 and naive_max > 1.5)

    def ranks(x):
        return np.argsort(np.argsort(x)).astype(float)

    def spearman(x, y):
        return float(np.corrcoef(ranks(x), ranks(y))[0, 1])

    hl5 = dp["ternary3_L2_half_life"]
    ok5 = ~np.isnan(hl5)
    sp_c = spearman(hl5[ok5], dp["ternary3_L2_half_life_avg"][ok5])
    sp_w = spearman(hl5[ok5], dp["ternary3_L2_omega"][ok5])
    add(17, "Exp 5 ternary(3), L = 2: rank correlation of measured half-lives with the corrected "
            "prediction, and with ω", "0.99, 0.97", f"{sp_c:.2f}, {sp_w:.2f}", sp_c >= 0.95)

    l1_max = float(np.max(gap["ternary_half_life"] / gap["ternary_half_life_avg"]))
    gap5 = [float(np.nanmax(dp[f"{k}_half_life"] / dp[f"{k}_half_life_avg"])) for k in keys]
    add(18, "Exp 5: largest measured / averaged-kernel half-life (pauli(1) L = 3, binary(2) L = 2, "
            "ternary(3) L = 2; L = 1 largest: 2.5)", "4.3, 8.1, 4.4",
        ", ".join(f"{x:.1f}" for x in gap5) + f" (L = 1: {l1_max:.1f})",
        all(x > 1.2 * l1_max for x in gap5))

    def spec_err(k):
        hl, sp = dp[f"{k}_half_life"], dp[f"{k}_half_life_spec"]
        m = ~np.isnan(hl) & ~np.isnan(sp)
        return float(np.mean(np.abs(sp[m] / hl[m] - 1)))

    errs5 = [spec_err(k) for k in keys]
    add(19, "Exp 5: each run's frozen kernel (phase-averaged) vs measured half-life, mean |relative "
            "error| per config", "8.7%, 3.5%, 5.2%", ", ".join(f"{100 * x:.1f}%" for x in errs5),
        max(errs5) <= 0.15)

    lk = [("binary2_L2", 6), ("ternary3_L2", 25)]
    leak = [(float(dp[f"{k}_leak_weight"][w - 1]), float(dp[f"{k}_leak_m1_share"][w - 1])) for k, w in lk]
    add(20, "Exp 5: slowest learned frequency (binary(2) L = 2 ω = 6; ternary(3) L = 2 ω = 25): share of "
            "local weight on fast modes / share of E[G_ωω] they carry", "2.2% / 49%; 2.0% / 47%",
        "; ".join(f"{100 * a:.1f}% / {100 * b:.0f}%" for a, b in leak),
        all(a < 0.05 and b > 0.3 for a, b in leak))

    Gd = ternary["G_diag"]
    rel_std = Gd.std(0) / Gd.mean(0)
    extra.update(
        binary_half_life=hb, binary_half_life_pred=binary["half_life_pred"],
        ternary_half_life=ht, ternary_half_life_pred=ternary["half_life_pred"],
        ternary_r1=red, ternary_rate_measured=rate_meas, ternary_rate_pred=ternary["rate_pred"],
        ternary_kernel_rel_std_by_group={f"R={R}": float(rel_std[red == R].mean()) for R in (4, 2, 1)},
        ternary_slowdown_by_group=dict(zip(["R=4", "R=2", "R=1"], slow_g)),
        frozen_kernel_max_error=frozen_err,
        exp3_half_life_equal=ht, exp3_half_life_dominant=dom["half_life"],
        exp3_rel_cross_equal=ce, exp3_rel_cross_dominant=cd,
        exp4_two_moment_rel_err=dict(binary=err2[0], ternary=err2[1]),
        exp4_avg_kernel_rel_err=dict(binary=err_avg[0], ternary=err_avg[1]),
        exp4_spectral_rel_err=dict(binary=err_spec[0], ternary=err_spec[1]),
        exp4_ternary_slowdown_diag_only=dict(zip(["R=4", "R=2", "R=1"], diag_g)),
        exp4_ternary_slowdown_measured=dict(zip(["R=4", "R=2", "R=1"], meas_g)),
        exp4_ternary_spread_share_of_log_slowdown=dict(
            zip(["R=4", "R=2", "R=1"], [float(np.log(a) / np.log(b)) for a, b in zip(diag_g, meas_g)])),
        exp4_k_diag=dict(binary=gap["binary_k_diag"], ternary=gap["ternary_k_diag"]),
        exp4_half_life_2mom=dict(binary=gap["binary_half_life_2mom"], ternary=gap["ternary_half_life_2mom"]),
        exp5_m1_over_corrected=m1r, exp5_naive_max_error=naive_max,
        exp5_ternary_L2_spearman=dict(corrected=sp_c, omega=sp_w),
        exp5_max_slowdown=dict(zip(keys, gap5)), exp5_spectral_rel_err=dict(zip(keys, errs5)),
        exp5_leak=dict((f"{k} w={w}", dict(weight=a, rate_share=b)) for (k, w), (a, b) in zip(lk, leak)),
        exp5_half_life={k: dp[f"{k}_half_life"] for k in keys},
        exp5_half_life_avg={k: dp[f"{k}_half_life_avg"] for k in keys},
        exp5_frozen_err={k: float(dp[f"{k}_frozen_err"]) for k in keys},
        init_output_max_abs=max(float(binary["f0_max"]), float(ternary["f0_max"]), float(dom["f0_max"])),
    )
    return checks, extra


def to_json(o):
    if isinstance(o, dict):
        return {str(k): to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_json(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def acceptance_markdown(checks, runtime):
    lines = ["| # | Check | Expected (prototype) | Measured | Result |", "| --- | --- | --- | --- | --- |"]
    for c in checks:
        lines.append(f"| {c['id']} | {c['check']} | {c['expected']} | {c['measured']} | "
                     f"{'pass' if c['passed'] else '**FAIL**'} |")
    n = sum(c["passed"] for c in checks)
    lines.append("")
    lines.append(f"{n} of {len(checks)} checks pass. Last full run: {runtime:.0f} s.")
    return "\n".join(lines)


def update_readme(table):
    path = os.path.join(HERE, "README.md")
    if not os.path.exists(path):
        return
    text = open(path, encoding="utf-8").read()
    new = re.sub(r"(<!-- ACCEPTANCE:START -->).*?(<!-- ACCEPTANCE:END -->)",
                 lambda m: f"{m.group(1)}\n{table}\n{m.group(2)}", text, flags=re.S)
    open(path, "w", encoding="utf-8").write(new)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    timings = {}

    def timed(name, fn):
        t = time.time()
        out = fn()
        timings[name] = time.time() - t
        print(f"[{timings[name]:6.1f} s] {name}", flush=True)
        return out

    e1 = timed("experiment 1: theory check", ex.theory_check)
    binary, ternary = timed("experiment 2: learning order", ex.learning_order)
    dom = timed("experiment 3: amplitude", ex.amplitude)
    gap = timed("experiment 4: kernel gap", lambda: ex.kernel_gap(binary, ternary))
    dp = timed("experiment 5: depth", ex.depth)

    np.savez_compressed(os.path.join(RESULTS, "exp1_theory.npz"), **e1)
    np.savez_compressed(os.path.join(RESULTS, "exp2_binary.npz"), **binary)
    np.savez_compressed(os.path.join(RESULTS, "exp2_ternary.npz"), **ternary)
    np.savez_compressed(os.path.join(RESULTS, "exp3_amplitude.npz"), **dom)
    np.savez_compressed(os.path.join(RESULTS, "exp4_kernel_gap.npz"), **gap)
    np.savez_compressed(os.path.join(RESULTS, "exp5_depth.npz"), **dp)

    paths = timed("figures", lambda: figures.make_all(RESULTS, FIGURES))
    checks, extra = acceptance(e1, binary, ternary, dom, gap, dp)
    runtime = time.time() - t0

    summary = dict(config=dict(eta=ex.ETA, T=ex.T, seeds=[int(ex.SEEDS[0]), int(ex.SEEDS[-1])]),
                   runtime_seconds=runtime, timings_seconds=timings,
                   acceptance=checks, numbers=extra)
    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(to_json(summary), fh, indent=2, ensure_ascii=False)

    table = acceptance_markdown(checks, runtime)
    update_readme(table)
    print("\n" + table)
    print("\nfigures:")
    for p in paths:
        print("  " + os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
