"""Figures 1-4. Reads the .npz files in results/ and writes PNGs to figures/."""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

# One colour per single-layer redundancy value r1(omega), used in figures 2-4.
RCOL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100",
        5: "#e87ba4", 6: "#008300", 7: "#6250d6"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 200, "figure.dpi": 100,
})


def _redundancy_handles(values, marker=None):
    kw = dict(marker=marker, ls="", ms=6) if marker else dict(lw=2)
    return [Line2D([], [], color=RCOL[int(v)], label=f"redundancy {int(v)}", **kw)
            for v in sorted(set(int(v) for v in values), reverse=True)]


def _plain_log(axis, ticks):
    """Log axis with plain-number tick labels at the given positions."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _spread(y, gap, lo):
    """Push label positions apart so neighbours are at least `gap` apart."""
    order = np.argsort(y)
    out = np.array(y, float)
    prev = lo - gap
    for i in order:
        out[i] = max(out[i], prev + gap)
        prev = out[i]
    return out


def fig1(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, key, label in zip(axes, ["pauli1_L3", "pauli2_L2"], ["pauli(1), L = 3", "pauli(2), L = 2"]):
        om = res[f"{key}_omega"]
        norm = lambda v: v / v[-1]
        ax.bar(om, norm(res[f"{key}_var_mc"]), width=0.55, color="#b9b8b0",
               label="Monte Carlo Var($c_\\omega$)")
        ax.plot(om, norm(res[f"{key}_R"]), ls="", marker="o", ms=8, mfc="none", mec="#e34948",
                mew=1.8, label="naive: $R(\\omega)$")
        ax.plot(om, norm(res[f"{key}_var_pred"]), ls="", marker="D", ms=6, color=INK,
                label="corrected formula")
        ax.set_xticks(om)
        ax.set_xlabel("frequency $\\omega$")
        ax.set_title(label, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("value relative to top frequency")
    axes[0].legend(loc="upper right")
    fig.suptitle("Monte Carlo variance follows the corrected formula, not $R(\\omega)$, at L ≥ 2",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig2(binary, ternary, path, steps=150):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, res, label in zip(axes, [binary, ternary], ["binary(3)", "ternary(3)"]):
        rho, om, red = res["rho_mean"][:steps + 1], res["omega"], res["r1"]
        t = np.arange(len(rho))
        for k in np.argsort(-red):
            ax.plot(t, rho[:, k], color=RCOL[int(red[k])], lw=1.6)
        ends = rho[-1]
        ypos = _spread(ends, 0.043, 0.01)
        for k in range(len(om)):
            ax.annotate(f"{om[k]}", xy=(steps, ends[k]), xytext=(steps * 1.045, ypos[k]),
                        va="center", ha="left", fontsize=7.5, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=RCOL[int(red[k])], lw=0.9,
                                        shrinkA=0, shrinkB=1))
        ax.text(steps * 1.045, min(ypos.max() + 0.06, 1.0), "$\\omega$", fontsize=8, color=MUTED,
                va="center", clip_on=False)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, steps)
        ax.set_ylim(0, 1)
        ax.set_xlabel("GD step")
        ax.set_title(label, loc="left")
        ax.legend(handles=_redundancy_handles(red), loc="upper right",
                  ncol=2 if len(set(red)) > 4 else 1)
    axes[0].set_ylabel("projected residual $\\rho_\\omega$ (mean over 100 seeds)")
    fig.suptitle("Learning order follows redundancy: low-frequency-first for binary, "
                 "not ordered by $\\omega$ for ternary", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(w_pad=3.5)
    fig.savefig(path)
    plt.close(fig)


def fig3(binary, ternary, path, seed=0):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 5.3))
    for res, mk in [(binary, "o"), (ternary, "s")]:
        om, red = res["omega"], res["r1"]
        ax.scatter(om, res["half_life_pred"], marker=mk, s=70, facecolors="none",
                   edgecolors=MUTED, linewidths=1.1, zorder=2)
        ax.scatter(om, res["half_life"], marker=mk, s=34, c=[RCOL[int(v)] for v in red],
                   edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_yscale("log")
    _plain_log(ax.yaxis, [5, 10, 20, 50, 100])
    ax.set_xticks(np.arange(1, 14))
    ax.set_xlabel("frequency $\\omega$")
    ax.set_ylabel("half-life (GD steps)")
    ax.set_title("Half-life: measured vs averaged-kernel prediction", loc="left")
    shape = [Line2D([], [], ls="", marker="o", color=MUTED, ms=6, label="binary(3), measured"),
             Line2D([], [], ls="", marker="s", color=MUTED, ms=6, label="ternary(3), measured"),
             Line2D([], [], ls="", marker="o", mfc="none", mec=MUTED, ms=8,
                    label="averaged-kernel prediction (open)")]
    leg = ax.legend(handles=shape, loc="upper left", bbox_to_anchor=(0.0, -0.16), borderaxespad=0,
                    ncol=3, columnspacing=1.2, handletextpad=0.3)
    ax.add_artist(leg)
    ax.legend(handles=_redundancy_handles(np.concatenate([binary["r1"], ternary["r1"]]), "o"),
              loc="upper left", bbox_to_anchor=(0.0, -0.24), borderaxespad=0, ncol=4,
              columnspacing=1.2, handletextpad=0.3)

    a = ternary["amps"][0]
    t = np.arange(ternary["r"].shape[1])
    ends = []
    for w in (1, 5, 9):
        k = w - 1
        col = RCOL[int(ternary["r1"][k])]
        bx.plot(t, np.abs(ternary["r"][seed, :, k]) / a, color=col, lw=2.2)
        bx.plot(t, np.abs(ternary["r_frozen"][seed, :, k]) / a, color=INK, lw=0.9, ls=(0, (4, 3)))
        ends.append(np.abs(ternary["r"][seed, -1, k]) / a)
    bx.set_yscale("log")
    for w, y in zip((1, 5, 9), ends):
        bx.annotate(f"$\\omega$ = {w}", xy=(t[-1], y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, annotation_clip=False)
    bx.set_xlim(0, t[-1])
    bx.set_xlabel("GD step")
    bx.set_ylabel("$|r_\\omega(t)|\\,/\\,a$")
    bx.set_title(f"One run (ternary, seed {seed}): its own frozen kernel tracks it", loc="left")
    bx.legend(handles=[Line2D([], [], color=MUTED, lw=2.2, label="trained model"),
                       Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 3)), label="frozen-kernel prediction")]
              + _redundancy_handles(ternary["r1"][[0, 4]]), loc="lower left")
    fig.suptitle("The averaged kernel gives the order but underestimates half-lives; "
                 "the per-run kernel predicts the run", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(w_pad=3)
    fig.savefig(path)
    plt.close(fig)


def fig4(equal, dom, path, hi=5):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    panels = [("half_life", "Projected half-life: unchanged by amplitude"),
              ("rel_cross", "Relative-error crossing time: shifts with amplitude")]
    om, red = equal["omega"], equal["r1"]
    for ax, (key, title) in zip(axes, panels):
        x, y = equal[key], dom[key]
        lo = np.nanmin([x, y]) / 1.5
        up = np.nanmax([x, y]) * 1.5
        ax.plot([lo, up], [lo, up], color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.scatter(x, y, s=42, c=[RCOL[int(v)] for v in red], edgecolors="white",
                   linewidths=0.7, zorder=3)
        k = hi - 1
        ax.scatter([x[k]], [y[k]], s=170, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=4)
        ax.annotate(f"$\\omega$ = {hi}", xy=(x[k], y[k]), xytext=(10, -12),
                    textcoords="offset points", ha="left", va="top", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, up)
        ax.set_ylim(lo, up)
        ticks = [t for t in (5, 10, 20, 50, 100, 200) if lo <= t <= up]
        _plain_log(ax.xaxis, ticks)
        _plain_log(ax.yaxis, ticks)
        ax.set_aspect("equal")
        ax.set_xlabel("equal amplitudes (GD steps)")
        ax.set_ylabel(f"dominant $\\omega$ = {hi} (GD steps)")
        ax.set_title(title, loc="left")
    axes[0].legend(handles=_redundancy_handles(red, "o") +
                   [Line2D([], [], color=MUTED, lw=0.9, ls="--", label="y = x")], loc="lower right")
    fig.suptitle("Amplitude does not change the rate; the relative-error metric makes it look "
                 "as if it does", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path)
    plt.close(fig)


def make_all(results_dir="results", figures_dir="figures"):
    os.makedirs(figures_dir, exist_ok=True)
    load = lambda name: dict(np.load(os.path.join(results_dir, name)))
    e1, b, t, d = (load("exp1_theory.npz"), load("exp2_binary.npz"),
                   load("exp2_ternary.npz"), load("exp3_amplitude.npz"))
    paths = [os.path.join(figures_dir, n) for n in
             ("fig1_theory_check.png", "fig2_learning_order.png",
              "fig3_halflife.png", "fig4_amplitude.png")]
    fig1(e1, paths[0])
    fig2(b, t, paths[1])
    fig3(b, t, paths[2])
    fig4(t, d, paths[3])
    return paths
