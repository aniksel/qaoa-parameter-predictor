import numpy as np
import matplotlib.pyplot as plt

# QAOA angle ranges: beta in [0, pi], gamma in [0, 2*pi]
BETA_RANGE = (0.0, np.pi)
GAMMA_RANGE = (0.0, 2.0 * np.pi)


def beta_gamma_arrays(df, p, seed=None):
    sub = df[df["p"] == p]
    if seed is not None:
        sub = sub[sub["seed"] == seed]
    beta = np.array(sub["beta"].tolist(), dtype=float)
    gamma = np.array(sub["gamma"].tolist(), dtype=float)
    return beta, gamma


def wrap_to_range(values, lo, hi):
    # fold periodic angles into [lo, hi)
    return lo + np.mod(np.asarray(values, dtype=float) - lo, hi - lo)


def circular_heatmap(values, ax, title, value_range, bins=36, cmap="viridis",
                     r_inner=0.7, r_outer=1.0):
    lo, hi = value_range
    angles = wrap_to_range(values, lo, hi).ravel()

    # count predicted angles falling into each angular sector
    counts, edges = np.histogram(angles, bins=bins, range=(lo, hi))

    theta, r = np.meshgrid(edges, [r_inner, r_outer])
    mesh = ax.pcolormesh(theta, r, counts[np.newaxis, :], cmap=cmap, shading="flat")

    ax.set_thetamin(np.degrees(lo))
    ax.set_thetamax(np.degrees(hi))
    ax.set_ylim(0, r_outer)
    ax.set_yticks([])
    ax.set_title(title, pad=20)
    plt.colorbar(mesh, ax=ax, pad=0.1, label="count")


def plot_param_concentration(df, p, seed=None, bins=36, cmap="viridis"):
    beta, gamma = beta_gamma_arrays(df, p, seed)
    tag = f"p={p}" if seed is None else f"p={p}, seed={seed}"

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"}
    )
    circular_heatmap(beta, axes[0], f"Beta on unit circle ({tag})", BETA_RANGE, bins, cmap)
    circular_heatmap(gamma, axes[1], f"Gamma on unit circle ({tag})", GAMMA_RANGE, bins, cmap)
    plt.tight_layout()
    plt.show()


def plot_all(df, bins=36, cmap="viridis"):
    for p in sorted(df["p"].unique()):
        for seed in sorted(df[df["p"] == p]["seed"].unique()):
            plot_param_concentration(df, p, seed, bins, cmap)