import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.stats import shapiro, mannwhitneyu


def get_max_values_from_runs(data: dict[int, pd.DataFrame], parameter: str) -> np.ndarray:
    dfs = [data[k] for k in sorted(data.keys())]

    return np.array([
        df[parameter].to_numpy().max()
        for df in dfs
    ])


def shapiro_wilk(
        dict_a: dict[int, pd.DataFrame],
        dict_b: dict[int, pd.DataFrame],
        parameter: str,
        alpha: float = 0.05
) -> None:
    a_data = get_max_values_from_runs(dict_a, parameter)
    b_data = get_max_values_from_runs(dict_b, parameter)

    stat_a, p_a = shapiro(a_data)
    stat_b, p_b = shapiro(b_data)

    print(f"Shapiro–Wilk test for '{parameter}':")
    print(f"A: stat={stat_a:.7f}, p={p_a:.7f}")
    print(f"B: stat={stat_b:.7f}, p={p_b:.7f}")

    if p_a > alpha:
        print(f"\t-> Query A: Data looks normally distributed (p={p_a:.7f} > 0.05)")
    else:
        print(f"\t-> Query A: Data is NOT normally distributed (p={p_a:.7f} ≤ 0.05)")

    if p_b > alpha:
        print(f"\t-> Query B: Data looks normally distributed (p={p_b:.7f} > 0.05)")
    else:
        print(f"\t-> Query B: Data is NOT normally distributed (p={p_b:.7f} ≤ 0.05)")

    print("\n")


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size:
    δ = ( (#A>B) - (#B>A) ) / (nA * nB)
    """
    n_a = len(a)
    n_b = len(b)
    greater = 0
    smaller = 0

    for x in a:
        greater += np.sum(x > b)
        smaller += np.sum(x < b)

    return (greater - smaller) / (n_a * n_b)


def mann_whitney_u_test(
        dict_a: dict[int, pd.DataFrame],
        dict_b: dict[int, pd.DataFrame],
        parameter: str,
        alpha: float = 0.05
) -> None:
    a_data = get_max_values_from_runs(dict_a, parameter)
    b_data = get_max_values_from_runs(dict_b, parameter)

    stat, p = mannwhitneyu(a_data, b_data, alternative="two-sided")

    delta = cliffs_delta(a_data, b_data)

    print(f"Mann–Whitney U test for '{parameter}':")
    print(f"U-statistic={stat:.7f}, p={p:.7f}")
    print(f"Cliff's delta={delta:.7f}")

    if p > alpha:
        print(f"\t-> No significant difference (p={p:.7f} > {alpha})")
    else:
        print(f"\t-> Significant difference (p={p:.7f} ≤ {alpha})")

    abs_delta = abs(delta)
    if abs_delta < 0.147:
        effect = "negligible"
    elif abs_delta < 0.33:
        effect = "small"
    elif abs_delta < 0.474:
        effect = "medium"
    else:
        effect = "large"

    print(f"\t-> Effect size: {effect} (|δ|={abs_delta:.7f})")
    print("\n")


def linechart(
        dict_a: dict[int, pd.DataFrame],
        dict_b: dict[int, pd.DataFrame],
        parameter: str,
        label_a: str = "A",
        label_b: str = "B",
        title: str | None = None,
) -> None:
    iterations = sorted(dict_a.keys())
    x_axis = np.array(iterations, dtype=int)

    y_axis_a = get_max_values_from_runs(dict_a, parameter)
    y_axis_b = get_max_values_from_runs(dict_b, parameter)

    plt.figure(figsize=(10, 6))

    plt.plot(x_axis, y_axis_a, marker="o", label=label_a)
    plt.plot(x_axis, y_axis_b, marker="o", label=label_b)

    plt.xlabel("Iteration")
    plt.ylabel(parameter)

    if title is None:
        title = f"Comparison of '{parameter}' Across Runs"

    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
