from __future__ import annotations

import warnings as _warnings

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap, wilcoxon


def bootstrap_median_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    med = float(np.median(values))
    if len(values) < 2 or np.ptp(values) == 0:
        return med, med
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = scipy_bootstrap(
            data=(values,),
            statistic=np.median,
            n_resamples=n_bootstrap,
            confidence_level=confidence,
            method="BCa",
        )
    bca_failed = any(
        "BCa" in str(w.message)
        or "degenerate" in str(w.message).lower()
        or "empty slice" in str(w.message).lower()
        or "invalid value" in str(w.message).lower()
        for w in caught
    )
    if bca_failed:
        result = scipy_bootstrap(
            data=(values,),
            statistic=np.median,
            n_resamples=n_bootstrap,
            confidence_level=confidence,
            method="percentile",
        )
    return float(result.confidence_interval.low), float(
        result.confidence_interval.high
    )


def descriptive_stats(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    n = len(values)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if n > 1 else np.nan
    med = float(np.median(values))
    ci_lo, ci_hi = bootstrap_median_ci(values, n_bootstrap, confidence)

    return {
        "n": n,
        "median": med,
        "mean": mean_val,
        "std": std_val,
        "cv": std_val / mean_val if mean_val > 0 and n > 1 else np.nan,
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "median_ci_lower": ci_lo,
        "median_ci_upper": ci_hi,
    }


def vargha_delaney_a12(a: np.ndarray, b: np.ndarray) -> float:
    m, n = len(a), len(b)
    total = 0.0
    for ai in a:
        total += np.sum(ai > b) + 0.5 * np.sum(ai == b)
    return total / (m * n)


def classify_a12(a12: float) -> str:
    a = max(a12, 1 - a12)
    if a >= 0.71:
        return "large"
    if a >= 0.64:
        return "medium"
    if a >= 0.56:
        return "small"
    return "negligible"


def holm_bonferroni(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p_values)
    corrected = np.empty(m)
    for i, idx in enumerate(order):
        corrected[idx] = p_values[idx] * (m - i)
    sorted_corrected = corrected[order].copy()
    for i in range(1, m):
        sorted_corrected[i] = max(sorted_corrected[i], sorted_corrected[i - 1])
    corrected[order] = sorted_corrected
    corrected = np.minimum(corrected, 1.0)
    return corrected < alpha, corrected


def pairwise_comparison(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    stat, p_value = wilcoxon(a - b, alternative="two-sided")
    a12 = vargha_delaney_a12(a, b)

    ratios = a / np.where(b == 0, np.nan, b)
    ratios = ratios[~np.isnan(ratios)]
    ratio_lo, ratio_hi = bootstrap_median_ci(ratios, n_bootstrap, confidence)

    diffs = a - b
    diff_lo, diff_hi = bootstrap_median_ci(diffs, n_bootstrap, confidence)

    return {
        "n_paired": len(a),
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "a12": float(a12),
        "a12_category": classify_a12(a12),
        "ratio_median": float(np.median(ratios)) if len(ratios) > 0 else np.nan,
        "ratio_ci_lower": ratio_lo,
        "ratio_ci_upper": ratio_hi,
        "diff_median": float(np.median(diffs)),
        "diff_ci_lower": diff_lo,
        "diff_ci_upper": diff_hi,
    }


def cross_pass_aggregation(pass_medians: np.ndarray) -> dict:
    headline = float(np.median(pass_medians))
    spread = float(np.ptp(pass_medians))
    n = len(pass_medians)
    return {
        "headline_median": headline,
        "pass_1_median": float(pass_medians[0]) if n > 0 else np.nan,
        "pass_2_median": float(pass_medians[1]) if n > 1 else np.nan,
        "pass_3_median": float(pass_medians[2]) if n > 2 else np.nan,
        "pass_range": spread,
        "pass_range_relative": spread / headline if headline > 0 else np.nan,
        "consistent": bool(spread / headline <= 0.10) if headline > 0 else False,
    }
