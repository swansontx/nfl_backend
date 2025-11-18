"""Smoothing & rolling feature utilities

Placeholder functions for empirical-Bayes smoothing, rolling averages, etc.
"""


def eb_smooth(counts, alpha=1.0):
    # naive prior smoothing placeholder
    total = sum(counts)
    n = len(counts)
    return [(c+alpha)/(1+alpha*n) for c in counts]
