"""
IBD Analysis - F-Matrix Symbolic Computation

This package computes the stationary F-matrix (Identity By Descent probability)
using symbolic computation.
"""

from .f_matrix_symbolic import (
    compute_f_matrix_stationary,
    load_results_by_case,
    save_results,
    get_case_name,
)

__all__ = [
    'compute_f_matrix_stationary',
    'load_results_by_case',
    'save_results',
    'get_case_name',
]
