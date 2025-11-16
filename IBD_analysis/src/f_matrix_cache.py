"""
F-matrix caching system for concentric parameter space analysis.

Caches only F-matrices (not distance calculations or concentric judgments),
allowing fast recomputation with different distance metrics.
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional, List, Tuple, Dict


def make_cache_key(N: float, m: float, alpha: float) -> Tuple[float, float, float]:
    """
    Create a cache key from parameters with appropriate rounding.

    Args:
        N, m, alpha: Parameter values

    Returns:
        Rounded tuple suitable for dictionary keys
    """
    return (round(N, 10), round(m, 15), round(alpha, 15))


def f_matrix_to_columns(F_matrix: np.ndarray, M: int) -> Dict[str, float]:
    """
    Convert F-matrix to dictionary of column values.

    Args:
        F_matrix: (M, M) F-matrix
        M: Number of agents

    Returns:
        Dictionary with keys 'F_i_j' for i, j in [0, M)

    Examples:
        >>> F = np.array([[0.5, 0.2], [0.2, 0.5]])
        >>> f_matrix_to_columns(F, 2)
        {'F_0_0': 0.5, 'F_0_1': 0.2, 'F_1_0': 0.2, 'F_1_1': 0.5}
    """
    result = {}
    for i in range(M):
        for j in range(M):
            result[f'F_{i}_{j}'] = float(F_matrix[i, j])
    return result


def columns_to_f_matrix(row: pd.Series, M: int) -> np.ndarray:
    """
    Reconstruct F-matrix from DataFrame row.

    Args:
        row: DataFrame row with 'F_i_j' columns
        M: Number of agents

    Returns:
        (M, M) F-matrix

    Examples:
        >>> row = pd.Series({'F_0_0': 0.5, 'F_0_1': 0.2, 'F_1_0': 0.2, 'F_1_1': 0.5})
        >>> columns_to_f_matrix(row, 2)
        array([[0.5, 0.2],
               [0.2, 0.5]])
    """
    F = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            F[i, j] = row[f'F_{i}_{j}']
    return F


class FMatrixCache:
    """
    Cache for F-matrices computed at different parameter combinations.

    Stores F-matrices indexed by (N, m, alpha) parameters.
    Does NOT store distance calculations or concentric judgments,
    which can be quickly recomputed from cached F-matrices.

    Directory structure:
        cache_dir/
        ├── M3/
        │   ├── case1.parquet
        │   ├── case2.parquet
        │   └── ...
        ├── M5/
        │   └── ...
        └── metadata.json

    Examples:
        >>> cache = FMatrixCache('results/f_matrix_cache', M=3)
        >>>
        >>> # Save results
        >>> results = [
        ...     {'N': 100, 'm': 0.01, 'alpha': 0.001, 'F_matrix': F1},
        ...     {'N': 200, 'm': 0.02, 'alpha': 0.002, 'F_matrix': F2},
        ... ]
        >>> cache.append_results('case1', results)
        >>>
        >>> # Load cached F-matrix
        >>> F = cache.get_f_matrix('case1', N=100, m=0.01, alpha=0.001)
        >>>
        >>> # Check what's missing
        >>> params = [(100, 0.01, 0.001), (300, 0.03, 0.003)]
        >>> missing = cache.get_missing_params('case1', params)
    """

    def __init__(self, cache_dir: str, M: int):
        """
        Initialize cache.

        Args:
            cache_dir: Base directory for cache storage
            M: Number of agents
        """
        self.cache_dir = cache_dir
        self.M = M
        self.cache_path = os.path.join(cache_dir, f'M{M}')
        self.metadata_path = os.path.join(cache_dir, 'metadata.json')

        os.makedirs(self.cache_path, exist_ok=True)

    def _get_case_file(self, case: str) -> str:
        """Get filepath for case."""
        return os.path.join(self.cache_path, f'{case}.parquet')

    def load_case(self, case: str) -> pd.DataFrame:
        """
        Load cached F-matrices for a case.

        Args:
            case: Case name (e.g., 'case1', 'case2')

        Returns:
            DataFrame with columns: N, m, alpha, F_0_0, F_0_1, ..., F_{M-1}_{M-1}
            Empty DataFrame if cache doesn't exist
        """
        filepath = self._get_case_file(case)

        if not os.path.exists(filepath):
            # Return empty DataFrame with correct columns
            columns = ['N', 'm', 'alpha']
            for i in range(self.M):
                for j in range(self.M):
                    columns.append(f'F_{i}_{j}')
            return pd.DataFrame(columns=columns)

        return pd.read_parquet(filepath)

    def save_case(self, case: str, df: pd.DataFrame):
        """
        Save F-matrix DataFrame for a case.

        Args:
            case: Case name
            df: DataFrame with N, m, alpha, and F_i_j columns
        """
        filepath = self._get_case_file(case)
        df.to_parquet(filepath, index=False)

        # Update metadata
        self._update_metadata(case)

    def get_f_matrix(self, case: str, N: float, m: float, alpha: float) -> Optional[np.ndarray]:
        """
        Get cached F-matrix for specific parameters.

        Args:
            case: Case name
            N, m, alpha: Parameter values

        Returns:
            (M, M) F-matrix if cached, None otherwise
        """
        df = self.load_case(case)

        if len(df) == 0:
            return None

        # Match parameters with rounding
        key = make_cache_key(N, m, alpha)

        matches = df[
            (df['N'].round(10) == key[0]) &
            (df['m'].round(15) == key[1]) &
            (df['alpha'].round(15) == key[2])
        ]

        if len(matches) == 0:
            return None

        # Return F-matrix from first match
        return columns_to_f_matrix(matches.iloc[0], self.M)

    def has_params(self, case: str, N: float, m: float, alpha: float) -> bool:
        """
        Check if parameters are cached.

        Args:
            case: Case name
            N, m, alpha: Parameter values

        Returns:
            True if cached, False otherwise
        """
        return self.get_f_matrix(case, N, m, alpha) is not None

    def get_missing_params(self, case: str, param_list: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Get list of parameters not in cache.

        Args:
            case: Case name
            param_list: List of (N, m, alpha) tuples

        Returns:
            List of parameters not in cache
        """
        return [
            params for params in param_list
            if not self.has_params(case, *params)
        ]

    def append_results(self, case: str, results: List[Dict]):
        """
        Append new F-matrix results to cache.

        Args:
            case: Case name
            results: List of dicts with keys 'N', 'm', 'alpha', 'F_matrix'

        Examples:
            >>> results = [
            ...     {'N': 100, 'm': 0.01, 'alpha': 0.001, 'F_matrix': np.eye(3)},
            ... ]
            >>> cache.append_results('case1', results)
        """
        # Load existing cache
        df_existing = self.load_case(case)

        # Convert new results to DataFrame
        rows = []
        for result in results:
            row = {
                'N': result['N'],
                'm': result['m'],
                'alpha': result['alpha'],
            }
            row.update(f_matrix_to_columns(result['F_matrix'], self.M))
            rows.append(row)

        df_new = pd.DataFrame(rows)

        # Combine and remove duplicates (keep first occurrence)
        if len(df_existing) > 0:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['N', 'm', 'alpha'], keep='first')
        else:
            df_combined = df_new

        # Save
        self.save_case(case, df_combined)

    def clear_cache(self, case: Optional[str] = None):
        """
        Clear cache.

        Args:
            case: If specified, clear only this case. Otherwise clear all.
        """
        if case is not None:
            filepath = self._get_case_file(case)
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Cleared cache for M={self.M}, {case}")
        else:
            # Clear all cases for this M
            if os.path.exists(self.cache_path):
                for filename in os.listdir(self.cache_path):
                    if filename.endswith('.parquet'):
                        os.remove(os.path.join(self.cache_path, filename))
                print(f"Cleared all cache for M={self.M}")

    def _update_metadata(self, case: str):
        """Update metadata file with cache information."""
        metadata = {}

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

        # Update entry for this M and case
        key = f'M{self.M}_{case}'
        metadata[key] = {
            'last_updated': datetime.now().isoformat(),
            'num_entries': len(self.load_case(case)),
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_stats(self, case: str) -> Dict:
        """
        Get statistics about cached data.

        Args:
            case: Case name

        Returns:
            Dictionary with statistics
        """
        df = self.load_case(case)

        if len(df) == 0:
            return {'num_entries': 0}

        return {
            'num_entries': len(df),
            'N_range': (df['N'].min(), df['N'].max()),
            'm_range': (df['m'].min(), df['m'].max()),
            'alpha_range': (df['alpha'].min(), df['alpha'].max()),
        }
