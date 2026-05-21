"""FRC linear-algebra analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class OPRCalculator:
    @staticmethod
    def calculate(
        design_matrix: sp.spmatrix, scores: np.ndarray, ridge_lambda: float = 0.0
    ) -> np.ndarray:
        design = design_matrix.tocsc()
        scores = np.asarray(scores, dtype=float)
        n_cols = design.shape[1]
        lhs = design.T @ design
        if ridge_lambda:
            lhs = lhs + ridge_lambda * sp.eye(n_cols, format="csc")
        rhs = design.T @ scores
        return np.asarray(spla.spsolve(lhs.tocsc(), rhs))


class ScheduleTopologyAnalyzer:
    @staticmethod
    def analyze_design(design_matrix: sp.spmatrix) -> pd.DataFrame:
        design = design_matrix.tocsr()
        appearances = np.asarray(design.sum(axis=0)).ravel()
        return pd.DataFrame({"team_index": np.arange(len(appearances)), "appearances": appearances})


class TournamentPoints:
    @staticmethod
    def calculate(wins: int = 0, losses: int = 0, ties: int = 0, ranking_points: int = 0) -> int:
        return int(2 * wins + ties + ranking_points + 0 * losses)
