"""Particiones a nivel de paciente: validación cruzada estratificada repetida."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from .config import SEED, N_FOLDS, N_REPEATS


def outer_splits(y: np.ndarray, n_folds: int = N_FOLDS, n_repeats: int = N_REPEATS, seed: int = SEED):
    """Genera (repeat, fold, train_idx, test_idx). Cada paciente aparece una única vez en test por repetición."""
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
        yield k // n_folds, k % n_folds, tr, te


def inner_cv(seed: int = SEED, n_folds: int = 5):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
