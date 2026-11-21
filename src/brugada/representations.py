"""Construcción (y caché) del latido medio y de la tabla de 72 características de cada paciente."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import DATA_PROCESSED
from .data import load_dataset, Dataset
from .preprocess import preprocess
from .beats import mean_beats_dataset
from .features import features_dataset

QUALITY_COLS = ["n_beats", "beat_corr"]  # métricas de calidad del latido, no son características clínicas


def build(variant: str = "bandpass", rpeak_method: str = "neurokit", force: bool = False,
          **ds_kw) -> tuple[Dataset, np.ndarray, pd.DataFrame]:
    """Devuelve (dataset, latidos medios (n, 70, 12), tabla de características). Cachea por variante."""
    ds = load_dataset(**ds_kw)
    tag = f"{variant}_{rpeak_method}"
    mb_f = DATA_PROCESSED / f"mean_beats_{tag}.npy"
    ft_f = DATA_PROCESSED / f"features_{tag}.csv"
    if mb_f.exists() and ft_f.exists() and not force and not ds_kw:
        return ds, np.load(mb_f), pd.read_csv(ft_f)
    MB, infos = mean_beats_dataset(preprocess(ds.X, variant), method=rpeak_method)
    F = features_dataset(MB, infos)
    if not ds_kw:
        np.save(mb_f, MB)
        F.to_csv(ft_f, index=False)
    return ds, MB, F


def feature_names(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in QUALITY_COLS]


def feature_matrix(F: pd.DataFrame) -> np.ndarray:
    """Matriz (n, 72) de características clínicas."""
    return F[feature_names(F)].values.astype(np.float32)
