"""Carga del dataset Brugada-HUCA (PhysioNet) y construcción de etiquetas."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
import wfdb
from .config import DATA_RAW, DATA_PROCESSED, LEADS, N_SAMPLES, POSITIVE_CODES_DEFAULT


@dataclass
class Dataset:
    X: np.ndarray            # (n, 1200, 12) señal cruda en mV
    y: np.ndarray            # (n,) etiqueta binaria
    meta: pd.DataFrame       # metadatos alineados con X
    ids: np.ndarray          # patient_id como str

    def __len__(self):
        return len(self.y)


def read_records() -> list[str]:
    """Lee RECORDS eliminando duplicados (PhysioNet incluye 596382 dos veces)."""
    recs = [l.strip() for l in open(DATA_RAW / "RECORDS") if l.strip()]
    seen, out = set(), []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def load_raw(cache: bool = True) -> tuple[np.ndarray, list[str]]:
    """Devuelve (X, ids) con X de forma (n, 1200, 12). Cachea en .npz."""
    cache_f = DATA_PROCESSED / "raw_signals.npz"
    if cache and cache_f.exists():
        d = np.load(cache_f, allow_pickle=True)
        return d["X"], list(d["ids"])
    recs = read_records()
    sigs, ids = [], []
    for r in recs:
        rec = wfdb.rdrecord(str(DATA_RAW / r))
        assert rec.sig_name == LEADS, rec.sig_name
        assert rec.p_signal.shape == (N_SAMPLES, 12), rec.p_signal.shape
        sigs.append(rec.p_signal.astype(np.float32))
        ids.append(r.split("/")[-1])
    X = np.stack(sigs)
    if cache:
        np.savez_compressed(cache_f, X=X, ids=np.array(ids))
    return X, ids


def load_dataset(positive_codes: Iterable[int] = POSITIVE_CODES_DEFAULT,
                 exclude_codes: Iterable[int] = ()) -> Dataset:
    """Carga señales + metadatos y construye la etiqueta binaria.

    positive_codes: valores de `brugada` considerados positivos.
    exclude_codes: valores de `brugada` que se eliminan del estudio (p.ej. (2,) para
                   excluir los casos atípicos en un análisis de sensibilidad).
    """
    X, ids = load_raw()
    meta = pd.read_csv(DATA_RAW / "metadata.csv")
    meta["patient_id"] = meta["patient_id"].astype(str)
    meta = meta.set_index("patient_id").loc[ids].reset_index()
    keep = ~meta["brugada"].isin(list(exclude_codes))
    meta = meta[keep].reset_index(drop=True)
    X = X[keep.values]
    y = meta["brugada"].isin(list(positive_codes)).astype(int).values
    return Dataset(X=X, y=y, meta=meta, ids=meta["patient_id"].values)


def dataset_fingerprint() -> str:
    """Hash SHA256 de los ficheros de señal para documentar la versión exacta del dataset."""
    h = hashlib.sha256()
    for r in read_records():
        for ext in (".hea", ".dat"):
            h.update((DATA_RAW / (r + ext)).read_bytes())
    return h.hexdigest()[:16]
