"""Reglas clínicas explícitas (referencia sin aprendizaje, capítulo 8 de la memoria).

Criterios del patrón de Brugada con sus umbrales de consenso en V1 o V2:
  - QRS ancho:        duración del QRS >= 120 ms.
  - Elevación del ST: ST en J o en J+40 ms >= 0,2 mV (2 mm).
  - Onda T negativa:  mínimo de la onda T <= -0,1 mV.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_THR = dict(qrs_ms=120.0, st_mv=0.2, t_mv=0.1)


def pattern_flags(F: pd.DataFrame, qrs_ms=120.0, st_mv=0.2, t_mv=0.1, leads=("V1", "V2")) -> pd.DataFrame:
    st = np.max([np.maximum(F[f"{l}_st_j"], F[f"{l}_st_j40"]) for l in leads], axis=0)
    tneg = np.min([F[f"{l}_t_neg"] for l in leads], axis=0)
    return pd.DataFrame({
        "wide_qrs": (F["qrs_dur_ms"] >= qrs_ms).astype(int).values,
        "st_elev": (st >= st_mv).astype(int),
        "t_neg": (tneg <= -t_mv).astype(int),
        "st_level": st,
        "t_level": tneg,
    })


RULES = {
    "2_de_3": lambda p: ((p.wide_qrs + p.st_elev + p.t_neg) >= 2).astype(int),
    "3_de_3": lambda p: (p.wide_qrs & p.st_elev & p.t_neg),   # aproxima el patrón tipo 1
}


def rule_predict(F: pd.DataFrame, rule: str) -> np.ndarray:
    return RULES[rule](pattern_flags(F, **DEFAULT_THR)).astype(int).values


def rule_score(F: pd.DataFrame, leads=("V1", "V2")) -> np.ndarray:
    """Puntuación continua: máxima elevación del ST + profundidad de la onda T negativa en V1/V2 (mV)."""
    p = pattern_flags(F, leads=leads)
    return (p.st_level + np.clip(-p.t_level, 0, None)).values
