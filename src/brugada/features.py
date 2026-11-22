"""Extracción de características electrocardiográficas del latido medio.

Todas las medidas se calculan sobre el latido medio (70 muestras = 700 ms, R en la
muestra 25) ya centrado en el nivel PR. Las unidades son ms y mV.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from .config import FS, BEAT_PRE, LEAD_IDX
from .beats import BeatInfo

MS = 1000 / FS  # ms por muestra (10 ms)


def qrs_bounds(beat: np.ndarray, leads=("V1", "V2", "II", "V5"), frac: float = 0.08,
               hold: int = 3) -> tuple[int, int]:
    """Estima el inicio y el final del QRS a partir de la energía de la derivada en varias
    derivaciones. Devuelve (inicio, final) como índices del latido."""
    idx = [LEAD_IDX[l] for l in leads]
    d = np.abs(np.gradient(beat[:, idx], axis=0)).sum(axis=1)
    d = uniform_filter1d(d, 2)
    thr = frac * d[BEAT_PRE - 6:BEAT_PRE + 8].max()
    on = BEAT_PRE
    while on > 3 and not np.all(d[on - hold:on] < thr):
        on -= 1
    off = BEAT_PRE
    while off < len(beat) - hold - 1 and not np.all(d[off:off + hold] < thr):
        off += 1
    on = int(np.clip(on, BEAT_PRE - 12, BEAT_PRE - 2))     # inicio entre 20 y 120 ms antes de R
    off = int(np.clip(off, BEAT_PRE + 2, BEAT_PRE + 18))   # final entre 20 y 180 ms después de R
    return on, off


def lead_features(b: np.ndarray, on: int, off: int, name: str) -> dict:
    """Características morfológicas de una derivación (vector 1D del latido medio)."""
    f = {}
    j = off
    ms = lambda k: min(len(b) - 1, j + int(round(k / MS)))
    f[f"{name}_st_j"] = float(b[j])
    f[f"{name}_st_j40"] = float(b[ms(40)])
    f[f"{name}_st_j80"] = float(b[ms(80)])
    f[f"{name}_st_slope"] = float((b[ms(80)] - b[j]) / 80.0)            # mV/ms
    f[f"{name}_st_area"] = float(b[j:ms(80) + 1].sum() * MS)            # mV·ms
    f[f"{name}_st_max"] = float(b[j:ms(120) + 1].max())
    # onda T: extremo de mayor |amplitud| entre J+100 ms y el final del latido
    t0, t1 = ms(100), len(b)
    seg = b[t0:t1]
    k = int(np.argmax(np.abs(seg)))
    f[f"{name}_t_amp"] = float(seg[k])
    f[f"{name}_t_pos"] = float((t0 + k - BEAT_PRE) * MS)               # ms desde R
    f[f"{name}_t_neg"] = float(seg.min())
    f[f"{name}_t_negative"] = float(seg.min() < -0.1 and abs(seg.min()) > 0.5 * seg.max())
    # QRS: amplitudes R, S, r'
    q = b[on:off + 1]
    f[f"{name}_r_amp"] = float(q.max())
    f[f"{name}_s_amp"] = float(q.min())
    f[f"{name}_qrs_area"] = float(np.abs(q).sum() * MS)
    # r' (segunda deflexión positiva tras la S) y patrón rSr'
    smin = int(np.argmin(q))
    tail = q[smin:]
    f[f"{name}_rprime_amp"] = float(tail.max()) if len(tail) > 1 else 0.0
    # pendiente final del QRS: un descenso lento desde el pico hasta J es la forma coved
    peak_idx = on + int(np.argmax(np.abs(q)))
    f[f"{name}_qrs_end_slope"] = float((b[off] - b[peak_idx]) / max(1, (off - peak_idx)) / MS)
    # cociente ST/QRS: elevación relativa al voltaje del paciente
    f[f"{name}_st_r_ratio"] = float(b[ms(40)] / (abs(q).max() + 1e-6))
    return f


def beat_features(beat: np.ndarray, info: BeatInfo | None = None) -> dict:
    on, off = qrs_bounds(beat)
    f = {
        "qrs_onset_ms": (on - BEAT_PRE) * MS,
        "qrs_offset_ms": (off - BEAT_PRE) * MS,
        "qrs_dur_ms": (off - on) * MS,
    }
    for lead in ("V1", "V2", "V3", "II"):
        f.update(lead_features(beat[:, LEAD_IDX[lead]], on, off, lead))
    # medidas combinadas V1/V2
    f["v12_st_j40_max"] = max(f["V1_st_j40"], f["V2_st_j40"])
    f["v12_t_neg_min"] = min(f["V1_t_neg"], f["V2_t_neg"])
    f["v12_st_r_ratio_max"] = max(f["V1_st_r_ratio"], f["V2_st_r_ratio"])
    if info is not None:
        f["hr_bpm"] = info.hr_bpm
        f["rr_cv"] = info.rr_cv
        f["n_beats"] = info.n_beats
        f["beat_corr"] = info.beat_corr
    return f


def features_dataset(MB: np.ndarray, infos: list[BeatInfo] | None = None) -> pd.DataFrame:
    rows = [beat_features(MB[i], infos[i] if infos else None) for i in range(len(MB))]
    return pd.DataFrame(rows)


def flatten_beats(MB: np.ndarray, leads=("V1", "V2")) -> np.ndarray:
    """Representación 'muestra a muestra' del latido medio: (n, len(leads)*70)."""
    idx = [LEAD_IDX[l] for l in leads]
    return MB[:, :, idx].transpose(0, 2, 1).reshape(len(MB), -1)
