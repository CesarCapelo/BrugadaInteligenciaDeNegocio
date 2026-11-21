"""Detección de latidos, segmentación, alineamiento y latido medio.

Los picos R se detectan en una derivación de referencia (por defecto II, la más
habitual clínicamente) y los mismos fiduciales se usan para segmentar V1 y V2, de
modo que el latido medio de todas las derivaciones queda alineado temporalmente.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import neurokit2 as nk
from .config import FS, BEAT_PRE, BEAT_POST, BEAT_LEN, LEAD_IDX


@dataclass
class BeatInfo:
    rpeaks: np.ndarray          # índices de picos R (muestras)
    beats: np.ndarray           # (n_beats, BEAT_LEN, L) latidos individuales
    mean_beat: np.ndarray       # (BEAT_LEN, L) latido medio (mediana o media)
    n_beats: int
    hr_bpm: float
    rr_cv: float                # coef. de variación de RR (irregularidad)
    beat_corr: float            # correlación media de cada latido con el latido medio (calidad)
    quality_ok: bool
    # --- trazabilidad de la selección de instancias (latidos) ---
    n_rpeaks: int = 0           # picos R detectados
    n_segmented: int = 0        # latidos completos dentro de la ventana (se pierden los de los bordes)
    n_rejected_corr: int = 0    # descartados por baja correlación con la plantilla
    n_rejected_rr: int = 0      # descartados por RR anómalo (latidos prematuros / pausas)
    rejection_fallback: bool = False  # True si el criterio dejaba < 3 latidos y se conservaron todos
    corr_all: np.ndarray | None = None  # correlación de cada latido segmentado con la plantilla


def detect_rpeaks(sig: np.ndarray, method: str = "neurokit", fs: int = FS) -> np.ndarray:
    """Detecta picos R en una señal 1D. Métodos: neurokit, pantompkins1985, hamilton2002, elgendi2010."""
    try:
        cleaned = nk.ecg_clean(sig.astype(float), sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=method, correct_artifacts=True)
        r = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        r = np.array([], dtype=int)
    return r


def refine_rpeaks(sig: np.ndarray, rpeaks: np.ndarray, radius: int = 5) -> np.ndarray:
    """Reajusta cada pico al máximo absoluto local (±radius muestras)."""
    out = []
    for r in rpeaks:
        lo, hi = max(0, r - radius), min(len(sig), r + radius + 1)
        out.append(lo + int(np.argmax(np.abs(sig[lo:hi]))))
    return np.array(out, dtype=int)


def segment_beats(X: np.ndarray, rpeaks: np.ndarray, pre: int = BEAT_PRE, post: int = BEAT_POST) -> np.ndarray:
    """X: (T, L). Devuelve (n_beats, pre+post, L) descartando latidos incompletos."""
    T = X.shape[0]
    beats = [X[r - pre:r + post] for r in rpeaks if r - pre >= 0 and r + post <= T]
    if not beats:
        return np.zeros((0, pre + post, X.shape[1]), dtype=X.dtype)
    return np.stack(beats)


def align_beats(beats: np.ndarray, ref_lead: int, max_shift: int = 3) -> np.ndarray:
    """Alinea cada latido con la plantilla mediana por correlación cruzada (±max_shift)."""
    if len(beats) < 2:
        return beats
    template = np.median(beats[:, :, ref_lead], axis=0)
    out = np.empty_like(beats)
    for i, b in enumerate(beats):
        best, best_s = -np.inf, 0
        for s in range(-max_shift, max_shift + 1):
            shifted = np.roll(b[:, ref_lead], s)
            c = np.corrcoef(shifted, template)[0, 1]
            if c > best:
                best, best_s = c, s
        out[i] = np.roll(b, best_s, axis=0)
    return out


def extract_beats(X: np.ndarray, ref_lead: str = "II", method: str = "neurokit",
                  aggregate: str = "median", reject_outliers: bool = True,
                  min_corr: float = 0.6, reject_rr: float | None = None) -> BeatInfo:
    """Pipeline completo para un registro (T, L): detección -> segmentación -> alineamiento -> latido medio.

    Selección de instancias (latidos) antes de agregar:
      * reject_outliers/min_corr: descarta latidos cuya correlación con la plantilla mediana es < min_corr
        (ectópicos, artefactos, ruido).
      * reject_rr: si se indica (p. ej. 0.2), descarta latidos cuyo RR previo se desvía más de ese
        porcentaje del RR mediano (latidos prematuros o tras una pausa).
    Si los criterios dejan menos de 3 latidos se conservan todos (rejection_fallback=True).
    Con los valores por defecto el comportamiento es idéntico al de la versión original."""
    ref = LEAD_IDX[ref_lead]
    rpeaks = detect_rpeaks(X[:, ref], method=method)
    if len(rpeaks) < 3:  # fallback: probar con V2 / V5 que suelen tener QRS de gran amplitud
        for alt in ("V5", "V2", "I"):
            rpeaks = detect_rpeaks(X[:, LEAD_IDX[alt]], method=method)
            if len(rpeaks) >= 3:
                ref = LEAD_IDX[alt]
                break
    rpeaks = refine_rpeaks(X[:, ref], rpeaks)
    beats = segment_beats(X, rpeaks)
    seg_idx = np.where((rpeaks - BEAT_PRE >= 0) & (rpeaks + BEAT_POST <= X.shape[0]))[0]
    rr = np.diff(rpeaks) / FS if len(rpeaks) > 1 else np.array([np.nan])
    hr = 60.0 / np.nanmean(rr) if len(rr) and np.isfinite(rr).any() else np.nan
    rr_cv = float(np.nanstd(rr) / np.nanmean(rr)) if len(rr) > 1 else np.nan
    if len(beats) == 0:
        return BeatInfo(rpeaks, beats, np.zeros((BEAT_LEN, X.shape[1]), np.float32), 0, hr, rr_cv, 0.0, False,
                        n_rpeaks=len(rpeaks), n_segmented=0)
    beats = align_beats(beats, ref)
    template = np.median(beats, axis=0)
    corr = np.array([np.corrcoef(b[:, ref], template[:, ref])[0, 1] for b in beats])
    corr = np.nan_to_num(corr)
    corr_all = corr.copy()
    n_seg = len(beats)
    keep_corr = corr >= min_corr if reject_outliers else np.ones(n_seg, bool)
    keep_rr = np.ones(n_seg, bool)
    if reject_rr is not None and len(rpeaks) > 2:
        rr_s = np.diff(rpeaks)
        med = np.median(rr_s)
        prev_rr = np.array([rr_s[i - 1] if i > 0 else rr_s[0] for i in seg_idx])
        keep_rr = np.abs(prev_rr - med) / med <= reject_rr
    keep = keep_corr & keep_rr
    fallback = keep.sum() < 3
    if not fallback:
        beats, corr = beats[keep], corr[keep]
    mean_beat = np.median(beats, axis=0) if aggregate == "median" else beats.mean(axis=0)
    # centrar el latido medio restando el nivel del segmento PR (isoelectrico) por derivacion
    iso = mean_beat[BEAT_PRE - 8:BEAT_PRE - 3].mean(axis=0, keepdims=True)
    mean_beat = (mean_beat - iso).astype(np.float32)
    beats = (beats - iso).astype(np.float32)
    ok = len(beats) >= 5 and float(corr.mean()) >= min_corr
    return BeatInfo(rpeaks, beats, mean_beat, len(beats), float(hr), rr_cv, float(corr.mean()), bool(ok),
                    n_rpeaks=len(rpeaks), n_segmented=n_seg,
                    n_rejected_corr=0 if fallback else int((~keep_corr).sum()),
                    n_rejected_rr=0 if fallback else int((keep_corr & ~keep_rr).sum()),
                    rejection_fallback=bool(fallback and (~keep).any()), corr_all=corr_all)


def mean_beats_dataset(X: np.ndarray, **kw) -> tuple[np.ndarray, list[BeatInfo]]:
    """Aplica extract_beats a (n, T, L). Devuelve (n, BEAT_LEN, L) y la lista de BeatInfo."""
    infos = [extract_beats(x, **kw) for x in X]
    return np.stack([b.mean_beat for b in infos]), infos
