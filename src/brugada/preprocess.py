"""Filtrado de la señal (capítulo 4 de la memoria).

Todas las funciones operan sobre arrays (n, T, L) o (T, L), son deterministas y no ajustan ningún
parámetro con los datos, por lo que pueden aplicarse antes de particionar sin fuga de información.
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from .config import FS


def bandpass(X: np.ndarray, low: float = 0.5, high: float = 40.0, order: int = 4, fs: int = FS) -> np.ndarray:
    """Butterworth paso banda de orden 4 aplicado en ambas direcciones (fase cero)."""
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, X, axis=-2).astype(np.float32)


def lowpass(X: np.ndarray, cutoff: float = 40.0, order: int = 4, fs: int = FS) -> np.ndarray:
    sos = signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, X, axis=-2).astype(np.float32)


def remove_baseline_median(X: np.ndarray, win1_ms: int = 200, win2_ms: int = 600, fs: int = FS) -> np.ndarray:
    """Eliminación de la línea de base con doble filtro de mediana (200 y 600 ms; de Chazal et al., 2004)."""
    w1 = int(win1_ms * fs / 1000) | 1
    w2 = int(win2_ms * fs / 1000) | 1
    base = signal.medfilt(X, kernel_size=_kernel(X.ndim, w1))
    base = signal.medfilt(base, kernel_size=_kernel(X.ndim, w2))
    return (X - base).astype(np.float32)


def _kernel(ndim: int, w: int):
    k = [1] * ndim
    k[-2] = w
    return tuple(k)


# Alternativas comparadas en la tabla 4.1 de la memoria ("bandpass" es la elegida).
PREPROCESS_VARIANTS = {
    "bandpass": lambda X: bandpass(X),                                # paso banda 0,5-40 Hz
    "median_lowpass": lambda X: lowpass(remove_baseline_median(X)),   # doble mediana + paso bajo 40 Hz
    "bandpass_narrow": lambda X: bandpass(X, 0.67, 30.0),             # paso banda 0,67-30 Hz
    "raw": lambda X: X,                                               # sin filtrar
}


def preprocess(X: np.ndarray, variant: str = "bandpass") -> np.ndarray:
    return PREPROCESS_VARIANTS[variant](X)
