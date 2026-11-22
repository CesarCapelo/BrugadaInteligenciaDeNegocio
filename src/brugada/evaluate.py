"""Métricas para clasificación desbalanceada, umbral de Youden e intervalos de confianza bootstrap."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve

METRICS = ["sensitivity", "specificity", "precision", "npv", "f1", "balanced_acc", "auroc", "auprc", "mcc"]


def metrics_from_pred(y, yhat, score=None) -> dict:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    m = dict(sensitivity=sens, specificity=spec, precision=prec, npv=npv, f1=f1,
             balanced_acc=(sens + spec) / 2, mcc=mcc, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    if score is not None and len(np.unique(y)) == 2:
        m["auroc"] = roc_auc_score(y, score)
        m["auprc"] = average_precision_score(y, score)
    else:
        m["auroc"] = m["auprc"] = np.nan
    return m


def youden_threshold(y, score) -> float:
    """Umbral que maximiza sensibilidad + especificidad - 1. Solo se usa con datos de entrenamiento."""
    fpr, tpr, thr = roc_curve(np.asarray(y), np.asarray(score))
    return float(thr[int(np.argmax(tpr - fpr))])


def bootstrap_ci(y, score, yhat, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """IC bootstrap (percentil) de las métricas sobre las predicciones fuera de muestra (OOF)."""
    rng = np.random.default_rng(seed)
    y, score, yhat = map(np.asarray, (y, score, yhat))
    rows, n = [], len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        rows.append(metrics_from_pred(y[idx], yhat[idx], score[idx]))
    B = pd.DataFrame(rows)[METRICS]
    point = pd.Series(metrics_from_pred(y, yhat, score))[METRICS]
    return pd.DataFrame({"mean": point, "ci_low": B.quantile(alpha / 2), "ci_high": B.quantile(1 - alpha / 2)})


def summarize_folds(fold_metrics: list[dict]) -> pd.DataFrame:
    """Media y desviación típica entre particiones."""
    df = pd.DataFrame(fold_metrics)
    cols = [c for c in METRICS if c in df]
    return pd.DataFrame({"mean": df[cols].mean(), "std": df[cols].std()})
