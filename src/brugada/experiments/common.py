"""Validación cruzada común a todos los experimentos (sección 4.5 de la memoria).

  * Particiones a nivel de paciente, estratificadas y repetidas: 5 folds x 3 repeticiones, semilla 42.
  * Todo ajuste (escalado, hiperparámetros, umbral) usa solo la partición de entrenamiento del fold.
  * Se guardan las predicciones fuera de muestra (OOF) de cada paciente en cada repetición.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import RESULTS, SEED, N_FOLDS, N_REPEATS
from ..splits import outer_splits
from ..evaluate import metrics_from_pred, summarize_folds, bootstrap_ci


def run_cv(fit_predict, X, y, name: str, representation: str = "features", ids=None,
           out_dir: Path | None = None, verbose: bool = True) -> dict:
    """fit_predict(Xtr, ytr, Xte, seed) -> (puntuación_test, umbral, info).
    El umbral debe elegirse solo con datos de entrenamiento."""
    out_dir = Path(out_dir or RESULTS / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(y)
    oof_score = np.full((N_REPEATS, n), np.nan)
    oof_pred = np.full((N_REPEATS, n), -1)
    fold_metrics = []
    for rep, fold, tr, te in outer_splits(y):
        s, thr, _ = fit_predict(X[tr], y[tr], X[te], SEED + rep * 100 + fold)
        p = (s >= thr).astype(int)
        oof_score[rep, te], oof_pred[rep, te] = s, p
        m = metrics_from_pred(y[te], p, s)
        m.update(rep=rep, fold=fold, threshold=thr)
        fold_metrics.append(m)
    summ = summarize_folds(fold_metrics)
    ci = bootstrap_ci(y, oof_score[0], oof_pred[0], seed=SEED)  # IC sobre la 1.ª repetición (1 predicción/paciente)
    res = dict(name=name, representation=representation, n=int(n), n_pos=int(y.sum()),
               folds=summ["mean"].to_dict(), folds_std=summ["std"].to_dict(), ci=ci.to_dict())
    tag = f"{name}__{representation}"
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(out_dir / f"{tag}_oof.npz", score=oof_score, pred=oof_pred, y=y,
                        ids=np.array(ids) if ids is not None else np.arange(n))
    pd.DataFrame(fold_metrics).to_csv(out_dir / f"{tag}_folds.csv", index=False)
    if verbose:
        f_ = res["folds"]
        print(f"{tag:45s} AUROC={f_['auroc']:.3f}±{res['folds_std']['auroc']:.3f} "
              f"sens={f_['sensitivity']:.2f} esp={f_['specificity']:.2f} BA={f_['balanced_acc']:.3f}", flush=True)
    return res
