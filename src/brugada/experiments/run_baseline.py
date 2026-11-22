"""Capítulo 9: modelo de clasificación de referencia frente a las reglas clínicas (tabla 9).

  * LDA con contracción (Ledoit-Wolf) y regresión logística sobre las 72 características (CV anidada 5x3).
  * Reglas clínicas fijas «2 de 3» y «3 de 3» (umbrales de consenso, sin aprendizaje).
  * Puntuación clínica continua (ST + |T|) con umbral de Youden aprendido en el entrenamiento.
Salidas: results/runs/<modelo>__features{.json, _oof.npz, _folds.csv} y results/modelo_referencia.csv.
"""
import numpy as np, pandas as pd
from ..config import RESULTS
from ..representations import build, feature_matrix
from ..rules import RULES, rule_predict, rule_score
from ..evaluate import youden_threshold
from ..models import make_fit_predict
from .common import run_cv


def main():
    ds, MB, F = build("bandpass")
    res = {}
    for m in ("lda", "logreg"):
        res[m] = run_cv(make_fit_predict(m), feature_matrix(F), ds.y, m, ids=ds.ids)

    # Las reglas trabajan sobre la tabla de características: se pasan los índices de fila como X.
    idx = np.arange(len(ds.y))[:, None]
    rows_of = lambda Xi: F.iloc[Xi[:, 0].astype(int)]
    for rule in RULES:
        def fixed(Xtr, ytr, Xte, seed, rule=rule):
            return rule_predict(rows_of(Xte), rule).astype(float), 0.5, {}
        res[f"regla_{rule}"] = run_cv(fixed, idx, ds.y, f"regla_{rule}", ids=ds.ids)

    def score(Xtr, ytr, Xte, seed):
        return rule_score(rows_of(Xte)), youden_threshold(ytr, rule_score(rows_of(Xtr))), {}
    res["puntuacion_clinica"] = run_cv(score, idx, ds.y, "puntuacion_clinica", ids=ds.ids)

    T = pd.DataFrame([dict(modelo=k, auroc=r["folds"]["auroc"], auroc_dt=r["folds_std"]["auroc"],
                           ic95_bajo=r["ci"]["ci_low"]["auroc"], ic95_alto=r["ci"]["ci_high"]["auroc"],
                           **{m: r["folds"][m] for m in ("sensitivity", "specificity", "precision", "npv", "balanced_acc", "mcc")})
                      for k, r in res.items()])
    T.to_csv(RESULTS / "modelo_referencia.csv", index=False)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
