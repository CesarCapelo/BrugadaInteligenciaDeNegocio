"""Modelos lineales del trabajo con validación cruzada anidada (capítulo 8 de la memoria).

  * lda:    análisis discriminante lineal con contracción de Ledoit-Wolf (sin hiperparámetros).
  * logreg: regresión logística L2 con pesos de clase balanceados; C elegido por CV interna.
            Es también el clasificador fijo con el que se comparan las alternativas de los capítulos 3-5.
"""
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_predict
from .splits import inner_cv
from .evaluate import youden_threshold


def model_zoo() -> dict:
    """Modelo -> (pipeline, rejilla de hiperparámetros)."""
    return {
        "logreg": (Pipeline([("sc", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
                   {"clf__C": [0.01, 0.1, 1, 10]}),
        "lda": (Pipeline([("sc", StandardScaler()),
                          ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
                {}),
    }


def make_fit_predict(model_name: str):
    """Función fit_predict para experiments.common.run_cv. Todo se ajusta solo con el entrenamiento:
    escalado, hiperparámetros (GridSearchCV interno, AUROC) y umbral (Youden sobre predicciones OOF internas)."""
    def fit_predict(Xtr, ytr, Xte, seed):
        pipe, grid = model_zoo()[model_name]
        cv = inner_cv(seed)
        if grid:
            gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True).fit(Xtr, ytr)
            best, params = gs.best_estimator_, gs.best_params_
        else:
            best, params = pipe.fit(Xtr, ytr), {}
        s_inner = cross_val_predict(best, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr = youden_threshold(ytr, s_inner)
        return best.predict_proba(Xte)[:, 1], thr, dict(best_params=params)
    return fit_predict
