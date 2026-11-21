"""Capítulo 6: extracción y selección de características.

1. Análisis univariante (tabla 6.1): medianas por clase, AUC orientado y test de Mann-Whitney.
2. Redundancia: pares de variables con |r| > 0,9.
3. Selección de características: cuatro métodos que producen una ORDENACIÓN de las 72 variables,
   calculada solo con el entrenamiento de cada fold (validación cruzada 5x3 a nivel de paciente):
     filtros:   ANOVA F, información mutua (Kraskov, k = 5)
     embebidos: regularización L1 (orden de entrada al relajar la penalización), Random Forest (500 árboles)
     referencia: orden aleatorio
   Con las k primeras variables se entrena una regresión logística (C elegido por CV interna) y se mide el
   AUROC en el test del fold.
Salidas en results/: caracteristicas_univariante.csv, caracteristicas_resumen.json,
seleccion_auroc.csv (tabla 6.2 y figura 6.3), seleccion_folds.csv, seleccion_frecuencia_top10.csv (figura 6.4),
seleccion_tiempo.csv.
"""
import time, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from ..representations import build, feature_names
from ..splits import outer_splits
from ..config import RESULTS, SEED

K_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 72]


def rank_anova(X, y, seed):
    return np.argsort(-np.nan_to_num(f_classif(X, y)[0]))


def rank_mi(X, y, seed):
    return np.argsort(-mutual_info_classif(X, y, n_neighbors=5, random_state=seed))


def rank_l1(X, y, seed):
    """Orden de entrada de cada variable en el camino de regularización L1 (de penalización fuerte a débil)."""
    Z = StandardScaler().fit_transform(X)
    Cs = np.logspace(-3, 1, 40)
    enter = np.full(Z.shape[1], len(Cs), float)
    last = None
    for i, C in enumerate(Cs):
        coef = LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced",
                                  max_iter=5000).fit(Z, y).coef_[0]
        enter[(enter == len(Cs)) & (np.abs(coef) > 1e-8)] = i
        last = np.abs(coef)
    return np.lexsort((-last, enter))   # primero las que entran antes; en empate, mayor |coeficiente| final


def rank_rf(X, y, seed):
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=3,
                                random_state=seed, n_jobs=3).fit(X, y)
    return np.argsort(-rf.feature_importances_)


def rank_random(X, y, seed):
    return np.random.default_rng(seed).permutation(X.shape[1])


METHODS = {
    "ANOVA F": rank_anova,
    "Información mutua": rank_mi,
    "Regularización L1": rank_l1,
    "Random Forest": rank_rf,
    "Orden aleatorio": rank_random,
}


def evaluate(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                               scoring="roc_auc", class_weight="balanced", max_iter=3000).fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.decision_function(sc.transform(Xte)))


def univariate(F, names, y):
    rows = []
    for c in names:
        a, b = F[c][y == 0], F[c][y == 1]
        auc = roc_auc_score(y, F[c])
        rows.append(dict(variable=c, mediana_normal=a.median(), mediana_brugada=b.median(),
                         auc_orientado=max(auc, 1 - auc), p_mannwhitney=stats.mannwhitneyu(a, b).pvalue))
    return pd.DataFrame(rows).sort_values("auc_orientado", ascending=False)


def main():
    ds, MB, F = build("bandpass")
    names = feature_names(F)
    X, y, n = F[names].values.astype(np.float64), ds.y, len(names)

    # ---------- 1-2. análisis univariante y redundancia ----------
    U = univariate(F, names, y)
    U.to_csv(RESULTS / "caracteristicas_univariante.csv", index=False)
    C = np.nan_to_num(np.corrcoef(X, rowvar=False))
    hi = np.abs(C) > 0.9
    qrs = F.qrs_dur_ms
    info = {"n_variables": n,
            "pares_r_mayor_0,9": int(np.triu(hi, 1).sum()),
            "variables_en_algun_par_r_mayor_0,9": int((hi.sum(0) - 1 > 0).sum()),
            "qrs_valores_distintos": int(qrs.nunique()),
            "pct_qrs_entre_90_y_130ms": 100 * float(qrs.between(90, 130).mean()),
            "pct_T_negativa_V2_brugada": 100 * float(F.V2_t_negative[y == 1].mean()),
            "pct_T_negativa_V2_normal": 100 * float(F.V2_t_negative[y == 0].mean())}
    json.dump(info, open(RESULTS / "caracteristicas_resumen.json", "w"), indent=1, ensure_ascii=False)
    print(U.head(10).round(4).to_string(index=False)); print(json.dumps(info, indent=1, ensure_ascii=False))

    # ---------- 3. selección de características ----------
    rows, ranks, times = [], {m: [] for m in METHODS}, {m: [] for m in METHODS}
    for rep, fold, tr, te in outer_splits(y):
        seed = SEED + rep * 100 + fold
        for m, fn in METHODS.items():
            t0 = time.time(); r = fn(X[tr], y[tr], seed); times[m].append(time.time() - t0)
            ranks[m].append(r)
            for k in K_GRID:
                rows.append(dict(method=m, k=k, rep=rep, fold=fold,
                                 auroc=evaluate(X[tr][:, r[:k]], y[tr], X[te][:, r[:k]], y[te], seed)))
        print(f"repetición {rep}, fold {fold}: ok", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "seleccion_folds.csv", index=False)
    A = R.groupby(["method", "k"]).agg(auroc=("auroc", "mean"), auroc_std=("auroc", "std")).reset_index()
    A.to_csv(RESULTS / "seleccion_auroc.csv", index=False)
    freq = pd.DataFrame({m: pd.Series(np.bincount(np.concatenate([r[:10] for r in ranks[m]]), minlength=n) / len(ranks[m]),
                                      index=names) for m in METHODS if m != "Orden aleatorio"})
    freq.to_csv(RESULTS / "seleccion_frecuencia_top10.csv")
    pd.Series({m: np.mean(v) for m, v in times.items()}, name="segundos_por_fold").to_csv(RESULTS / "seleccion_tiempo.csv")
    print(A[A.k.isin([3, 5, 10, 72])].pivot(index="method", columns="k", values="auroc").round(3).to_string())
    print("\nFrecuencia media en el top-10:\n", freq.mean(1).sort_values(ascending=False).head(8).round(2).to_string())


if __name__ == "__main__":
    main()
