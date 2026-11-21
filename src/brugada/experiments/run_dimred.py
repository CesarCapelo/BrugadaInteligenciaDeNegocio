"""Capítulo 7: visualización mediante reducción de dimensionalidad (PCA, t-SNE y UMAP).

Se aplica sin etiquetas a dos representaciones estandarizadas de los 363 pacientes: las 72 características
y el latido medio de V1 y V2 muestra a muestra (2 x 70 = 140 valores). Para no sacar conclusiones solo de la
imagen, se mide el AUROC de un k-NN (k = 15, CV 5x3) entrenado solo con las dos coordenadas de cada mapa y,
como referencia, con todas las variables. Salidas: results/dimred_knn.csv y results/dimred_pca.json.
"""
import json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import umap
from ..representations import build, feature_names
from ..features import flatten_beats
from ..config import RESULTS, SEED


def knn_auc(Z, y):
    aucs = [roc_auc_score(y[te], KNeighborsClassifier(15).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1])
            for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(Z, y)]
    return float(np.mean(aucs)), float(np.std(aucs))


def embed(X):
    return {
        "PCA": PCA(n_components=2, random_state=SEED).fit_transform(X),
        "t-SNE": TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=SEED).fit_transform(X),
        "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(X),
    }


def datasets(F, MB):
    return {"72 características": StandardScaler().fit_transform(F[feature_names(F)].values),
            "latido medio V1+V2": StandardScaler().fit_transform(flatten_beats(MB, ("V1", "V2")))}


def main():
    ds, MB, F = build("bandpass")
    y = ds.y
    rows = []
    for dname, X in datasets(F, MB).items():
        for mname, Z in embed(X).items():
            auc, sd = knn_auc(Z, y)
            rows.append(dict(datos=dname, metodo=mname, knn_auroc=auc, knn_auroc_std=sd))
        auc, sd = knn_auc(X, y)
        rows.append(dict(datos=dname, metodo="espacio original", knn_auroc=auc, knn_auroc_std=sd))
    Q = pd.DataFrame(rows)
    Q.to_csv(RESULTS / "dimred_knn.csv", index=False)
    print(Q.round(3).to_string(index=False))

    X = datasets(F, MB)["72 características"]
    pca = PCA(random_state=SEED).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    load = pd.DataFrame(pca.components_[:2].T, index=feature_names(F), columns=["PC1", "PC2"])
    info = dict(varianza_PC1=float(pca.explained_variance_ratio_[0]), varianza_PC2=float(pca.explained_variance_ratio_[1]),
                componentes_para_90pct=int(np.searchsorted(cum, 0.90) + 1),
                cargas_PC1=load.PC1.abs().sort_values(ascending=False).head(8).index.tolist(),
                cargas_PC2=load.PC2.abs().sort_values(ascending=False).head(8).index.tolist())
    json.dump(info, open(RESULTS / "dimred_pca.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(info, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
