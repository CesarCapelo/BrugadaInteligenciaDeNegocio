"""Capítulo 6: selección de instancias (rechazo de latidos) y latido medio.

Criterios comparados (deterministas y sin usar la etiqueta, por lo que no hay fuga de información):
  none               sin rechazo
  corr0.5 ... corr0.9  correlación con la plantilla mediana del paciente < umbral (0,6 por defecto)
  rr0.2              intervalo RR previo desviado más de un 20 % del RR mediano
  corr0.6+rr0.2      combinación de ambos
Clasificador fijo: regresión logística sobre las 72 características. Salidas en results/:
  instancias_resumen.csv       tabla 6 (latidos descartados, cambio del latido medio, AUROC)
  instancias_por_paciente.csv  embudo de latidos y correlación del peor latido (criterio por defecto)
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from ..config import RESULTS, LEAD_IDX
from ..data import load_dataset
from ..preprocess import preprocess
from ..beats import mean_beats_dataset
from ..features import features_dataset
from ..representations import feature_matrix
from ..models import make_fit_predict
from .common import run_cv

CRITERIA = {
    "none": dict(reject_outliers=False),
    "corr0.5": dict(min_corr=0.5),
    "corr0.6": dict(min_corr=0.6),           # criterio por defecto
    "corr0.7": dict(min_corr=0.7),
    "corr0.8": dict(min_corr=0.8),
    "corr0.9": dict(min_corr=0.9),
    "rr0.2": dict(reject_outliers=False, reject_rr=0.2),
    "corr0.6+rr0.2": dict(min_corr=0.6, reject_rr=0.2),
}


def per_patient(infos, ds):
    return pd.DataFrame({
        "id": ds.ids, "y": ds.y,
        "n_rpeaks": [i.n_rpeaks for i in infos], "n_segmented": [i.n_segmented for i in infos],
        "n_kept": [i.n_beats for i in infos], "fallback": [i.rejection_fallback for i in infos],
        "corr_min_all": [float(np.min(i.corr_all)) if i.corr_all is not None and len(i.corr_all) else np.nan for i in infos]})


def main():
    ds = load_dataset()
    Xp = preprocess(ds.X, "bandpass")
    v12 = [LEAD_IDX["V1"], LEAD_IDX["V2"]]
    rows, ref_mb = [], None
    for name, kw in CRITERIA.items():
        MB, infos = mean_beats_dataset(Xp, **kw)
        P = per_patient(infos, ds)
        if name == "none":
            ref_mb = MB
        if name == "corr0.6":
            P.to_csv(RESULTS / "instancias_por_paciente.csv", index=False)
        res = run_cv(make_fit_predict("logreg"), feature_matrix(features_dataset(MB, infos)), ds.y,
                     "instancias", name, ids=ds.ids, out_dir=RESULTS / "runs_instancias")
        dmb = np.abs(MB[:, :, v12] - ref_mb[:, :, v12]).max(axis=(1, 2))   # cambio del latido medio (mV)
        rej = P.n_segmented - P.n_kept
        rows.append(dict(criterio=name, latidos_segmentados=int(P.n_segmented.sum()), latidos_descartados=int(rej.sum()),
                         pct_descartados=100 * rej.sum() / P.n_segmented.sum(), pacientes_con_descartes=int((rej > 0).sum()),
                         pacientes_fallback=int(P.fallback.sum()),
                         pct_desc_normal=100 * rej[P.y == 0].sum() / P.n_segmented[P.y == 0].sum(),
                         pct_desc_brugada=100 * rej[P.y == 1].sum() / P.n_segmented[P.y == 1].sum(),
                         cambio_latido_mV_p95=float(np.percentile(dmb, 95)),
                         auroc=res["folds"]["auroc"], auroc_std=res["folds_std"]["auroc"]))
    S = pd.DataFrame(rows)
    S.to_csv(RESULTS / "instancias_resumen.csv", index=False)
    print(S.round(3).to_string(index=False))

    P = pd.read_csv(RESULTS / "instancias_por_paciente.csv")
    info = {"picos_R": int(P.n_rpeaks.sum()), "latidos_completos": int(P.n_segmented.sum()), "conservados": int(P.n_kept.sum()),
            "latidos_por_registro_min_mediana_max": [int(P.n_segmented.min()), float(P.n_segmented.median()), int(P.n_segmented.max())],
            "corr_peor_latido_mediana_normal": float(P.corr_min_all[P.y == 0].median()),
            "corr_peor_latido_mediana_brugada": float(P.corr_min_all[P.y == 1].median()),
            "corr_peor_latido_mannwhitney_p": float(stats.mannwhitneyu(P.corr_min_all[P.y == 1], P.corr_min_all[P.y == 0]).pvalue),
            "pct_pacientes_sin_latidos_bajo_0,9": 100 * float((P.corr_min_all >= 0.9).mean())}
    json.dump(info, open(RESULTS / "instancias_embudo.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(info, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
