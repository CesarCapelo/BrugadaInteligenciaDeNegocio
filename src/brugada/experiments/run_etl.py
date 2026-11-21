"""Capítulo 3: datos, proceso ETL y control de calidad.

Calcula la composición del conjunto (tabla 3.1), las comprobaciones de calidad (tabla 3.2) y el análisis
de sensibilidad de la definición de la etiqueta (sección 3.4). Salidas en results/:
  etl_resumen.json        composición, huella SHA-256 y comprobaciones de calidad
  etl_calidad_latidos.csv latidos válidos, frecuencia cardíaca y correlación con la plantilla por paciente
  runs_etl/               validaciones cruzadas de los análisis de sensibilidad
"""
import json
import numpy as np, pandas as pd
from scipy import stats, signal
from ..config import RESULTS, DATA_RAW, FS, LEAD_IDX
from ..data import load_dataset, read_records, dataset_fingerprint
from ..preprocess import preprocess
from ..beats import mean_beats_dataset
from ..representations import build, feature_matrix
from ..models import make_fit_predict
from .common import run_cv

SAT_MV = 3.2   # |x| >= 3,2 mV se considera posible saturación (rango del conversor ±3,3 mV)
OUT = RESULTS / "runs_etl"


def main():
    ds = load_dataset()
    m = ds.meta
    raw_index = [l.strip() for l in open(DATA_RAW / "RECORDS") if l.strip()]
    out = {"huella_sha256": dataset_fingerprint(), "n_pacientes": len(ds)}

    # ---------------- composición (tabla 3.1) ----------------
    out["composicion"] = {
        "pacientes": m.brugada.value_counts().sort_index().to_dict(),
        "basal_patologico": m.groupby("brugada").basal_pattern.sum().to_dict(),
        "muerte_subita": m.groupby("brugada").sudden_death.sum().to_dict(),
        "positivos_{1,2}": int(ds.y.sum()), "negativos_{0}": int((ds.y == 0).sum())}

    # ---------------- comprobaciones de calidad (tabla 3.2) ----------------
    X = ds.X
    dup = pd.Series(raw_index).value_counts()
    meta_ids = set(pd.read_csv(DATA_RAW / "metadata.csv").patient_id.astype(str))
    sig_ids = {r.split("/")[-1] for r in read_records()}
    q = {"entradas_RECORDS": len(raw_index), "registros_unicos": len(set(raw_index)),
         "duplicados": dup[dup > 1].index.str.split("/").str[-1].tolist(),
         "ids_coinciden_senal_metadatos": sig_ids == meta_ids,
         "valores_ausentes": int(np.isnan(X).sum()) + int(m.isna().sum().sum()),
         "derivaciones_planas": int((X.std(axis=1) < 1e-3).sum())}

    # saturación: en cualquier derivación y, en detalle, en V1-V2
    sat = np.abs(X) >= SAT_MV
    q["saturacion_registros"] = int(sat.any(axis=(1, 2)).sum())
    Xp = preprocess(X, "bandpass")
    MB, infos = mean_beats_dataset(Xp)
    det = []
    for i in np.where(sat[:, :, [LEAD_IDX["V1"], LEAD_IDX["V2"]]].any(axis=(1, 2)))[0]:
        t = np.where(sat[i][:, [LEAD_IDX["V1"], LEAD_IDX["V2"]]].any(axis=1))[0]
        dist = np.abs(t[:, None] - infos[i].rpeaks[None, :]).min(axis=1) * 1000 / FS
        det.append(dict(id=str(ds.ids[i]), muestras=int(len(t)), dist_min_R_ms=float(dist.min()), dist_max_R_ms=float(dist.max())))
    q["saturacion_V1V2"] = det

    # ruido de alta frecuencia y deriva de la línea de base (potencia relativa en V1, Welch)
    f, P = signal.welch(X, fs=FS, axis=1, nperseg=256)
    hf = P[:, f > 40, LEAD_IDX["V1"]].sum(1) / P[:, :, LEAD_IDX["V1"]].sum(1)
    lf = P[:, f < 0.5, LEAD_IDX["V1"]].sum(1) / P[:, :, LEAD_IDX["V1"]].sum(1)
    q["ruido_>40Hz_V1_mediana_%"] = 100 * float(np.median(hf))
    q["ruido_>40Hz_V1_p95_%"] = 100 * float(np.percentile(hf, 95))
    q["deriva_<0,5Hz_V1_mediana_%"] = 100 * float(np.median(lf))

    # calidad de los latidos y frecuencia cardíaca
    B = pd.DataFrame({"id": ds.ids, "y": ds.y, "n_latidos": [i.n_beats for i in infos],
                      "fc_lpm": [i.hr_bpm for i in infos], "corr_plantilla": [i.beat_corr for i in infos],
                      "control_ok": [i.quality_ok for i in infos]})
    B.to_csv(RESULTS / "etl_calidad_latidos.csv", index=False)
    q["latidos_mediana"] = float(B.n_latidos.median())
    q["corr_plantilla_media"] = float(B.corr_plantilla.mean())
    q["registros_que_superan_el_control"] = int(B.control_ok.sum())
    q["fc_mediana_lpm"], q["fc_min_lpm"], q["fc_max_lpm"] = float(B.fc_lpm.median()), float(B.fc_lpm.min()), float(B.fc_lpm.max())
    q["fc_mannwhitney_p"] = float(stats.mannwhitneyu(B.fc_lpm[B.y == 1], B.fc_lpm[B.y == 0]).pvalue)
    out["calidad"] = q

    # ¿influyen los registros saturados en V1-V2? Modelo de referencia (LDA) con y sin ellos
    _, _, F = build("bandpass")
    Xf = feature_matrix(F)
    keep = ~np.isin(ds.ids.astype(str), [d["id"] for d in det])
    full = run_cv(make_fit_predict("lda"), Xf, ds.y, "lda_todos", ids=ds.ids, out_dir=OUT)
    sin = run_cv(make_fit_predict("lda"), Xf[keep], ds.y[keep], "lda_sin_saturados", ids=ds.ids[keep], out_dir=OUT)
    out["saturacion_efecto_auroc"] = {"todos": full["folds"]["auroc"], "sin_saturados_V1V2": sin["folds"]["auroc"]}

    # ---------------- definición de la etiqueta (sección 3.4) ----------------
    # Referencia: atípicos como positivos (regresión logística, el clasificador fijo de las comparaciones)
    lr = make_fit_predict("logreg")
    ref = run_cv(lr, Xf, ds.y, "etiqueta_atipicos_positivos", ids=ds.ids, out_dir=OUT)
    ds_ex, _, F_ex = build("bandpass", exclude_codes=(2,))
    exc = run_cv(lr, feature_matrix(F_ex), ds_ex.y, "etiqueta_sin_atipicos", ids=ds_ex.ids, out_dir=OUT)
    y_neg = m.brugada.isin([1]).astype(int).values
    neg = run_cv(lr, Xf, y_neg, "etiqueta_atipicos_negativos", ids=ds.ids, out_dir=OUT)
    out["etiqueta_auroc"] = {k: {"media": r["folds"]["auroc"], "dt": r["folds_std"]["auroc"]}
                             for k, r in (("atipicos_positivos", ref), ("sin_atipicos", exc), ("atipicos_negativos", neg))}

    with open(RESULTS / "etl_resumen.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
