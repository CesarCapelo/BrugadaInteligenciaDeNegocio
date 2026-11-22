"""Capítulo 5: comparación de alternativas en cada paso del preprocesado (tabla 5).

Cada alternativa se aplica al proceso completo y se mide con el mismo clasificador fijo (regresión logística
sobre las 72 características, validación cruzada anidada 5x3). Salidas: results/preprocesado_alternativas.csv
y results/runs_preprocesado/.
"""
import pandas as pd
from ..config import RESULTS
from ..data import load_dataset
from ..preprocess import preprocess, PREPROCESS_VARIANTS
from ..beats import mean_beats_dataset
from ..features import features_dataset
from ..representations import build, feature_matrix
from ..models import make_fit_predict
from .common import run_cv

OUT = RESULTS / "runs_preprocesado"


def main():
    ds = load_dataset()
    lr = make_fit_predict("logreg")
    rows = []

    def evaluate(step, alternative, F):
        r = run_cv(lr, feature_matrix(F), ds.y, f"{step}", alternative, ids=ds.ids, out_dir=OUT)
        rows.append(dict(paso=step, alternativa=alternative, auroc=r["folds"]["auroc"], dt=r["folds_std"]["auroc"]))

    for v in PREPROCESS_VARIANTS:                                            # filtrado
        evaluate("filtrado", v, build(v)[2])
    for m in ("neurokit", "pantompkins1985", "hamilton2002", "elgendi2010"):  # detector de picos R
        evaluate("detector_R", m, build("bandpass", rpeak_method=m)[2])
    Xp = preprocess(ds.X, "bandpass")
    for lead in ("II", "V5", "V2"):                                          # derivación de referencia
        MB, infos = mean_beats_dataset(Xp, ref_lead=lead)
        evaluate("derivacion_referencia", lead, features_dataset(MB, infos))
    for agg in ("median", "mean"):                                           # agregación de latidos
        MB, infos = mean_beats_dataset(Xp, aggregate=agg)
        evaluate("agregacion", agg, features_dataset(MB, infos))

    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "preprocesado_alternativas.csv", index=False)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
