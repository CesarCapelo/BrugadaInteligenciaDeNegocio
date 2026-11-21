"""Genera las figuras vectoriales (PDF) de la memoria en memoria/figs/ a partir de los datos y de results/.

Requiere haber ejecutado antes los experimentos (ver README). Desde la raíz del repositorio:

    python memoria/make_figures.py
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal, stats
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from brugada.representations import build, feature_names
from brugada.preprocess import preprocess
from brugada.beats import extract_beats, segment_beats, detect_rpeaks, refine_rpeaks
from brugada.config import LEAD_IDX, BEAT_PRE, FS, RESULTS

OUT = Path(__file__).resolve().parent / "figs"; OUT.mkdir(exist_ok=True)
NUM = {}
C_NORM, C_BRU, C_ACC, C_GREY = "#4E79A7", "#C0392B", "#1F4E79", "#8C8C8C"
FAM = {"filtro": "#2E6DA4", "embebido": "#3A8E5C", "ref": "#8C8C8C"}
W = 6.1  # ancho de texto en pulgadas
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.formatter.use_locale": False,
    "lines.linewidth": 1.0, "legend.frameon": False})


def comma(ax, axis="both", fmt="{:.1f}"):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, p: fmt.format(v).replace(".", ","))
    if axis in ("x", "both"): ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"): ax.yaxis.set_major_formatter(f)


def panel(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("ok", name)


ds, MB, F = build("bandpass")
y, ids, meta = ds.y, np.array([str(i) for i in ds.ids]), ds.meta
Xraw = ds.X
Xp = preprocess(Xraw, "bandpass")
names = feature_names(F)
t_beat = (np.arange(70) - BEAT_PRE) * 1000 / FS

# ------------------------------------------------------------------ F1 señal cruda vs filtrada
f, P = signal.welch(Xraw, fs=FS, axis=1, nperseg=256)
lf = P[:, f < 0.5, :].sum(1) / P.sum(1)
cand = np.where((y == 1) & (meta.basal_pattern.values == 1))[0]
k1 = cand[np.argmax(lf[cand, LEAD_IDX["V2"]])]
NUM["ejemplo_senal_id"] = ids[k1]
r = refine_rpeaks(Xp[k1, :, LEAD_IDX["II"]], detect_rpeaks(Xp[k1, :, LEAD_IDX["II"]]))
tt = np.arange(Xraw.shape[1]) / FS
fig, axes = plt.subplots(3, 1, figsize=(W, 3.9), sharex=True)
axes[0].plot(tt, Xraw[k1, :, LEAD_IDX["V2"]], c=C_GREY, lw=0.8); axes[0].set_title("Derivación V2 cruda", loc="left", fontweight="normal")
axes[1].plot(tt, Xp[k1, :, LEAD_IDX["V2"]], c=C_BRU, lw=0.8); axes[1].set_title("V2 filtrada (Butterworth paso banda 0,5–40 Hz, fase cero)", loc="left", fontweight="normal")
axes[2].plot(tt, Xp[k1, :, LEAD_IDX["II"]], c=C_ACC, lw=0.8)
axes[2].plot(r / FS, Xp[k1, r, LEAD_IDX["II"]], "v", c="k", ms=3.5)
axes[2].set_title(f"Derivación II filtrada con los {len(r)} picos R detectados (NeuroKit2)", loc="left", fontweight="normal")
for a, l in zip(axes, "abc"):
    a.set_ylabel("mV"); panel(a, l); comma(a, "y")
axes[2].set_xlabel("tiempo (s)"); axes[2].set_xlim(0, 12)
fig.tight_layout(h_pad=0.6); save(fig, "senal_preprocesado")

# ------------------------------------------------------------------ F2 calidad y composición
fig, axes = plt.subplots(1, 3, figsize=(W, 2.1), gridspec_kw={"width_ratios": [1, 1.25, 1]})
ct = pd.crosstab(meta.brugada, meta.basal_pattern)
labs = ["Normal\n(0)", "Brugada\n(1)", "Atípico\n(2)"]
axes[0].bar(labs, ct[0].values, color=[C_NORM, C_BRU, "#8E44AD"], alpha=0.55, label="ECG basal normal")
axes[0].bar(labs, ct[1].values, bottom=ct[0].values, color=[C_NORM, C_BRU, "#8E44AD"], hatch="////", edgecolor="white", lw=0, label="ECG basal patológico")
for i, (a_, b_) in enumerate(zip(ct[0].values, ct[1].values)):
    axes[0].text(i, a_ + b_ + 4, f"{a_ + b_}", ha="center", fontsize=7)
axes[0].set_ylabel("pacientes"); axes[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=6); axes[0].set_title("Clases", loc="left"); panel(axes[0], "a")
Pm = P.mean(0)
for lead, c in (("V1", C_ACC), ("V2", C_BRU)):
    axes[1].semilogy(f, Pm[:, LEAD_IDX[lead]], c=c, label=lead)
axes[1].axvspan(0, 0.5, color=C_GREY, alpha=0.18, lw=0); axes[1].axvspan(40, 50, color=C_GREY, alpha=0.18, lw=0)
axes[1].text(38.5, Pm[:, LEAD_IDX["V1"]].max() * 0.02, "ruido\n> 40 Hz", fontsize=6, color="0.35", ha="right")
axes[1].set_xlabel("frecuencia (Hz)"); axes[1].set_ylabel("PSD media (mV²/Hz)"); axes[1].set_xlim(0, 50); axes[1].legend(); axes[1].set_title("Espectro medio", loc="left"); panel(axes[1], "b")
q = pd.read_csv(RESULTS / "etl_calidad_latidos.csv").rename(columns={"fc_lpm": "hr"})
bins = np.arange(35, 165, 7.5)
for cls, c, lab in ((0, C_NORM, "Normal"), (1, C_BRU, "Brugada")):
    axes[2].hist(q.hr[q.y == cls], bins=bins, color=c, alpha=0.55, density=True, label=lab)
axes[2].set_xlabel("frecuencia cardíaca (lpm)"); axes[2].set_yticks([]); axes[2].spines["left"].set_visible(False); axes[2].legend(); axes[2].set_title("Ritmo", loc="left"); panel(axes[2], "c")
fig.tight_layout(w_pad=1.2); save(fig, "calidad_datos")
NUM["fc_mediana_normal"], NUM["fc_mediana_brugada"] = float(q.hr[q.y == 0].median()), float(q.hr[q.y == 1].median())
NUM["fc_mannwhitney_p"] = float(stats.mannwhitneyu(q.hr[q.y == 0], q.hr[q.y == 1]).pvalue)

# ------------------------------------------------------------------ F3 segmentación, alineamiento y rechazo
P2 = pd.read_csv(RESULTS / "instancias_por_paciente.csv"); P2["id"] = P2.id.astype(str)
# ejemplo didáctico: ritmo normal (75 lpm) con 4 de 14 latidos anómalos
k3 = int(np.where(ids == "3118448")[0][0]); NUM["ejemplo_latidos_id"] = ids[k3]
info = extract_beats(Xp[k3])
raw_beats = segment_beats(Xp[k3], info.rpeaks)
from brugada.beats import align_beats
al = align_beats(raw_beats, LEAD_IDX["II"])
iso = np.median(al, 0)[BEAT_PRE - 8:BEAT_PRE - 3].mean(0, keepdims=True)
rej = info.corr_all < 0.6
fig, axes = plt.subplots(1, 3, figsize=(W, 2.1), sharex=True)
for b in raw_beats - iso: axes[0].plot(t_beat, b[:, LEAD_IDX["II"]], c=C_GREY, lw=0.5, alpha=0.8)
axes[0].set_title("II, sin alinear", loc="left", fontweight="normal"); panel(axes[0], "a")
for b, rj in zip(al - iso, rej): axes[1].plot(t_beat, b[:, LEAD_IDX["II"]], c=C_BRU if rj else C_GREY, lw=0.9 if rj else 0.5, alpha=0.9, zorder=3 if rj else 1)
axes[1].set_title("II, alineados", loc="left", fontweight="normal"); NUM["ejemplo_latidos_descartados"] = int(rej.sum()); NUM["ejemplo_latidos_total"] = int(len(rej)); panel(axes[1], "b")
for b, rj in zip(al - iso, rej):
    if not rj: axes[2].plot(t_beat, b[:, LEAD_IDX["V2"]], c=C_GREY, lw=0.5, alpha=0.7)
axes[2].plot(t_beat, info.mean_beat[:, LEAD_IDX["V2"]], c="k", lw=1.6, label="latido medio")
axes[2].set_title("V2, latido medio", loc="left", fontweight="normal"); axes[2].legend(loc="lower right"); panel(axes[2], "c")
for a in axes: a.set_xlabel("ms desde el pico R"); a.axvline(0, c="0.6", lw=0.5, ls=":"); comma(a, "y")
axes[0].set_ylabel("mV"); fig.tight_layout(w_pad=0.8); save(fig, "latidos_segmentacion")

# ------------------------------------------------------------------ F4 latido medio por clase
fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=False)
for ax, lead in zip(axes, ("V1", "V2")):
    for lo_, hi_, lab in ((-50, 60, "QRS"), (60, 180, "ST"), (180, 450, "T")):
        ax.axvspan(lo_, hi_, color="0.5", alpha=0.06 if lab != "ST" else 0.13, lw=0)
    for cls, c, lab in ((0, C_NORM, f"Normal (n = {(y == 0).sum()})"), (1, C_BRU, f"Brugada (n = {(y == 1).sum()})")):
        m = MB[y == cls, :, LEAD_IDX[lead]]
        ax.fill_between(t_beat, np.percentile(m, 25, 0), np.percentile(m, 75, 0), color=c, alpha=0.18, lw=0)
        ax.plot(t_beat, np.median(m, 0), c=c, lw=1.4, label=lab)
    ymax = ax.get_ylim()[1]
    for x_, lab in ((5, "QRS"), (120, "ST"), (315, "T")):
        ax.text(x_, ymax * 0.97, lab, ha="center", va="top", fontsize=7, color="0.35")
    ax.axhline(0, c="0.6", lw=0.5); ax.set_xlabel("ms desde el pico R"); ax.set_title(f"Derivación {lead}", loc="left"); comma(ax, "y")
axes[0].set_ylabel("mV"); axes[1].legend(loc="lower right"); panel(axes[0], "a"); panel(axes[1], "b")
fig.tight_layout(w_pad=1.0); save(fig, "latido_medio_clases")

# ------------------------------------------------------------------ F5 selección de instancias
S = pd.read_csv(RESULTS / "instancias_resumen.csv")
lab_c = {"none": "ninguno", "corr0.5": "r<0,5", "corr0.6": "r<0,6", "corr0.7": "r<0,7", "corr0.8": "r<0,8", "corr0.9": "r<0,9", "rr0.2": "RR±20 %", "corr0.6+rr0.2": "r<0,6\n+RR"}
fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
xs = np.arange(len(S))
axes[0].bar(xs, S.pct_descartados, color=[C_ACC if c == "corr0.6" else "#BFC9D6" for c in S.criterio], width=0.62)
axes[0].set_xticks(xs); axes[0].set_xticklabels([lab_c[c] for c in S.criterio], fontsize=6.3)
axes[0].set_ylabel("% de latidos descartados"); comma(axes[0], "y")
ax2 = axes[0].twinx(); ax2.spines["right"].set_visible(True)
ax2.errorbar(xs, S.auroc, yerr=S.auroc_std / np.sqrt(15), fmt="o", c=C_BRU, ms=3.5, capsize=2, lw=0.8)
ax2.set_ylim(0.88, 0.95); ax2.set_ylabel("AUROC (± error típico)", color=C_BRU); ax2.tick_params(axis="y", colors=C_BRU); comma(ax2, "y", "{:.2f}")
axes[0].set_title("Criterios de rechazo", loc="left"); panel(axes[0], "a")
for cls, c, lab in ((0, C_NORM, "Normal"), (1, C_BRU, "Brugada")):
    axes[1].hist(P2.corr_min_all[P2.y == cls].clip(lower=0), bins=np.linspace(0, 1, 26), color=c, alpha=0.55, density=True, label=lab)
axes[1].axvline(0.6, c="k", ls="--", lw=0.8); axes[1].text(0.58, axes[1].get_ylim()[1] * 0.45, "umbral 0,6", ha="right", fontsize=6.5)
axes[1].set_xlabel("correlación del peor latido"); axes[1].set_yticks([]); axes[1].spines["left"].set_visible(False); axes[1].legend(loc="upper left"); comma(axes[1], "x")
axes[1].set_title("Peor latido por paciente", loc="left"); panel(axes[1], "b")
fig.tight_layout(w_pad=1.5); save(fig, "seleccion_instancias")

# ------------------------------------------------------------------ F6 características discriminantes
sel = [("V1_st_slope", "Pendiente ST, V1", "mV/ms", 1000), ("V2_st_slope", "Pendiente ST, V2", "mV/ms", 1000),
       ("qrs_dur_ms", "Duración QRS", "ms", 1), ("V2_t_amp", "Amplitud T, V2", "mV", 1),
       ("V2_rprime_amp", "Amplitud r', V2", "mV", 1), ("V1_st_j40", "ST en J+40 ms, V1", "mV", 1)]
fig, axes = plt.subplots(2, 3, figsize=(W, 3.6))
rng = np.random.default_rng(0)
for ax, (col, tit, unit, sc), l in zip(axes.ravel(), sel, "abcdef"):
    vals = [F[col].values[y == c] * sc for c in (0, 1)]
    bp = ax.boxplot(vals, widths=0.5, showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=1))
    for patch, c in zip(bp["boxes"], (C_NORM, C_BRU)): patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for i, v in enumerate(vals):
        ax.scatter(i + 1 + rng.uniform(-0.17, 0.17, len(v)), v, s=2.5, c=(C_NORM, C_BRU)[i], alpha=0.45, lw=0)
    auc = roc_auc_score(y, F[col]); auc = max(auc, 1 - auc)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Normal", "Brugada"])
    ax.set_title(f"{tit} (AUC {auc:.2f})".replace(".", ","), loc="left", fontweight="bold" if auc > 0.7 else "normal")
    ax.set_ylabel(f"{unit}" + (" × 10⁻³" if sc == 1000 else "")); panel(ax, l)
    lo_, hi_ = np.percentile(np.concatenate(vals), [1, 99]); pad = (hi_ - lo_) * 0.08; ax.set_ylim(lo_ - pad, hi_ + pad)
    comma(ax, "y", "{:.1f}" if unit != "ms" else "{:.0f}")
fig.tight_layout(h_pad=1.0, w_pad=0.8); save(fig, "caracteristicas_clave")
vc = F.qrs_dur_ms.value_counts().sort_index(); NUM["qrs_valores_distintos"] = int(len(vc)); NUM["qrs_valores"] = {int(k): int(v) for k, v in vc.items()}

# ------------------------------------------------------------------ F7 matriz de correlación
from scipy.cluster.hierarchy import linkage, leaves_list
X = F[names].values.astype(float)
Cm = np.nan_to_num(np.corrcoef(X, rowvar=False)); n = len(names)
order = leaves_list(linkage(1 - np.abs(Cm[np.triu_indices(n, 1)]), "average"))
fig, ax = plt.subplots(figsize=(W, W * 0.92))
im = ax.imshow(Cm[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
lab = [names[i].replace("_", " ") for i in order]
ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=90, fontsize=4.2); ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=4.2)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015); cb.set_label("correlación de Pearson"); cb.ax.tick_params(labelsize=6)
save(fig, "correlacion_caracteristicas")
iu = np.triu_indices(n, 1); NUM["pares_r09"] = int((np.abs(Cm[iu]) > 0.9).sum()); NUM["pares_r07"] = int((np.abs(Cm[iu]) > 0.7).sum()); NUM["pares_total"] = int(len(iu[0]))
NUM["vars_con_r09"] = int(((np.abs(Cm) > 0.9).sum(0) - 1 > 0).sum())

# ------------------------------------------------------------------ F8 selección de características: AUROC según k
FAM_OF = {"ANOVA F": "filtro", "Información mutua": "filtro", "Regularización L1": "embebido", "Random Forest": "embebido", "Orden aleatorio": "ref"}
STYLE = {"ANOVA F": "-", "Información mutua": "--", "Regularización L1": "-", "Random Forest": "--", "Orden aleatorio": ":"}
A = pd.read_csv(RESULTS / "seleccion_auroc.csv")
fig, ax = plt.subplots(figsize=(W * 0.72, 2.7))
for m in FAM_OF:
    g = A[A.method == m]
    ax.plot(g.k, g.auroc, STYLE[m], c=FAM[FAM_OF[m]], marker="o", ms=2.6, lw=1.1, label=m)
a72 = A[(A.k == 72) & (A.method == "ANOVA F")].auroc.iloc[0]
ax.axhline(a72, c="k", lw=0.6, ls=":")
ax.text(1.05, a72 + 0.006, f"todas las variables ({a72:.3f})".replace(".", ","), fontsize=6.3, color="0.3", va="bottom")
ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20, 45, 72]); ax.set_xticklabels([1, 2, 3, 5, 10, 20, 45, 72]); ax.set_xlabel("número de variables k")
ax.grid(alpha=0.25, lw=0.4); ax.set_ylim(0.45, 0.95); ax.set_ylabel("AUROC medio (15 folds)"); comma(ax, "y")
ax.legend(loc="lower right", fontsize=6.8, frameon=False)
fig.tight_layout(); save(fig, "seleccion_caracteristicas_curvas")

# ------------------------------------------------------------------ F9 frecuencia de selección
fr = pd.read_csv(RESULTS / "seleccion_frecuencia_top10.csv", index_col=0)
cons = fr.mean(1).sort_values(ascending=False).head(12).index
fig, ax = plt.subplots(figsize=(W, 2.2))
im = ax.imshow(fr.loc[cons].T.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(cons))); ax.set_xticklabels([c.replace("_", " ") for c in cons], rotation=40, ha="right", fontsize=6.5)
ax.set_yticks(range(fr.shape[1])); ax.set_yticklabels(fr.columns, fontsize=7)
for i in range(fr.shape[1]):
    for j, c in enumerate(cons):
        v = fr.loc[c].iloc[i]; ax.text(j, i, f"{v:.0%}".replace("%", ""), ha="center", va="center", fontsize=5.6, color="white" if v > 0.6 else "0.25")
ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("fracción de folds en el top-10"); cb.ax.tick_params(labelsize=6)
save(fig, "seleccion_caracteristicas_frecuencia")

# ------------------------------------------------------------------ F11 reducción de dimensionalidad
from brugada.experiments.run_dimred import embed, datasets
Q = pd.read_csv(RESULTS / "dimred_knn.csv")
# Pacientes que la mayoría de los modelos del TFG clasifica mal (fracción de aciertos < 0,5; ver README)
H = pd.read_csv(RESULTS / "tfg_aciertos_por_paciente.csv"); H["id"] = H.id.astype(str)
hard = H.set_index("id").fraccion_aciertos.reindex(ids).values < 0.5
data = datasets(F, MB)
E = {d: embed(Xd) for d, Xd in data.items()}
fig, axes = plt.subplots(2, 3, figsize=(W, 4.0))
for i, d in enumerate(data):
    for j, mname in enumerate(("PCA", "t-SNE", "UMAP")):
        ax = axes[i, j]; Z = E[d][mname]
        for cls, c, lab in ((0, C_NORM, "Normal"), (1, C_BRU, "Brugada")):
            ax.scatter(Z[y == cls, 0], Z[y == cls, 1], s=4, c=c, alpha=0.55, lw=0, label=lab, zorder=2 if cls else 1)
        ax.scatter(Z[hard, 0], Z[hard, 1], s=16, facecolors="none", edgecolors="k", lw=0.5, label="error persistente", zorder=3)
        qq = Q[(Q.datos == d) & (Q.metodo == mname)].iloc[0]
        ax.set_title(f"{mname} · {d}", loc="left", fontsize=7.3)
        ax.text(0.02, 0.02, f"k-NN AUROC {qq.knn_auroc:.2f}".replace(".", ","), transform=ax.transAxes, fontsize=6, va="bottom", color="0.25")
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(True) for s in ax.spines.values()]; [s.set_linewidth(0.4) for s in ax.spines.values()]
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), markerscale=2)
fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.8, w_pad=0.5); save(fig, "reduccion_dimensionalidad")

from sklearn.decomposition import PCA
Xs = data["72 características"]; pca = PCA(random_state=42).fit(Xs); cum = np.cumsum(pca.explained_variance_ratio_)
load = pd.DataFrame(pca.components_[:2].T, index=names, columns=["PC1", "PC2"])
fig, axes = plt.subplots(1, 3, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
axes[0].bar(range(1, 26), pca.explained_variance_ratio_[:25] * 100, color="#BFC9D6"); axes[0].plot(range(1, 26), cum[:25] * 100, "o-", c=C_ACC, ms=2)
axes[0].axhline(90, c="0.5", ls=":", lw=0.6); axes[0].set_xlabel("componente"); axes[0].set_ylabel("% varianza (acumulada)"); axes[0].set_title("Varianza explicada", loc="left"); panel(axes[0], "a")
for ax, pc, l in zip(axes[1:], ("PC1", "PC2"), "bc"):
    top = load[pc].abs().sort_values(ascending=False).head(8).index[::-1]
    ax.barh(range(8), load.loc[top, pc], color=[C_BRU if v > 0 else C_NORM for v in load.loc[top, pc]], height=0.6)
    ax.set_yticks(range(8)); ax.set_yticklabels([t.replace("_", " ") for t in top], fontsize=6); ax.axvline(0, c="0.4", lw=0.5)
    ax.set_title(f"Cargas de {pc} ({pca.explained_variance_ratio_[int(pc[-1]) - 1] * 100:.0f} %)", loc="left"); comma(ax, "x"); panel(ax, l)
fig.tight_layout(w_pad=0.6); save(fig, "pca_varianza_cargas")

# ------------------------------------------------------------------ F12 modelo de referencia
def oof(name):
    d = np.load(RESULTS / "runs" / f"{name}_oof.npz", allow_pickle=True); return d["y"], d["score"], d["pred"]
yl, sl, pl = oof("lda__features"); _, sr, _ = oof("puntuacion_clinica__features"); _, sg, _ = oof("logreg__features")
_, _, p2 = oof("regla_2_de_3__features"); _, _, p4 = oof("regla_3_de_3__features")
fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
for s, c, lab, lw in ((sl[0], C_ACC, "LDA", 1.5), (sg[0], "#6A9FCF", "Regresión logística", 1.0), (sr[0], C_GREY, "Puntuación clínica ST+|T|", 1.0)):
    fpr, tpr, _ = roc_curve(yl, s); ax.plot(fpr, tpr, c=c, lw=lw, label=f"{lab} ({roc_auc_score(yl, s):.3f})".replace(".", ","))
for p, mk, lab in ((p2[0], "s", "Regla «2 de 3»"), (p4[0], "D", "Regla «3 de 3» (tipo 1)")):
    cmx = confusion_matrix(yl, p, labels=[0, 1]); tn, fp, fn, tp = cmx.ravel()
    ax.plot(fp / (fp + tn), tp / (tp + fn), mk, c="k", ms=4, mfc="white", label=lab)
ax.plot([0, 1], [0, 1], c="0.75", ls=":", lw=0.6); ax.set_xlabel("1 − especificidad"); ax.set_ylabel("sensibilidad"); ax.legend(loc="lower right", fontsize=6.2)
comma(ax); ax.set_title("Curvas ROC (predicciones fuera de muestra)", loc="left"); panel(ax, "a")
cm_all = sum(confusion_matrix(yl, pl[r_], labels=[0, 1]) for r_ in range(pl.shape[0])) / pl.shape[0]
ax = axes[1]; ax.imshow(cm_all / cm_all.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        v = cm_all[i, j]; ax.text(j, i, f"{v:.0f}\n({v / cm_all[i].sum():.0%})".replace("%", " %"), ha="center", va="center", fontsize=7.5, color="white" if (cm_all[i, j] / cm_all[i].sum()) > 0.6 else "0.15")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Brugada"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Brugada"])
ax.set_xlabel("predicción"); ax.set_ylabel("clase real"); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(length=0)
ax.set_title("Matriz de confusión LDA (media de 3 rep.)", loc="left"); panel(ax, "b")
fig.tight_layout(w_pad=1.5); save(fig, "modelo_referencia")
NUM["auc_oof_lda"], NUM["auc_oof_regla"], NUM["auc_oof_lr"] = float(roc_auc_score(yl, sl[0])), float(roc_auc_score(yl, sr[0])), float(roc_auc_score(yl, sg[0]))
NUM["cm_lda_media"] = cm_all.round(1).tolist()

json.dump(NUM, open(OUT / "numeros.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: v for k, v in NUM.items() if k != "qrs_valores"}, indent=1, ensure_ascii=False, default=float)[:3000])
