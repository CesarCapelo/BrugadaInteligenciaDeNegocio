"""Rutas, constantes y semillas globales del proyecto."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.environ.get("BRUGADA_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
for _p in (DATA_PROCESSED, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

FS = 100                      # Hz
DURATION_S = 12               # s
N_SAMPLES = FS * DURATION_S   # 1200
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX = {l: i for i, l in enumerate(LEADS)}

# Ventana del latido medio alrededor del pico R (en muestras a 100 Hz)
BEAT_PRE = 25    # 250 ms antes de R
BEAT_POST = 45   # 450 ms después de R
BEAT_LEN = BEAT_PRE + BEAT_POST  # 70 muestras = 700 ms

SEED = 42
N_FOLDS = 5
N_REPEATS = 3

# Definición de la etiqueta binaria.
# metadata.brugada: 0 = normal, 1 = Brugada confirmado, 2 = otros/atípicos.
# Por defecto se considera positivo brugada in {1, 2} (76 positivos; ver la sección 3.4 de la memoria).
POSITIVE_CODES_DEFAULT = (1, 2)
