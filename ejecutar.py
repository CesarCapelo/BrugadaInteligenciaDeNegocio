"""Ejecuta todo el trabajo: los seis experimentos (capítulos 4 a 9) y las figuras de la memoria.

Uso, desde esta carpeta y con el entorno instalado (ver README.md):
    .venv/bin/python ejecutar.py           (Windows: .venv\\Scripts\\python ejecutar.py)
Los resultados se guardan en results/ y las figuras en memoria/figuras/.
"""
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PASOS = [
    ("Capítulo 4. Datos, ETL y calidad", ["-m", "brugada.experiments.run_etl"]),
    ("Capítulo 5. Preprocesado de la señal", ["-m", "brugada.experiments.run_preprocessing"]),
    ("Capítulo 6. Selección de instancias", ["-m", "brugada.experiments.run_instance_selection"]),
    ("Capítulo 7. Características", ["-m", "brugada.experiments.run_features"]),
    ("Capítulo 8. Visualización", ["-m", "brugada.experiments.run_dimred"]),
    ("Capítulo 9. Modelo de referencia", ["-m", "brugada.experiments.run_baseline"]),
    ("Figuras de la memoria", [str(RAIZ / "memoria" / "make_figures.py")]),
]

if not (RAIZ / "data" / "raw" / "RECORDS").exists():
    sys.exit("Faltan los datos en data/raw/. Se descargan con: bash scripts/download_data.sh")

inicio = time.time()
for nombre, args in PASOS:
    print(f"\n== {nombre}", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", *args], cwd=RAIZ / "src", check=True)
print(f"\nTerminado en {(time.time() - inicio) / 60:.1f} minutos.")
