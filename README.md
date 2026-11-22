# Preprocesamiento y extracción de características de señales ECG para la detección del síndrome de Brugada

Trabajo de teoría de Inteligencia de Negocio (Grado en Ingeniería Informática en Tecnologías de la
Información, Universidad de Oviedo). Autor: César Capelo. Es la primera parte de la memoria del Trabajo de Fin de
Grado *Análisis comparativo de metodologías de aprendizaje automático para la detección del síndrome de Brugada*
(tutor: Nahuel Costa Cortez).

Contiene el código que procesa los ECG del conjunto Brugada-HUCA y construye la tabla de 72 variables clínicas de
cada paciente, la memoria en LaTeX ([`memoria/main.pdf`](memoria/main.pdf)) y los resultados con los que se ha
escrito.

| Capítulo de la memoria | Qué se hace | Script |
|---|---|---|
| 4. Datos y proceso ETL | Carga, integración con los metadatos, duplicados, saturación, ruido, deriva, definición de la etiqueta | `run_etl` |
| 5. Preprocesado de la señal | Filtro paso banda de fase cero, detección de picos R, segmentación y alineamiento; comparación de alternativas | `run_preprocessing` |
| 6. Selección de instancias | Rechazo de latidos anómalos (morfológico y temporal) y latido medio por paciente | `run_instance_selection` |
| 7. Características | 72 variables clínicas, análisis univariante, redundancia y comparación de 4 métodos de selección | `run_features` |
| 8. Visualización | PCA, t-SNE y UMAP | `run_dimred` |
| 9. Modelo de referencia | LDA con contracción y regresión logística frente a las reglas clínicas | `run_baseline` |

## Resultados principales

* El conjunto de datos es de buena calidad. El índice publicado tiene un registro duplicado (paciente 596382),
  que se elimina.
* Solo se descarta el 0,9 % de los latidos, sin sesgo entre clases. Ningún criterio de rechazo cambia el
  resultado, porque la mediana ya es robusta.
* La pendiente del segmento ST discrimina mucho más que su elevación (AUC 0,845 frente a 0,530).
* Los cuatro métodos de selección rinden prácticamente igual. Con solo 5 variables se conserva casi el 99 %
  del rendimiento, y los cuatro métodos coinciden en las variables que eligen (pendiente del ST, duración del QRS
  y onda T en V2).
* Ninguna proyección en dos dimensiones separa las clases.
* El LDA con contracción alcanza un AUROC de 0,928 (IC 95 %: 0,88–0,96), con sensibilidad 0,87 y
  especificidad 0,88. La puntuación basada en los criterios clínicos se queda en 0,754, y la regla «3 de 3» solo detecta al 22 % de los pacientes.

## Datos

Costa Cortez, N. y García Iglesias, D. (2026). *Brugada-HUCA: 12-Lead ECG Recordings for the Study of Brugada
Syndrome* (v1.0.0). PhysioNet. <https://physionet.org/content/brugada-huca/1.0.0/> (licencia CC BY-SA 4.0).

Son 363 pacientes con un ECG de 12 derivaciones de 12 s a 100 Hz. La etiqueta binaria es positiva si
`brugada` ∈ {1, 2}, lo que da 76 positivos y 287 negativos. En el comprimido de la entrega las señales ya
vienen en `data/raw/`; si esa carpeta está vacía, se descargan con el script de abajo.

## Instalación

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh      # solo si data/raw/ está vacío: descarga los datos de PhysioNet
```

En Windows: `py -3.12 -m venv .venv` y `.venv\Scripts\pip install -r requirements.txt`.

## Ejecución

Para ejecutarlo todo (los seis experimentos y las figuras de la memoria), desde la carpeta del proyecto:

```bash
.venv/bin/python ejecutar.py
```

En Windows: `.venv\Scripts\python ejecutar.py`. Los resultados se escriben en `results/` y las figuras en
`memoria/figuras/`.

Cada script también se puede ejecutar por separado desde `src/`:

```bash
cd src
../.venv/bin/python -m brugada.experiments.run_etl                 # cap. 4
../.venv/bin/python -m brugada.experiments.run_preprocessing       # cap. 5
../.venv/bin/python -m brugada.experiments.run_instance_selection  # cap. 6
../.venv/bin/python -m brugada.experiments.run_features            # cap. 7
../.venv/bin/python -m brugada.experiments.run_dimred              # cap. 8
../.venv/bin/python -m brugada.experiments.run_baseline            # cap. 9
../.venv/bin/python ../memoria/make_figures.py                     # figuras de la memoria
```

Todo el proceso tarda 10 minutos en un portátil. También se puede usar `make all`. La memoria se compila desde `memoria/` con
`pdflatex main.tex` dos veces (la segunda, para el índice).

La primera ejecución guarda en `data/processed/` las señales y los latidos medios. No lances dos experimentos a
la vez hasta que esa caché exista, porque los dos escribirían el mismo fichero.

## Protocolo de evaluación

* Validación cruzada estratificada a nivel de paciente, de 5 particiones repetida 3 veces (15 *folds*), con
  semilla 42.
* Todo lo que se ajusta con datos (escalado, hiperparámetros, ordenación de las variables y umbral de decisión)
  se calcula solo con la partición de entrenamiento de cada *fold*. No hay fuga de información.
* El umbral se elige con el índice de Youden sobre las predicciones fuera de muestra de la validación interna.
* El desbalance se trata con pesos de clase. Se informa del AUROC y de la exactitud balanceada: la media ± la
  desviación típica en los 15 *folds* y un intervalo de confianza *bootstrap* del 95 %.
* Se guardan las predicciones fuera de muestra de cada paciente (`results/runs*/…_oof.npz`), de modo que
  cualquier tabla o figura se puede regenerar sin volver a entrenar.

## Estructura

```
ejecutar.py           ejecuta todos los experimentos y las figuras
src/brugada/
  config.py           rutas, constantes (100 Hz, ventana del latido de -250 a +450 ms) y semilla
  data.py             lectura WFDB, eliminación del registro duplicado, etiqueta y huella SHA-256
  preprocess.py       filtros: paso banda de fase cero y alternativas comparadas
  beats.py            detección de picos R, segmentación, alineamiento, rechazo de latidos y latido medio
  features.py         límites del QRS y las 72 características clínicas
  rules.py            reglas clínicas («2 de 3», «3 de 3») y puntuación ST + |T|
  representations.py  construcción y caché del latido medio y de la tabla de características
  models.py           LDA con contracción y regresión logística con validación cruzada anidada
  splits.py           particiones a nivel de paciente
  evaluate.py         métricas, umbral de Youden e intervalos bootstrap
  experiments/        un script por capítulo (ver la tabla de arriba) y el bucle de validación común
memoria/              fuente LaTeX (main.tex y estilo.tex), figuras (figuras/), script de figuras y PDF compilado
results/              resultados: resúmenes .csv/.json y predicciones por paciente
scripts/              descarga de los datos
data/raw/             conjunto de datos de PhysioNet (incluido en el comprimido de la entrega)
```

## Licencia de los datos

El conjunto de datos Brugada-HUCA se distribuye con licencia CC BY-SA 4.0. Si usas estos datos, cita la
referencia de arriba y la de PhysioNet (Goldberger et al., 2000).
