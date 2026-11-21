# Reproduce todos los resultados y figuras de la memoria. Uso: make all
PY ?= ../.venv/bin/python -W ignore
EXP = cd src && $(PY) -m brugada.experiments

.PHONY: all data etl preprocesado instancias caracteristicas visualizacion modelo figuras memoria
all: etl preprocesado instancias caracteristicas visualizacion modelo figuras

data:            ; bash scripts/download_data.sh
etl:             ; $(EXP).run_etl
preprocesado:    ; $(EXP).run_preprocessing
instancias:      ; $(EXP).run_instance_selection
caracteristicas: ; $(EXP).run_features
visualizacion:   ; $(EXP).run_dimred
modelo:          ; $(EXP).run_baseline
figuras:         ; cd src && $(PY) ../memoria/make_figures.py
memoria:         ; cd memoria && pdflatex main.tex && pdflatex main.tex
