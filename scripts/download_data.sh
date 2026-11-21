#!/usr/bin/env bash
# Descarga Brugada-HUCA v1.0.0 desde PhysioNet (acceso abierto, CC BY-SA 4.0)
set -e
cd "$(dirname "$0")/../data/raw"
BASE=https://physionet.org/files/brugada-huca/1.0.0
for f in README.md metadata.csv metadata_dictionary.csv RECORDS LICENSE.txt SHA256SUMS.txt files.zip; do
  [ -f "$f" ] || curl -sL -o "$f" "$BASE/$f"
done
[ -d files ] || unzip -q files.zip
echo "OK: $(ls files | wc -l) registros"
