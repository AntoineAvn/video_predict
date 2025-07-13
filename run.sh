#!/bin/bash

# Activer l'environnement conda
eval "$(conda shell.bash hook)"
conda activate autonomous

# Exécuter le script de détection
python detect_dashcam.py

# Désactiver l'environnement conda à la fin
conda deactivate 