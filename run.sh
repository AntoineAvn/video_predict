#!/bin/bash

# Script pour lancer la détection dashcam avec différentes options

# Afficher l'aide
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  echo "Usage: ./run.sh [option]"
  echo ""
  echo "Options:"
  echo "  --video [path]    Lancer avec un fichier vidéo (défaut: video.mp4)"
  echo "  --webcam [id]     Lancer avec une webcam (défaut: 0)"
  echo "  --save            Sauvegarder la vidéo annotée"
  echo "  --help            Afficher cette aide"
  echo ""
  echo "Exemples:"
  echo "  ./run.sh --video dashcam.mp4"
  echo "  ./run.sh --webcam 1"
  echo "  ./run.sh --video dashcam.mp4 --save"
  exit 0
fi

VIDEO_PATH="video.mp4"
WEBCAM_ID=0
SAVE_OUTPUT=""

# Analyser les arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --video)
      USE_WEBCAM=false
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        VIDEO_PATH=$2
        shift
      fi
      ;;
    --webcam)
      USE_WEBCAM=true
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        WEBCAM_ID=$2
        shift
      fi
      ;;
    --save)
      TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
      SAVE_OUTPUT="--output output_${TIMESTAMP}.mp4"
      ;;
    *)
      echo "Option inconnue: $1"
      echo "Utilisez ./run.sh --help pour voir les options disponibles"
      exit 1
      ;;
  esac
  shift
done

# Exécuter le script Python avec les options appropriées
if [ "$USE_WEBCAM" = true ]; then
  echo "🎥 Lancement avec webcam $WEBCAM_ID..."
  python detect_dashcam.py --camera $WEBCAM_ID $SAVE_OUTPUT
else
  echo "📽️ Lancement avec vidéo: $VIDEO_PATH..."
  python detect_dashcam.py --video $VIDEO_PATH $SAVE_OUTPUT
fi 