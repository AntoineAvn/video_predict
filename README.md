# Détection d'objets sur vidéo dashcam avec YOLOv8

Ce projet permet de détecter en temps réel les objets pertinents pour la conduite autonome dans une vidéo de dashcam en utilisant YOLOv8.

## Prérequis

- Python 3.8+
- Environnement conda "autonomous" avec les packages suivants:
  - ultralytics
  - opencv-python

## Utilisation

1. Activez l'environnement conda:

```bash
conda activate autonomous
```

2. Exécutez le script:

```bash
python detect_dashcam.py
```

Ou utilisez le script shell:

```bash
./run.sh
```

3. Contrôles:
   - Appuyez sur 'q' pour quitter le programme

## Fonctionnalités

- Détection en temps réel des objets pertinents pour la conduite autonome:
  - Personnes (classe 0)
  - Vélos (classe 1)
  - Voitures (classe 2)
  - Motos (classe 3)
  - Bus (classe 5)
  - Camions (classe 7)
  - Feux de circulation (classe 9)
  - Panneaux stop (classe 11)
- Affichage du FPS pour évaluer les performances
- Optimisé pour CPU MacBook Pro M3
- Redémarrage automatique de la vidéo à la fin

## Notes

- Le modèle YOLOv8n est utilisé pour la détection
- Aucun enregistrement vidéo n'est effectué
