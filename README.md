# Détection Dashcam YOLOv8

Ce projet permet de détecter des objets et des panneaux de signalisation dans des vidéos de dashcam en utilisant YOLOv8.

## Fonctionnalités

- Détection d'objets avec YOLOv8 (modèle COCO)
- Détection de panneaux de signalisation (modèle personnalisé)
- Support pour fichiers vidéo et webcam
- Affichage en temps réel avec FPS
- Possibilité de sauvegarder la vidéo annotée
- Capture d'écran avec la touche 'S'
- Interface web Streamlit pour faciliter l'utilisation

## Prérequis

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics (YOLOv8)
- Streamlit

```bash
pip install -r requirements.txt
```

## Utilisation

### Interface Streamlit (recommandé)

```bash
# Lancer l'interface web
./run_app.sh

# Ou directement avec streamlit
streamlit run app.py
```

L'interface web permet de:

- Déposer un fichier vidéo pour analyse
- Sélectionner une webcam parmi celles détectées
- Prendre des captures d'écran pendant l'analyse
- Arrêter l'analyse à tout moment

### Mode ligne de commande

```bash
# Lancer avec un fichier vidéo
python detect_dashcam.py --video chemin/vers/video.mp4

# Lancer avec la webcam
python detect_dashcam.py --camera 0

# Options supplémentaires
python detect_dashcam.py --video video.mp4 --conf 0.3 --iou 0.5 --scale 0.5 --skip 1 --output resultat.mp4
```

## Options disponibles (mode ligne de commande)

- `--video`, `-v` : Chemin vers le fichier vidéo
- `--camera`, `-c` : ID de la caméra (0, 1, etc.)
- `--output`, `-o` : Sauvegarder la vidéo annotée
- `--conf` : Seuil de confiance (défaut: 0.25)
- `--iou` : Seuil IoU pour NMS (défaut: 0.55)
- `--scale` : Échelle de redimensionnement (défaut: 0.75)
- `--skip` : Nombre de frames à ignorer (défaut: 2)

## Contrôles

### Interface Streamlit

- Bouton "Arrêter la lecture" : Arrête l'analyse en cours
- Bouton "Prendre un screenshot" : Capture l'image actuelle

### Mode ligne de commande

- `Q` : Quitter
- `ESPACE` : Pause/Reprendre
- `S` : Capture d'écran
- `←/→` : Reculer/Avancer de 10 secondes (mode fichier vidéo)

## Modèles

- `yolov8n.pt` : Modèle YOLOv8 nano pour la détection d'objets (COCO)
- `best.pt` : Modèle personnalisé pour la détection de panneaux de signalisation

## Structure du projet

```
.
├── app.py              # Application Streamlit
├── detect_dashcam.py   # Script ligne de commande
├── run_app.sh          # Script de lancement de l'interface
├── yolov8n.pt          # Modèle COCO
├── best.pt             # Modèle panneaux de signalisation
└── README.md           # Documentation
```
