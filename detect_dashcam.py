import cv2
import time
from ultralytics import YOLO
import numpy as np

# Charger le modèle YOLOv8
model = YOLO('yolov8n.pt')

# Optimisation pour CPU M3
model.to('cpu')

# Configuration pour améliorer les performances
cv2.setNumThreads(6)  # Utiliser plusieurs threads pour OpenCV

# Ouvrir la vidéo
cap = cv2.VideoCapture('video.mp4')

# Vérifier si la vidéo est ouverte correctement
if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo")
    exit()

# Obtenir les propriétés de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Résolution vidéo: {frame_width}x{frame_height}, FPS: {fps}")

# Variables pour calculer le FPS réel
prev_time = 0
fps_list = []

# Classes pertinentes pour les voitures autonomes
# 0: personne, 1: vélo, 2: voiture, 3: moto, 5: bus, 7: camion, 9: feu de circulation, 11: panneau stop
relevant_classes = [0, 1, 2, 3, 5, 7, 9, 11]

# Paramètres pour le redimensionnement (améliore les performances)
scale_factor = 0.75  # Réduire la taille de l'image pour accélérer la détection
resize_width = int(frame_width * scale_factor)
resize_height = int(frame_height * scale_factor)

# Paramètres de détection
conf_threshold = 0.25
skip_frames = 2  # Traiter une image sur N pour améliorer les performances
frame_count = 0

# Variable pour stocker les derniers résultats valides
last_results = None
last_annotated_frame = None

# Paramètres de lissage pour éviter le scintillement
detection_persistence = 3  # Nombre de frames pendant lesquelles une détection persiste

while True:
    # Lire une frame de la vidéo
    ret, frame = cap.read()
    
    # Si la vidéo est terminée, redémarrer
    if not ret:
        print("Fin de la vidéo, redémarrage...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # Incrémenter le compteur de frames
    frame_count += 1
    
    # Calculer le FPS actuel
    current_time = time.time()
    fps_current = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time
    
    # Maintenir une liste des derniers FPS pour stabiliser l'affichage
    fps_list.append(fps_current)
    if len(fps_list) > 30:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)
    
    # Effectuer la détection sur certaines frames
    if frame_count % skip_frames == 0:
        # Redimensionner l'image pour accélérer la détection
        resized_frame = cv2.resize(frame, (resize_width, resize_height))
        
        # Effectuer la détection avec YOLOv8
        results = model.predict(resized_frame, conf=conf_threshold, classes=relevant_classes, verbose=False)
        
        # Stocker les résultats pour les utiliser dans les frames suivantes
        last_results = results
        
        # Visualiser les résultats sur l'image originale
        annotated_frame = results[0].plot()
        
        # Redimensionner à la taille originale pour l'affichage
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
        
        # Stocker la frame annotée pour les frames suivantes
        last_annotated_frame = annotated_frame.copy()
    else:
        # Pour les frames ignorées, utiliser la dernière frame annotée si disponible
        if last_annotated_frame is not None:
            # Créer une copie de la frame actuelle
            annotated_frame = frame.copy()
            
            # Si nous avons des résultats précédents, les appliquer à la frame actuelle
            if last_results is not None:
                # Récupérer les boîtes et les classes de la dernière détection
                boxes = last_results[0].boxes
                
                # Dessiner les boîtes sur la frame actuelle
                for box in boxes:
                    # Récupérer les coordonnées et la classe
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    # Ajuster les coordonnées pour la taille originale
                    x1 = int(x1 / scale_factor)
                    y1 = int(y1 / scale_factor)
                    x2 = int(x2 / scale_factor)
                    y2 = int(y2 / scale_factor)
                    
                    # Récupérer le nom de la classe
                    class_name = model.names[cls]
                    
                    # Dessiner la boîte
                    color = (0, 255, 0)  # Vert
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Ajouter le texte avec la classe et la confiance
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            annotated_frame = frame
    
    # Ajouter le FPS à l'image
    cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Afficher la frame
    cv2.imshow("Détection Dashcam YOLOv8", annotated_frame)
    
    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows() 