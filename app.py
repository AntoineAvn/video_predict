import streamlit as st
import cv2
import time
import torch, torchvision
from ultralytics import YOLO
import numpy as np
import platform
import os
from pathlib import Path
import tempfile

def configure_mac_camera(camera_id):
    """Configuration spéciale pour les caméras sur macOS"""
    cap = cv2.VideoCapture(camera_id)
    
    # Réglages pour améliorer la compatibilité avec macOS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Essayer différentes résolutions
    resolutions = [(1280, 720), (640, 480), (320, 240)]
    
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Tester si on peut lire une frame
        ret, _ = cap.read()
        if ret:
            st.success(f"✅ Caméra configurée avec succès: {width}x{height}")
            # Réinitialiser la caméra
            cap.release()
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
    
    # Si aucune résolution ne fonctionne, essayer sans configuration spéciale
    cap.release()
    return cv2.VideoCapture(camera_id)

def get_available_cameras():
    """Liste les caméras disponibles sur le système"""
    max_cameras = 10
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                available_cameras.append((i, width, height, fps))
            cap.release()
    
    return available_cameras

def save_uploaded_file(uploaded_file):
    """Sauvegarde temporaire du fichier uploadé"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def process_video(source_path, camera_id=None):
    """Traitement de la vidéo avec YOLOv8"""
    # Chargement des modèles
    with st.spinner("🔄 Chargement des modèles YOLOv8..."):
        model_coco = YOLO('yolov8n.pt')    # 80 classes COCO
        model_signs = YOLO('best.pt')      # modèle traffic‑sign

        model_coco.to('cpu')
        model_signs.to('cpu')

        SIGN_OFFSET = len(model_coco.names)   # =80
        names = model_coco.names.copy()
        names.update({k + SIGN_OFFSET: v for k, v in model_signs.names.items()})
        st.success("✅ Modèles chargés avec succès")

    # Paramètres globaux
    cv2.setNumThreads(6)
    
    # Initialiser la source vidéo
    is_video_file = camera_id is None
    
    if is_video_file:
        st.info(f"📹 Ouverture du fichier vidéo")
        cap = cv2.VideoCapture(source_path)
        source_name = Path(source_path).name
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        try:
            st.info(f"📷 Connexion à la webcam: {camera_id}")
            
            # Configuration spécifique pour macOS
            if platform.system() == 'Darwin':
                st.info("Détection de macOS - Configuration spéciale pour la caméra...")
                cap = configure_mac_camera(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id)
                
            source_name = f"Webcam-{camera_id}"
            
            # Vérifier si la caméra est ouverte
            if not cap.isOpened():
                st.error(f"❌ Impossible d'ouvrir la caméra {camera_id}")
                return
            
            # Tester la lecture d'une frame
            ret, test_frame = cap.read()
            if not ret:
                st.error(f"❌ La caméra {camera_id} est détectée mais ne peut pas être lue")
                return
            
            st.success("✅ Caméra connectée et fonctionnelle")
            
        except ValueError:
            st.error(f"❌ ID de caméra invalide: {camera_id}")
            return
    
    if not cap.isOpened():
        st.error("❌ Erreur à l'ouverture de la source vidéo")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    st.info(f"📊 Résolution vidéo : {frame_w}x{frame_h}, FPS source : {fps_src}")
    
    # Paramètres de détection
    conf_thr = 0.25
    iou_thr = 0.55
    scale = 0.75
    skip_frames = 2
    res_w, res_h = int(frame_w*scale), int(frame_h*scale)
    relevant_coco = [0,1,2,3,5,7,9,11]
    
    # Créer un placeholder pour l'affichage de la vidéo
    video_placeholder = st.empty()
    
    # Variables pour le calcul du FPS
    prev_time, fps_hist, frame_count = 0, [], 0
    last_xyxy, last_scores, last_labels = None, None, None
    
    # Bouton pour arrêter la lecture
    stop_button_col, screenshot_col = st.columns(2)
    with stop_button_col:
        stop_button = st.button("Arrêter la lecture")
    
    with screenshot_col:
        screenshot_button = st.button("Prendre un screenshot")
    
    screenshot_path = None
    
    # Boucle principale
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                st.info("🔄 Fin de la vidéo")
            else:
                st.error("❌ Erreur de lecture de la caméra")
            break
        
        frame_count += 1
        
        # Calcul du FPS lissé
        now = time.time()
        fps_curr = 1/(now - prev_time) if prev_time else 0
        prev_time = now
        fps_hist.append(fps_curr)
        if len(fps_hist) > 30: fps_hist.pop(0)
        fps_avg = sum(fps_hist) / len(fps_hist)
        
        if frame_count % skip_frames == 0:
            small = cv2.resize(frame, (res_w, res_h))

            # détection COCO
            r_coco = model_coco.predict(
                small, conf=conf_thr, classes=relevant_coco, verbose=False
            )[0]

            # détection traffic‑sign
            r_sign = model_signs.predict(small, conf=conf_thr, verbose=False)[0]
            sign_cls = r_sign.boxes.cls.clone() + SIGN_OFFSET  # clone + offset

            # fusion des boîtes
            xyxy = torch.cat([r_coco.boxes.xyxy, r_sign.boxes.xyxy])
            scores = torch.cat([r_coco.boxes.conf, r_sign.boxes.conf])
            labels = torch.cat([r_coco.boxes.cls, sign_cls])

            keep = torchvision.ops.nms(xyxy, scores, iou_thr)
            xyxy, scores, labels = xyxy[keep], scores[keep], labels[keep]

            # sauvegarde pour frames ignorées
            last_xyxy, last_scores, last_labels = xyxy, scores, labels

            # dessin
            annotated = frame.copy()
            for i in range(len(xyxy)):
                # 1) extraire & convertir
                x1f, y1f, x2f, y2f = xyxy[i].cpu().numpy()
                x1 = int(x1f / scale); y1 = int(y1f / scale)
                x2 = int(x2f / scale); y2 = int(y2f / scale)

                conf = float(scores[i].item())
                cls = int(labels[i].item())

                # 2) dessiner
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                txt = f"{names[cls]} {conf:.2f}"
                cv2.putText(annotated, txt, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            # ré‑affichage des dernières détections
            annotated = frame.copy()
            if last_xyxy is not None:
                for i in range(len(last_xyxy)):
                    x1f, y1f, x2f, y2f = last_xyxy[i].cpu().numpy()
                    x1 = int(x1f / scale); y1 = int(y1f / scale)
                    x2 = int(x2f / scale); y2 = int(y2f / scale)

                    conf = float(last_scores[i].item())
                    cls = int(last_labels[i].item())

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                    txt = f"{names[cls]} {conf:.2f}"
                    cv2.putText(annotated, txt, (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # Afficher le FPS et la source
        cv2.putText(annotated, f"FPS: {fps_avg:.1f}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(annotated, f"Source: {source_name}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # Convertir pour affichage dans Streamlit
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)
        
        # Prendre un screenshot si demandé
        if screenshot_button and screenshot_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, annotated)
            st.success(f"📸 Screenshot sauvegardé: {screenshot_path}")
            screenshot_button = False
    
    # Libération des ressources
    cap.release()
    st.success("✅ Session terminée!")

def main():
    st.set_page_config(
        page_title="Détection Dashcam - YOLOv8 (Par Antoine Avenia)",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("Détection Dashcam - YOLOv8 (Par Antoine Avenia)")
    st.subheader("Détection d'objets et panneaux de signalisation en temps réel")
    
    # Sidebar pour les options
    st.sidebar.title("Options")
    
    # Choix du mode
    mode = st.sidebar.radio("Choisir le mode d'entrée:", ["Fichier vidéo", "Webcam"])
    
    if mode == "Fichier vidéo":
        st.info("📹 Mode fichier vidéo sélectionné")
        uploaded_file = st.file_uploader("Déposer un fichier vidéo", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Sauvegarder le fichier temporairement
            temp_file_path = save_uploaded_file(uploaded_file)
            st.success(f"✅ Fichier uploadé: {uploaded_file.name}")
            
            # Bouton pour démarrer le traitement
            if st.button("Démarrer l'analyse"):
                process_video(temp_file_path)
    
    else:  # Mode Webcam
        st.info("📷 Mode webcam sélectionné")
        
        # Recherche des caméras disponibles
        with st.spinner("Recherche des caméras disponibles..."):
            available_cameras = get_available_cameras()
        
        if available_cameras:
            st.success(f"✅ {len(available_cameras)} caméras détectées")
            
            # Créer une liste de choix pour les caméras
            camera_options = [f"Caméra {i}: {width}x{height} @ {fps:.1f} FPS" 
                             for i, width, height, fps in available_cameras]
            
            selected_camera = st.selectbox("Sélectionner une caméra:", camera_options)
            
            # Extraire l'ID de la caméra sélectionnée
            camera_id = int(selected_camera.split(':')[0].replace('Caméra ', ''))
            
            # Bouton pour démarrer la webcam
            if st.button("Démarrer la webcam"):
                process_video(None, camera_id=camera_id)
        else:
            st.error("❌ Aucune caméra n'a été détectée")
            
            if platform.system() == 'Darwin':
                st.info("Note pour macOS:")
                st.info("• Si vous utilisez une caméra de continuité (iPhone/iPad), " 
                       "assurez-vous que l'appareil est déverrouillé et que "
                       "la fonction est activée dans les Réglages Système.")

if __name__ == "__main__":
    main() 