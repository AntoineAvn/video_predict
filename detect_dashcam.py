import cv2
import time
import torch, torchvision
from ultralytics import YOLO
import numpy as np
import argparse
from pathlib import Path
import platform
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Détection en temps réel avec YOLOv8 - Support fichier et caméra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

MODE FICHIER VIDÉO:
  python detect_dashcam.py --video video.mp4

MODE TEMPS RÉEL (WEBCAM):
  python detect_dashcam.py --camera 0  # Webcam par défaut
  python detect_dashcam.py --camera 1  # Webcam externe
        """
    )
    
    # Source input
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--video', '-v', help='Chemin vers le fichier vidéo')
    source_group.add_argument('--camera', '-c', help='ID de la caméra (0, 1, etc.)')
    
    # Options additionnelles
    parser.add_argument('--output', '-o', default=None, help='Sauvegarder la vidéo annotée')
    parser.add_argument('--conf', default=0.25, type=float, help='Seuil de confiance (défaut: 0.25)')
    parser.add_argument('--iou', default=0.55, type=float, help='Seuil IoU pour NMS (défaut: 0.55)')
    parser.add_argument('--scale', default=0.75, type=float, help='Échelle de redimensionnement (défaut: 0.75)')
    parser.add_argument('--skip', default=2, type=int, help='Nombre de frames à ignorer (défaut: 2)')
    parser.add_argument('--list-cameras', action='store_true', help='Lister les caméras disponibles')
    
    args = parser.parse_args()
    
    # Lister les caméras disponibles si demandé
    if args.list_cameras:
        list_available_cameras()
        return
    
    # ─── 1. Chargement des modèles ─────────────────────────────────
    print("🔄 Chargement des modèles YOLOv8...")
    model_coco  = YOLO('yolov8n.pt')    # 80 classes COCO
    model_signs = YOLO('best.pt')       # modèle traffic‑sign

    model_coco.to('cpu')
    model_signs.to('cpu')

    SIGN_OFFSET = len(model_coco.names)   # =80
    names = model_coco.names.copy()
    names.update({k + SIGN_OFFSET: v for k, v in model_signs.names.items()})
    print("✅ Modèles chargés avec succès")

    # ─── 2. Paramètres globaux ─────────────────────────────────────
    cv2.setNumThreads(6)
    
    # Initialiser la source vidéo
    is_video_file = False
    total_frames = 0
    
    if args.video:
        print(f"📹 Ouverture du fichier vidéo: {args.video}")
        cap = cv2.VideoCapture(args.video)
        source_name = Path(args.video).name
        is_video_file = True
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        try:
            camera_id = int(args.camera)
            print(f"📷 Connexion à la webcam: {camera_id}")
            
            # Configuration spécifique pour macOS
            if platform.system() == 'Darwin':
                print("🍎 Détection de macOS - Configuration spéciale pour la caméra...")
                cap = configure_mac_camera(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id)
                
            source_name = f"Webcam-{camera_id}"
            
            # Vérifier si la caméra est ouverte
            if not cap.isOpened():
                print(f"❌ Impossible d'ouvrir la caméra {camera_id}")
                print("💡 Essayez avec --list-cameras pour voir les caméras disponibles")
                return
            
            # Tester la lecture d'une frame
            ret, test_frame = cap.read()
            if not ret:
                print(f"❌ La caméra {camera_id} est détectée mais ne peut pas être lue")
                print("💡 Essayez une autre caméra ou vérifiez les permissions")
                return
            
            print("✅ Caméra connectée et fonctionnelle")
            
        except ValueError:
            print(f"❌ ID de caméra invalide: {args.camera}")
            return
    
    if not cap.isOpened():
        print("❌ Erreur à l'ouverture de la source vidéo")
        return

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src  = cap.get(cv2.CAP_PROP_FPS)
    print(f"📊 Résolution vidéo : {frame_w}x{frame_h}, FPS source : {fps_src}")
    
    if is_video_file:
        print(f"📊 Nombre total de frames : {total_frames}")

    # Initialiser l'enregistreur vidéo si demandé
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = args.output
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps_src,
            (frame_w, frame_h)
        )
        print(f"💾 Enregistrement vidéo activé: {output_path}")

    # Paramètres de détection
    scale       = args.scale
    res_w, res_h = int(frame_w*scale), int(frame_h*scale)

    conf_thr    = args.conf
    iou_thr     = args.iou
    skip_frames = args.skip
    relevant_coco = [0,1,2,3,5,7,9,11]

    # ─── 3. Initialisation de l'interface ─────────────────────────────
    window_name = "Détection Dashcam - YOLOv8 Fusion"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Créer une barre de progression pour la vidéo (seulement pour les fichiers vidéo)
    if is_video_file:
        cv2.createTrackbar('Position', window_name, 0, max(1, total_frames-1), lambda x: None)
        
    # Variables de contrôle
    paused = False
    slider_was_used = False
    current_pos = 0
    
    # ─── 4. Boucle principale ──────────────────────────────────────
    print("\n🚀 Démarrage de la détection en temps réel...")
    print("💡 Contrôles:")
    print("  • Q: Quitter")
    print("  • ESPACE: Pause/Reprendre")
    print("  • S: Screenshot")
    if is_video_file:
        print("  • ←/→: Reculer/Avancer de 10 secondes")
        print("  • Curseur: Navigation dans la vidéo")
    
    prev_time, fps_hist, frame_count = 0, [], 0
    last_xyxy, last_scores, last_labels = None, None, None
    last_annotated_frame = None
    consecutive_errors = 0
    max_errors = 5

    while True:
        # Vérifier si la position du curseur a été modifiée (pour les fichiers vidéo)
        if is_video_file:
            new_pos = cv2.getTrackbarPos('Position', window_name)
            if new_pos != current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                current_pos = new_pos
                slider_was_used = True
        
        # Si en pause, ne pas lire de nouvelle frame sauf si le curseur a été utilisé
        if paused and not slider_was_used:
            # Afficher la dernière frame annotée avec indication de pause
            if last_annotated_frame is not None:
                pause_frame = last_annotated_frame.copy()
                # Ajouter un indicateur de pause
                cv2.putText(pause_frame, "PAUSE", (frame_w - 120, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window_name, pause_frame)
            
            # Attendre les commandes clavier
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Espace pour reprendre
                paused = False
                print("▶️ Lecture")
            elif key == ord('s'):  # Screenshot
                save_screenshot(last_annotated_frame if last_annotated_frame is not None else frame)
            elif key == 81 and is_video_file:  # Flèche gauche (reculer)
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                new_frame = max(0, current_frame - int(fps_src * 10))  # Reculer de 10 secondes
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                current_pos = new_frame
                cv2.setTrackbarPos('Position', window_name, current_pos)
                slider_was_used = True
            elif key == 83 and is_video_file:  # Flèche droite (avancer)
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                new_frame = min(total_frames - 1, current_frame + int(fps_src * 10))  # Avancer de 10 secondes
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                current_pos = new_frame
                cv2.setTrackbarPos('Position', window_name, current_pos)
                slider_was_used = True
                
            continue
        
        # Réinitialiser le flag du curseur
        slider_was_used = False
        
        # Lire une nouvelle frame
        ret, frame = cap.read()
        if not ret:
            consecutive_errors += 1
            print(f"⚠️ Erreur de lecture (tentative {consecutive_errors}/{max_errors})")
            
            if consecutive_errors >= max_errors:
                if is_video_file:  # Si c'est un fichier vidéo, on peut le redémarrer
                    print("🔄 Fin de la vidéo - Redémarrage...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    current_pos = 0
                    cv2.setTrackbarPos('Position', window_name, 0)
                    consecutive_errors = 0
                    continue
                else:  # Si c'est une webcam, on quitte
                    print("❌ Trop d'erreurs de lecture de la caméra")
                    break
            
            # Petite pause avant de réessayer
            time.sleep(0.5)
            continue
        
        # Mettre à jour la position du curseur pour les fichiers vidéo
        if is_video_file:
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos('Position', window_name, current_pos)
        
        # Réinitialiser le compteur d'erreurs si on a réussi à lire une frame
        consecutive_errors = 0
        frame_count += 1

        # — calcul du FPS lissé
        now = time.time()
        fps_curr = 1/(now - prev_time) if prev_time else 0
        prev_time = now
        fps_hist.append(fps_curr)
        if len(fps_hist) > 30: fps_hist.pop(0)
        fps_avg = sum(fps_hist) / len(fps_hist)

        if frame_count % skip_frames == 0:
            small = cv2.resize(frame, (res_w, res_h))

            # → détection COCO
            r_coco = model_coco.predict(
                small, conf=conf_thr, classes=relevant_coco, verbose=False
            )[0]

            # → détection traffic‑sign
            r_sign = model_signs.predict(small, conf=conf_thr, verbose=False)[0]
            sign_cls = r_sign.boxes.cls.clone() + SIGN_OFFSET  # clone + offset

            # → fusion des boîtes
            xyxy   = torch.cat([r_coco.boxes.xyxy,   r_sign.boxes.xyxy])
            scores = torch.cat([r_coco.boxes.conf,   r_sign.boxes.conf])
            labels = torch.cat([r_coco.boxes.cls,    sign_cls])

            keep = torchvision.ops.nms(xyxy, scores, iou_thr)
            xyxy, scores, labels = xyxy[keep], scores[keep], labels[keep]

            # sauvegarde pour frames ignorées
            last_xyxy, last_scores, last_labels = xyxy, scores, labels

            # → dessin
            annotated = frame.copy()
            for i in range(len(xyxy)):
                # 1) extraire & convertir
                x1f, y1f, x2f, y2f = xyxy[i].cpu().numpy()
                x1 = int(x1f / scale); y1 = int(y1f / scale)
                x2 = int(x2f / scale); y2 = int(y2f / scale)

                conf = float(scores[i].item())
                cls  = int(labels[i].item())

                # 2) dessiner
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                txt = f"{names[cls]} {conf:.2f}"
                cv2.putText(annotated, txt, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            last_annotated_frame = annotated.copy()

        else:
            # ré‑affichage des dernières détections
            annotated = frame.copy() if last_annotated_frame is None else last_annotated_frame.copy()
            if last_xyxy is not None:
                for i in range(len(last_xyxy)):
                    x1f, y1f, x2f, y2f = last_xyxy[i].cpu().numpy()
                    x1 = int(x1f / scale); y1 = int(y1f / scale)
                    x2 = int(x2f / scale); y2 = int(y2f / scale)

                    conf = float(last_scores[i].item())
                    cls  = int(last_labels[i].item())

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                    txt = f"{names[cls]} {conf:.2f}"
                    cv2.putText(annotated, txt, (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # — afficher le FPS et la source
        cv2.putText(annotated, f"FPS: {fps_avg:.1f}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(annotated, f"Source: {source_name}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
        # Afficher la position actuelle pour les fichiers vidéo
        if is_video_file:
            current_time = current_pos / fps_src
            total_time = total_frames / fps_src
            time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}/{int(total_time//60):02d}:{int(total_time%60):02d}"
            cv2.putText(annotated, time_str, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Enregistrer la vidéo si demandé
        if video_writer:
            video_writer.write(annotated)

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Espace pour pause/reprendre
            paused = not paused
            if paused:
                print("⏸️ Pause")
            else:
                print("▶️ Lecture")
        elif key == ord('s'):  # Screenshot
            save_screenshot(annotated)
        elif key == 81 and is_video_file:  # Flèche gauche (reculer)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(0, current_frame - int(fps_src * 10))  # Reculer de 10 secondes
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            current_pos = new_frame
            cv2.setTrackbarPos('Position', window_name, current_pos)
        elif key == 83 and is_video_file:  # Flèche droite (avancer)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = min(total_frames - 1, current_frame + int(fps_src * 10))  # Avancer de 10 secondes
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            current_pos = new_frame
            cv2.setTrackbarPos('Position', window_name, current_pos)

    # ─── 5. Libération ──────────────────────────────────────────────
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("✅ Session terminée!")

def save_screenshot(frame):
    """Sauvegarde un screenshot avec horodatage"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_path = f"screenshot_{timestamp}.jpg"
    cv2.imwrite(screenshot_path, frame)
    print(f"📸 Screenshot sauvegardé: {screenshot_path}")

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
            print(f"✅ Caméra configurée avec succès: {width}x{height}")
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

def list_available_cameras():
    """Liste les caméras disponibles sur le système"""
    print("\n📷 Recherche des caméras disponibles...")
    
    # Nombre maximum de caméras à tester
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
    
    if available_cameras:
        print("✅ Caméras détectées:")
        for i, width, height, fps in available_cameras:
            print(f"  • Caméra {i}: {width}x{height} @ {fps:.1f} FPS")
        print("\n💡 Utilisez l'option --camera X pour sélectionner une caméra")
    else:
        print("❌ Aucune caméra n'a été détectée")
        print("💡 Vérifiez les connexions et les permissions")
    
    if platform.system() == 'Darwin':
        print("\n🍎 Note pour macOS:")
        print("  • Si vous utilisez une caméra de continuité (iPhone/iPad),")
        print("    assurez-vous que l'appareil est déverrouillé et que")
        print("    la fonction est activée dans les Réglages Système.")

if __name__ == "__main__":
    main()
