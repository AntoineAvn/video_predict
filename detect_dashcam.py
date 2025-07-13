import cv2
import time
import torch, torchvision
from ultralytics import YOLO
import numpy as np

# ─── 1. Chargement des modèles ─────────────────────────────────
model_coco  = YOLO('yolov8n.pt')    # 80 classes COCO
model_signs = YOLO('best.pt')       # ton modèle traffic‑sign

model_coco.to('cpu')
model_signs.to('cpu')

SIGN_OFFSET = len(model_coco.names)   # =80
names = model_coco.names.copy()
names.update({k + SIGN_OFFSET: v for k, v in model_signs.names.items()})

# ─── 2. Paramètres globaux ─────────────────────────────────────
cv2.setNumThreads(6)
cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Erreur à l'ouverture de la vidéo"); exit()

frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_src  = cap.get(cv2.CAP_PROP_FPS)
print(f"Résolution vidéo : {frame_w}x{frame_h}, FPS source : {fps_src}")

scale       = 0.75
res_w, res_h = int(frame_w*scale), int(frame_h*scale)

conf_thr    = 0.25
iou_thr     = 0.55
skip_frames = 2
relevant_coco = [0,1,2,3,5,7,9,11]

# ─── 3. Boucle principale ──────────────────────────────────────
prev_time, fps_hist, frame_count = 0, [], 0
last_xyxy, last_scores, last_labels = None, None, None
last_annotated_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
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

    # — afficher le FPS
    cv2.putText(annotated, f"FPS: {fps_avg:.1f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Détection Dashcam - YOLOv8 Fusion", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── 4. Libération ──────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
