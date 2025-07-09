# ============================================
# prepare_traffic_sign_dataset.py
# 🔧 Auteur : Antoine (adapté via ChatGPT)
# 🎯 Description :
#   Prépare le dataset YOLOv8 pour la classe unique 'traffic_sign'
#   à partir du dataset GTSDB (format Kaggle)
# ============================================

import os
import pandas as pd
import cv2

# --------------------------
# 🔧 CONFIGURATION
# --------------------------
# Chemins
csv_path = "archive/Train.csv"
images_root_dir = "archive"   # racine où se trouvent les images Train/...

# Output
output_images_dir = "traffic_sign/images/train"
output_labels_dir = "traffic_sign/labels/train"

# Créer les dossiers output si n'existent pas
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# --------------------------
# 📥 LECTURE DU CSV
# --------------------------
df = pd.read_csv(csv_path)

# --------------------------
# 🔎 FILTRAGE SPEED LIMITS
# --------------------------
# IDs speed limit confirmés (GTSDB)
speed_limit_ids = [0,1,2,3,4,5,7,8]

df_filtered = df[df["ClassId"].isin(speed_limit_ids)]
print(f"Nombre total d'annotations speed limit : {len(df_filtered)}")

# --------------------------
# 🔄 CONVERSION YOLO
# --------------------------
for idx, row in df_filtered.iterrows():
    # Chemin image complet
    img_rel_path = row["Path"]
    img_path = os.path.join(images_root_dir, img_rel_path)

    # Lire dimensions image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[⚠️] Image introuvable : {img_path}")
        continue
    h, w = img.shape[:2]

    # Bounding box en pixels
    x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]

    # Conversion YOLO (normalisée)
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h

    # Classe unique = 0 ('traffic_sign')
    yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    # Copier l'image dans le dossier output
    img_name = os.path.basename(img_rel_path)
    out_img_path = os.path.join(output_images_dir, img_name)
    cv2.imwrite(out_img_path, img)

    # Sauvegarder le label YOLO
    label_name = img_name.replace(".png",".txt")
    out_label_path = os.path.join(output_labels_dir, label_name)
    with open(out_label_path, "w") as f:
        f.write(yolo_line)

print("✅ Conversion terminée – dataset prêt au format YOLO.")
