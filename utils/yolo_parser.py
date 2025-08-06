import os
import random
import json
import numpy as np
from shutil import copyfile
from pathlib import Path

def polygon_to_bbox(points, img_width, img_height):
    """Convert polygon points to normalized YOLO bounding box."""
    points_array = np.array(points)
    x_min = np.min(points_array[:, 0])
    y_min = np.min(points_array[:, 1])
    x_max = np.max(points_array[:, 0])
    y_max = np.max(points_array[:, 1])
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def convert_hitl_to_yolo(ann_dir, img_dir, output_label_dir, class_maps):
    """Convert HITL polygon annotations to YOLO format."""
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Combine class maps for parts and damages
    all_classes = {**class_maps['parts'], **class_maps['damages']}
    
    for json_file in os.listdir(ann_dir):
        if json_file.endswith('.json'):
            with open(os.path.join(ann_dir, json_file), 'r') as f:
                ann = json.load(f)

            # --- NEW, extension-agnostic logic ---------------------------------
            stem = Path(json_file).stem          # strip .json            → "Car damages 2.jpg"
            stem = Path(stem).stem               # strip .jpg/.png if any → "Car damages 2"

            # look for any image whose *stem* matches ("Car damages 2")
            img_name = next((f for f in os.listdir(img_dir)
                             if Path(f).stem == stem and f.lower().endswith(('.png', '.jpg', '.jpeg'))),
                            None)
            if img_name is None:
                print(f"⚠  no image found for {json_file}")
                continue
            img_path = os.path.join(img_dir, img_name)
# -------------------------------------------------------------------


            if not os.path.exists(img_path):
                continue
            
            img_width, img_height = ann['size']['width'], ann['size']['height']
            txt_lines = []
            
            for obj in ann['objects']:
                class_id = obj['classId']
                if class_id in all_classes:
                    cls = all_classes[class_id]
                    points = obj['points']['exterior']
                    x_c, y_c, w, h = polygon_to_bbox(points, img_width, img_height)
                    txt_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            print(f"✓ Converted {json_file}  →  {Path(img_name).with_suffix('.txt').name}")
            if txt_lines:
                base = os.path.splitext(img_name)[0]
                with open(os.path.join(output_label_dir, base + ".txt"), "w") as out:
                    out.write("\n".join(txt_lines))

def split_dataset(raw_img_dir, raw_label_dir, proc_dir, train_ratio=0.8):
    """Split dataset into train/val and copy files."""
    imgs = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(imgs)
    split = int(len(imgs) * train_ratio)
    groups = {"train": imgs[:split], "val": imgs[split:]}

    for grp, files in groups.items():
        img_out = os.path.join(proc_dir, grp, "images")
        lbl_out = os.path.join(proc_dir, grp, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fn in files:
            base = os.path.splitext(fn)[0]
            copyfile(os.path.join(raw_img_dir, fn), os.path.join(img_out, fn))
            lbl_src = os.path.join(raw_label_dir, base + ".txt")
            if os.path.exists(lbl_src):
                copyfile(lbl_src, os.path.join(lbl_out, base + ".txt"))

if __name__ == "__main__":
    # Paths for HITL dataset
    ANN_DIR     = "data/raw/hitl/Car damages dataset/File1/ann"
    IMG_DIR     = "data/raw/hitl/Car damages dataset/File1/img"
    OUTPUT_LABEL_DIR = "data/annotations/hitl"
    PROC_DIR    = "data/processed"

    # Class maps based on meta.json and meta.json_1.json
    CLASS_MAP_PARTS = {
        11380316: 0,  # Quarter-panel
        11380317: 1,  # Front-wheel
        11380318: 2,  # Back-window
        11380319: 3,  # Trunk
        11380320: 4,  # Front-door
        11380321: 5,  # Rocker-panel
        11380322: 6,  # Grille
        11380323: 7,  # Windshield
        11380324: 8,  # Front-window
        11380325: 9,  # Back-door
        11380326: 10, # Headlight
        11380327: 11, # Back-wheel
        11380328: 12, # Back-windshield
        11380329: 13, # Hood
        11380330: 14, # Fender
        11380331: 15, # Tail-light
        11380332: 16, # License-plate
        11380333: 17, # Front-bumper
        11380334: 18, # Back-bumper
        11380335: 19, # Mirror
        11380336: 20, # Roof
    }
    CLASS_MAP_DAMAGES = {
        11380051: 21, # Missing part
        11380052: 22, # Broken part
        11380053: 23, # Scratch
        11380054: 24, # Cracked
        11380055: 25, # Dent
        11380056: 26, # Flaking
        11380057: 27, # Paint chip
        11380058: 28, # Corrosion
    }

    # Convert annotations to YOLO format
    convert_hitl_to_yolo(ANN_DIR, IMG_DIR, OUTPUT_LABEL_DIR, {'parts': CLASS_MAP_PARTS, 'damages': CLASS_MAP_DAMAGES})

    # Split dataset into train/val
    split_dataset(IMG_DIR, OUTPUT_LABEL_DIR, PROC_DIR, train_ratio=0.8)