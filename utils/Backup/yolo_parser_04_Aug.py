import os, json, random
from shutil import copyfile

def coco_to_yolo(coco_json_path, raw_img_dir, output_label_dir, class_map):
    """
    1. Reads COCO JSON annotations
    2. For each image, writes a .txt file with one line per bounding box:
       <class_id> <x_center> <y_center> <width> <height>  (all normalized 0–1)
    class_map maps COCO category_id → your YOLO class_id (0,1,2…).
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build lookup of images
    images = {img['id']: img for img in coco['images']}
    # Group annotations by image_id
    anns_by_img = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    os.makedirs(output_label_dir, exist_ok=True)

    for img_id, anns in anns_by_img.items():
        img = images[img_id]
        w, h = img['width'], img['height']
        txt_lines = []
        for ann in anns:
            cid = ann['category_id']
            if cid not in class_map:
                continue
            x,y,box_w,box_h = ann['bbox']
            # convert to YOLO center-format and normalize
            x_c = (x + box_w/2)  / w
            y_c = (y + box_h/2)  / h
            w_n =  box_w        / w
            h_n =  box_h        / h
            cls = class_map[cid]
            txt_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        # write one .txt per image
        img_name = img['file_name']
        base    = os.path.splitext(img_name)[0]
        with open(os.path.join(output_label_dir, base + ".txt"), "w") as out:
            out.write("\n".join(txt_lines))

def split_dataset(raw_img_dir, raw_label_dir, proc_dir, train_ratio=0.8):
    """
    Randomly shuffles all images, splits into train/val,
    and copies both the .jpg/.png and its .txt to the
    corresponding folders under data/processed.
    """
    imgs = [f for f in os.listdir(raw_img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(imgs)
    split = int(len(imgs) * train_ratio)
    groups = {
      "train": imgs[:split],
      "val":   imgs[split:]
    }

    for grp, files in groups.items():
        img_out = os.path.join(proc_dir, grp, "images")
        lbl_out = os.path.join(proc_dir, grp, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fn in files:
            base = os.path.splitext(fn)[0]
            # copy image
            copyfile(
              os.path.join(raw_img_dir, fn),
              os.path.join(img_out, fn)
            )
            # copy label (if it exists)
            lbl_src = os.path.join(raw_label_dir, base + ".txt")
            if os.path.exists(lbl_src):
                copyfile(lbl_src, os.path.join(lbl_out, base + ".txt"))

if __name__ == "__main__":
    # 1. Point to your downloaded COCO JSON and image folder:
    COCO_JSON     = "data/raw/kaggle/train/COCO_mul_train_annos.json"
    RAW_IMG_DIR   = "data/raw/kaggle/img"
    RAW_LABEL_DIR = "data/annotations/kaggle"  # Output for YOLO labels
    PROC_DIR      = "data/processed"

    # 2. Map your COCO categories → YOLO classes (0-indexed):
    CLASS_MAP = {
    1: 0,  # headlamp
    2: 1,  # rear_bumper
    3: 2,  # door
    4: 3,  # hood
    5: 4,  # front_bumper
    }

    # 3. Convert JSON → YOLO txt files:
    coco_to_yolo(COCO_JSON, RAW_IMG_DIR, RAW_LABEL_DIR, CLASS_MAP)

    # 4. Split & copy into train/val:
    split_dataset(RAW_IMG_DIR, RAW_LABEL_DIR, PROC_DIR, train_ratio=0.8)
