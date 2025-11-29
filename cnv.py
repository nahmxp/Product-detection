import os
import json
import random
import shutil
import zipfile
import tempfile
from pathlib import Path
from tqdm import tqdm
import yaml
from collections import defaultdict
from PIL import Image
import numpy as np


def coco_to_yolo(coco_json, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    with open(coco_json, 'r') as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    category_mapping = {cat_id: idx for idx, cat_id in enumerate(categories.keys())}

    # Create YOLO dirs
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Split dataset
    image_ids = list(images.keys())
    random.shuffle(image_ids)
    n_train = int(len(image_ids) * train_ratio)
    n_val = int(len(image_ids) * val_ratio)
    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:n_train+n_val])
    test_ids = set(image_ids[n_train+n_val:])

    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    for img_id, anns in tqdm(img_to_anns.items(), desc="Converting"):
        img_info = images[img_id]
        file_name = img_info["file_name"]

        # ✅ FIX: get width/height from JSON or from image
        if "width" in img_info and "height" in img_info:
            width, height = img_info["width"], img_info["height"]
        else:
            img_path = Path(coco_json).parent / file_name
            with Image.open(img_path) as im:
                width, height = im.size

        # Decide split
        if img_id in train_ids:
            split = "train"
        elif img_id in val_ids:
            split = "val"
        else:
            split = "test"

        # Copy image
        src_path = Path(coco_json).parent / file_name
        dst_path = Path(output_dir) / "images" / split / Path(file_name).name
        os.makedirs(dst_path.parent, exist_ok=True)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

        # Write YOLO label
        label_path = Path(output_dir) / "labels" / split / (Path(file_name).stem + ".txt")
        with open(label_path, "w") as f:
            for ann in anns:
                cat_id = ann["category_id"]
                class_id = category_mapping[cat_id]

                # Handle polygons if available
                if "segmentation" in ann and isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0:
                    poly = np.array(ann["segmentation"][0]).reshape(-1, 2)
                    poly[:, 0] /= width
                    poly[:, 1] /= height
                    poly_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
                    f.write(f"{class_id} {poly_str}\n")
                else:
                    # Fallback to bbox
                    x, y, w, h = ann["bbox"]
                    cx, cy = (x + w / 2) / width, (y + h / 2) / height
                    nw, nh = w / width, h / height
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # ✅ Create dataset.yaml
    yaml_dict = {
        "train": str(Path(output_dir) / "images/train"),
        "val": str(Path(output_dir) / "images/val"),
        "test": str(Path(output_dir) / "images/test"),
        "names": [categories[k] for k in sorted(categories.keys())]
    }
    with open(Path(output_dir) / "dataset.yaml", "w") as yf:
        yaml.dump(yaml_dict, yf, default_flow_style=False)

    print(f"\n✅ Conversion complete! Dataset saved in {output_dir}")


def unzip_and_convert(zip_path, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📦 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find coco.json inside extracted folder
        coco_json = None
        for root, _, files in os.walk(tmpdir):
            if "coco.json" in files:
                coco_json = os.path.join(root, "coco.json")
                break

        if coco_json is None:
            raise FileNotFoundError("❌ coco.json not found inside ZIP!")

        coco_to_yolo(coco_json, output_dir, train_ratio, val_ratio, test_ratio)


if __name__ == "__main__":
    unzip_and_convert(
        zip_path="./dataset/coco_dataset/coco_dataset.zip",   # your zip
        output_dir="././dataset/yolo_dataset",     # output folder
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
