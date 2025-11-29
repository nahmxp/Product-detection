import os
import shutil
from pathlib import Path
import albumentations as A
import numpy as np
import cv2
from tqdm import tqdm
import yaml
import math

# ---------- Helpers ----------
def parse_yolo_label(line):
    parts = line.strip().split()
    cls = int(parts[0])
    coords = list(map(float, parts[1:]))
    if len(coords) == 4:
        return {"class": cls, "bbox": coords, "poly": None}
    else:
        poly = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        return {"class": cls, "bbox": None, "poly": poly}

def save_yolo_label(path: Path, labels):
    with open(path, "w") as f:
        for lab in labels:
            cls = lab["class"]
            if lab["bbox"] is not None:
                cx, cy, w, h = lab["bbox"]
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            elif lab["poly"] is not None:
                poly_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in lab["poly"]])
                f.write(f"{cls} {poly_str}\n")

def poly_to_keypoints(poly_abs):
    return [(x, y) for x, y in poly_abs]

def reconstruct_polygons_from_keypoints(keypoints, poly_splits):
    polys = []
    for cls, start, length in poly_splits:
        pts = keypoints[start:start+length]
        polys.append((cls, pts))
    return polys

def filter_valid_keypoints(keypoints, img_w, img_h, margin=0):
    """Filter keypoints that are within image bounds"""
    valid_pts = []
    for x, y in keypoints:
        if -margin <= x <= img_w + margin and -margin <= y <= img_h + margin:
            # Clamp to image bounds
            x = max(0, min(x, img_w))
            y = max(0, min(y, img_h))
            valid_pts.append((x, y))
    return valid_pts

def polygon_to_bbox_norm(poly_pts, img_w, img_h):
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return [cx, cy, bw, bh]

def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def low_res(img, **kwargs):
    scale_factor = np.random.uniform(0.3, 0.5)  # 30%-50% of original
    h, w = img.shape[:2]
    new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_up = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img_up

# Albumentations wrapper
low_res_aug = ("low_res", A.Lambda(image=low_res, p=1.0))

# ---------- Main augment function ----------
def augment_dataset(input_dir, output_dir, input_yaml="dataset.yaml"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # --- Define augmentations ---
    geo_augs = [
        ("hflip", A.HorizontalFlip(p=1.0)),
        ("vflip", A.VerticalFlip(p=1.0)),
        ("rot90", A.Rotate(limit=(90, 90), p=1.0)),
        ("rot180", A.Rotate(limit=(180, 180), p=1.0)),
        ("rot270", A.Rotate(limit=(270, 270), p=1.0)),
        ("shear_x15", A.Affine(shear={"x": 15, "y": 0}, p=1.0)),
        ("shear_x-15", A.Affine(shear={"x": -15, "y": 0}, p=1.0)),
        ("shear_y15", A.Affine(shear={"x": 0, "y": 15}, p=1.0)),
        ("shear_y-15", A.Affine(shear={"x": 0, "y": -15}, p=1.0)),
    ]

    # Fine rotation augmentations (every 12 degrees)
    fine_rotations = [
        (f"rot{i}", A.Rotate(limit=(i, i), p=1.0))
        for i in range(12, 360, 12)
    ]

    # Zoom-out augmentations (scale down to smaller)
    zoom_outs = [
        (f"zoom_{scale}", A.Affine(scale=scale / 100.0, p=1.0))
        for scale in range(90, 30, -10)
    ]

    geo_augs = geo_augs + fine_rotations + zoom_outs

    # Photometric transforms (image-only)
    photo_augs = [
        ("brightness_contrast", A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7)),
        ("hsv_shift", A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.7)),
        ("rgb_shift", A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5)),
        ("clahe", A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3)),
        ("gamma", A.RandomGamma(gamma_limit=(80, 120), p=0.5)),
        ("motion_blur", A.MotionBlur(blur_limit=3, p=0.3)),
        ("gauss_noise", A.GaussNoise(p=0.3)),

        # 🌙 Low light simulation (darken + reduce contrast)
        ("low_light", A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4), contrast_limit=(-0.4, -0.2), p=1.0)),

        # ☀️ Overexposed / high brightness
        ("overexposed", A.RandomBrightnessContrast(brightness_limit=(0.4, 0.6), contrast_limit=(0.2, 0.4), p=1.0)),

        # ✨ Reflection reduction (reduce highlights + slight blur)
        ("reflection_reduce", A.Sequential([
            A.MedianBlur(blur_limit=5, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)
        ], p=1.0)),


        # low resolution simulation
        # low resolution simulation (works in all versions)
        ("low_res", A.Lambda(image=low_res, p=1.0)),


    ]

    # --- Process dataset ---
    for split in ["train", "val", "test"]:
        img_dir = input_dir / split / "images"
        lbl_dir = input_dir / split / "labels"
        out_img_dir = output_dir / split / "images"
        out_lbl_dir = output_dir / split / "labels"

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        print(f"[{split}] Found {len(img_files)} images in {img_dir}")

        for img_file in tqdm(img_files, desc=f"Augment {split}"):
            label_file = lbl_dir / (img_file.stem + ".txt")
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Warning: Could not read {img_file}")
                continue
            orig_h, orig_w = image.shape[:2]

            labels = []
            if label_file.exists():
                with open(label_file, "r") as f:
                    for line in f:
                        if line.strip():
                            labels.append(parse_yolo_label(line))

            shutil.copy(str(img_file), str(out_img_dir / img_file.name))
            if label_file.exists():
                shutil.copy(str(label_file), str(out_lbl_dir / label_file.name))

            # Separate bbox and polygon labels
            bbox_labels = []
            poly_labels = []

            for lab in labels:
                if lab["bbox"] is not None:
                    bbox_labels.append(lab)
                elif lab["poly"] is not None:
                    poly_labels.append(lab)

            # 1) Geometric augmentations
            if len(bbox_labels) > 0 or len(poly_labels) > 0:
                for name, aug in geo_augs:
                    # Process bboxes
                    bboxes_pascal = []
                    bboxes_cls = []
                    for lab in bbox_labels:
                        cx, cy, bw, bh = lab["bbox"]
                        x1 = (cx - bw/2) * orig_w
                        y1 = (cy - bh/2) * orig_h
                        x2 = (cx + bw/2) * orig_w
                        y2 = (cy + bh/2) * orig_h
                        bboxes_pascal.append([x1, y1, x2, y2])
                        bboxes_cls.append(lab["class"])

                    # Create transform based on what we have
                    if len(bboxes_pascal) > 0:
                        transform = A.Compose(
                            [aug],
                            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bboxes_cls"], min_visibility=0.1),
                        )
                        transformed = transform(image=image, bboxes=bboxes_pascal, bboxes_cls=bboxes_cls)
                    else:
                        transform = A.Compose([aug])
                        transformed = transform(image=image)

                    aug_img = transformed["image"]
                    new_h, new_w = aug_img.shape[:2]

                    new_labels = []
                    
                    # Add transformed bboxes
                    if len(bboxes_pascal) > 0 and "bboxes" in transformed:
                        for bbox, cls in zip(transformed["bboxes"], transformed["bboxes_cls"]):
                            x1, y1, x2, y2 = bbox
                            cx = ((x1 + x2) / 2) / new_w
                            cy = ((y1 + y2) / 2) / new_h
                            bw = (x2 - x1) / new_w
                            bh = (y2 - y1) / new_h
                            new_labels.append({"class": cls, "bbox": [cx, cy, bw, bh], "poly": None})

                    # Process each polygon separately
                    for poly_lab in poly_labels:
                        abs_poly = [(px * orig_w, py * orig_h) for px, py in poly_lab["poly"]]
                        keypoints = abs_poly
                        keypoints_cls = [poly_lab["class"]] * len(abs_poly)
                        
                        # Transform polygon keypoints
                        poly_transform = A.Compose(
                            [aug],
                            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
                        )
                        poly_transformed = poly_transform(image=image, keypoints=keypoints)
                        
                        # Filter and validate transformed keypoints
                        trans_pts = poly_transformed["keypoints"]
                        valid_pts = filter_valid_keypoints(trans_pts, new_w, new_h, margin=10)
                        
                        if len(valid_pts) >= 3:
                            # Normalize coordinates
                            norm = [(x / new_w, y / new_h) for x, y in valid_pts]
                            new_labels.append({"class": poly_lab["class"], "bbox": None, "poly": norm})
                        elif len(valid_pts) > 0:
                            # Convert degenerate polygon to bbox
                            bbox_norm = polygon_to_bbox_norm(valid_pts, new_w, new_h)
                            new_labels.append({"class": poly_lab["class"], "bbox": bbox_norm, "poly": None})

                    cv2.imwrite(str(out_img_dir / f"{img_file.stem}_{name}.jpg"), aug_img)
                    save_yolo_label(out_lbl_dir / f"{img_file.stem}_{name}.txt", new_labels)

            # 2) Photometric augmentations (preserve original annotations)
            for name, aug in photo_augs:
                transform = A.Compose([aug])
                transformed = transform(image=image)
                aug_img = transformed["image"]
                new_h, new_w = aug_img.shape[:2]

                cv2.imwrite(str(out_img_dir / f"{img_file.stem}_{name}.jpg"), aug_img)

                new_labels = []
                
                # Bbox labels (coordinates unchanged for photometric)
                for lab in bbox_labels:
                    cx, cy, bw, bh = lab["bbox"]
                    new_labels.append({"class": lab["class"], "bbox": [cx, cy, bw, bh], "poly": None})
                
                # Polygon labels (coordinates unchanged for photometric)
                for lab in poly_labels:
                    new_labels.append({"class": lab["class"], "bbox": None, "poly": lab["poly"]})

                save_yolo_label(out_lbl_dir / f"{img_file.stem}_{name}.txt", new_labels)

    # YAML Update
    input_yaml_path = input_dir / input_yaml
    output_yaml_path = output_dir / "data.yaml"
    if input_yaml_path.exists():
        with open(input_yaml_path, "r") as f:
            data_cfg = yaml.safe_load(f)
        data_cfg["train"] = str((output_dir / "train" / "images").resolve())
        data_cfg["val"] = str((output_dir / "val" / "images").resolve())
        data_cfg["test"] = str((output_dir / "test" / "images").resolve())
        with open(output_yaml_path, "w") as f:
            yaml.safe_dump(data_cfg, f, sort_keys=False)
    else:
        print("⚠️ No dataset yaml found in input_dir.")

    print(f"\n✅ Augmentation complete! Augmented dataset saved in {output_dir}")

# ---------- Entry point ----------
if __name__ == "__main__":
    augment_dataset("./dataset/N_Project_v1-product-detection-1_yolov11", "Oggy_Dataset")
