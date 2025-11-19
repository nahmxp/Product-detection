import os
import shutil
from pathlib import Path
import albumentations as A
import cv2
from tqdm import tqdm
import yaml
import math

# ---------- Helpers ----------
def parse_yolo_label(line):
    parts = line.strip().split()
    cls = int(parts[0])
    coords = list(map(float, parts[1:]))
    # If 4 coords -> bbox (cx cy w h). else polygon (x1 y1 x2 y2 ...)
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
    # keypoints expects flat list of (x,y) points
    return [(x, y) for x, y in poly_abs]

def reconstruct_polygons_from_keypoints(keypoints, poly_splits):
    # poly_splits: list of (class, start_idx, length)
    polys = []
    for cls, start, length in poly_splits:
        pts = keypoints[start:start+length]
        polys.append((cls, pts))
    return polys

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

# ---------- Main augment function ----------
def augment_dataset(input_dir, output_dir, input_yaml="dataset.yaml"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # remove existing output dir to start fresh
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # create train/val/test structure
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # --- Define augmentations ---
    # Geometric transforms (must transform polygons/keypoints)
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
        # You can add scale/translate if needed:
        # ("scale_up", A.Affine(scale=1.1, p=1.0)),
    ]

    # Photometric transforms (image-only)
    photo_augs = [
        ("brightness_contrast", A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7)),
        ("hsv_shift", A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.7)),
        ("rgb_shift", A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5)),
        ("clahe", A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3)),
        ("gamma", A.RandomGamma(gamma_limit=(80, 120), p=0.5)),
        ("motion_blur", A.MotionBlur(blur_limit=3, p=0.3)),
        ("gauss_noise", A.GaussNoise(p=0.3)),
    ]

    # For each split copy originals + augment
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

            # read labels (polygons or bboxes)
            labels = []
            if label_file.exists():
                with open(label_file, "r") as f:
                    for line in f:
                        if line.strip():
                            labels.append(parse_yolo_label(line))

            # save original image and label in output
            shutil.copy(str(img_file), str(out_img_dir / img_file.name))
            if label_file.exists():
                shutil.copy(str(label_file), str(out_lbl_dir / label_file.name))

            # Prepare polygon -> absolute coords, flatten into keypoints
            # We'll store: keypoints list, keypoints_cls, and poly_splits for reconstruction
            keypoints = []
            keypoints_cls = []
            poly_splits = []  # tuples (cls, start_idx, length)
            bboxes_pascal = []
            bboxes_cls = []

            for lab in labels:
                if lab["bbox"] is not None:
                    # bbox is normalized cx,cy,w,h
                    cx, cy, bw, bh = lab["bbox"]
                    x1 = (cx - bw/2) * orig_w
                    y1 = (cy - bh/2) * orig_h
                    x2 = (cx + bw/2) * orig_w
                    y2 = (cy + bh/2) * orig_h
                    bboxes_pascal.append([x1, y1, x2, y2])
                    bboxes_cls.append(lab["class"])
                elif lab["poly"] is not None:
                    abs_poly = [(px * orig_w, py * orig_h) for px, py in lab["poly"]]
                    start = len(keypoints)
                    for pt in abs_poly:
                        keypoints.append(pt)
                        keypoints_cls.append(lab["class"])
                    poly_splits.append((lab["class"], start, len(abs_poly)))

            # ---------------------
            # 1) APPLY GEOMETRIC AUGS (with keypoints/bboxes)
            # ---------------------
            if len(bboxes_pascal) > 0 or len(keypoints) > 0:
                # Compose single-transform wrappers for each geo aug that require keypoint/bbox processing
                for name, aug in geo_augs:
                    transform = A.Compose(
                        [aug],
                        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bboxes_cls"]),
                        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=["keypoints_cls"]),
                    )

                    transformed = transform(
                        image=image,
                        bboxes=bboxes_pascal,
                        bboxes_cls=bboxes_cls,
                        keypoints=keypoints,
                        keypoints_cls=keypoints_cls,
                    )

                    aug_img = transformed["image"]
                    aug_bboxes = transformed.get("bboxes", [])
                    aug_bboxes_cls = transformed.get("bboxes_cls", [])
                    aug_keypoints = transformed.get("keypoints", [])
                    aug_keypoints_cls = transformed.get("keypoints_cls", [])

                    # new dims
                    new_h, new_w = aug_img.shape[:2]

                    # save image
                    aug_img_name = f"{img_file.stem}_{name}.jpg"
                    aug_img_path = out_img_dir / aug_img_name
                    cv2.imwrite(str(aug_img_path), aug_img)

                    # rebuild labels: bboxes
                    new_labels = []
                    for bbox, cls in zip(aug_bboxes, aug_bboxes_cls):
                        x1, y1, x2, y2 = bbox
                        cx = ((x1 + x2) / 2) / new_w
                        cy = ((y1 + y2) / 2) / new_h
                        bw = (x2 - x1) / new_w
                        bh = (y2 - y1) / new_h
                        # clamp to [0,1]
                        cx = min(max(cx, 0.0), 1.0)
                        cy = min(max(cy, 0.0), 1.0)
                        bw = min(max(bw, 0.0), 1.0)
                        bh = min(max(bh, 0.0), 1.0)
                        new_labels.append({"class": cls, "bbox": [cx, cy, bw, bh], "poly": None})

                    # rebuild polygons from keypoints using poly_splits mapping
                    if poly_splits:
                        polys = reconstruct_polygons_from_keypoints(aug_keypoints, poly_splits)
                        for cls, pts in polys:
                            # normalize
                            norm = [(max(min(x / new_w, 1.0), 0.0), max(min(y / new_h, 1.0), 0.0)) for x, y in pts]
                            # If polygon became degenerate (all pts outside), optional: skip
                            if len(norm) >= 3:
                                new_labels.append({"class": cls, "bbox": None, "poly": norm})
                            else:
                                # fallback convert to bbox for stability
                                bbox_norm = polygon_to_bbox_norm([(x*new_w, y*new_h) for x,y in norm], new_w, new_h)
                                new_labels.append({"class": cls, "bbox": bbox_norm, "poly": None})

                    # save label
                    aug_lbl_path = out_lbl_dir / f"{img_file.stem}_{name}.txt"
                    save_yolo_label(aug_lbl_path, new_labels)

            # ---------------------
            # 2) APPLY PHOTOMETRIC (image-only) AUGS
            # ---------------------
            for name, aug in photo_augs:
                transform = A.Compose([aug])
                transformed = transform(image=image)
                aug_img = transformed["image"]
                new_h, new_w = aug_img.shape[:2]

                # Save image
                aug_img_name = f"{img_file.stem}_{name}.jpg"
                aug_img_path = out_img_dir / aug_img_name
                cv2.imwrite(str(aug_img_path), aug_img)

                # For photometric transforms polygons/bboxes unchanged (only pixels change)
                # Recreate labels from original coords but re-normalized to the new size (should be same)
                new_labels = []
                # bboxes
                for bbox, cls in zip(bboxes_pascal, bboxes_cls):
                    x1, y1, x2, y2 = bbox
                    cx = ((x1 + x2) / 2) / new_w
                    cy = ((y1 + y2) / 2) / new_h
                    bw = (x2 - x1) / new_w
                    bh = (y2 - y1) / new_h
                    cx = min(max(cx, 0.0), 1.0)
                    cy = min(max(cy, 0.0), 1.0)
                    bw = min(max(bw, 0.0), 1.0)
                    bh = min(max(bh, 0.0), 1.0)
                    new_labels.append({"class": cls, "bbox": [cx, cy, bw, bh], "poly": None})

                # polygons
                for cls, start, length in poly_splits:
                    pts = keypoints[start:start+length]  # absolute coords on original image
                    norm = [(x / new_w, y / new_h) for x, y in pts]
                    if len(norm) >= 3:
                        new_labels.append({"class": cls, "bbox": None, "poly": norm})
                    else:
                        bbox_norm = polygon_to_bbox_norm(pts, new_w, new_h)
                        new_labels.append({"class": cls, "bbox": bbox_norm, "poly": None})

                aug_lbl_path = out_lbl_dir / f"{img_file.stem}_{name}.txt"
                save_yolo_label(aug_lbl_path, new_labels)

    # Adapt YAML (if exists)
    input_yaml_path = input_dir / input_yaml
    output_yaml_path = output_dir / "data.yaml"

    if input_yaml_path.exists():
        with open(input_yaml_path, "r") as f:
            data_cfg = yaml.safe_load(f)

        # point train/val/test to new image folders
        data_cfg["train"] = str((output_dir / "train" / "images").resolve())
        data_cfg["val"] = str((output_dir / "val" / "images").resolve())
        data_cfg["test"] = str((output_dir / "test" / "images").resolve())

        with open(output_yaml_path, "w") as f:
            yaml.safe_dump(data_cfg, f, sort_keys=False)
    else:
        print("⚠️ No dataset yaml found in input_dir. Please create data.yaml manually or place it at input_dir/{}".format(input_yaml))

    print(f"\n✅ Augmentation complete! Augmented dataset saved in {output_dir}")

# ---------- Entry point ----------
if __name__ == "__main__":
    # Example usage:
    # augment_dataset("Dataset/ROBO", "yolo_dataset_aug")
    augment_dataset("Dataset/ROBO", "yolo_dataset_aug")
