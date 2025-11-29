import os
import cv2
import random
from pathlib import Path
import albumentations as A


# ---------------------------------------------------------
# YOLO Polygon Label Loader
# ---------------------------------------------------------
def load_yolo_poly(label_path):
    """
    Load YOLO segmentation label:
    cls x1 y1 x2 y2 ...
    returns list of dicts: {class, poly:[(x,y), ...]}
    """
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))

            poly = []
            for i in range(0, len(coords), 2):
                poly.append((coords[i], coords[i+1]))

            labels.append({"class": cls, "poly": poly})
    return labels


# ---------------------------------------------------------
# Save YOLO Polygon Label File
# ---------------------------------------------------------
def save_yolo_poly(out_path, labels):
    with open(out_path, "w") as f:
        for obj in labels:
            cls = obj["class"]
            poly = obj["poly"]

            line = str(cls)
            for (x,y) in poly:
                line += f" {x:.6f} {y:.6f}"
            f.write(line + "\n")


# ---------------------------------------------------------
# Convert YOLO-normalized polygon → absolute pixels
# ---------------------------------------------------------
def poly_norm_to_abs(poly, w, h):
    return [(x * w, y * h) for (x, y) in poly]


# ---------------------------------------------------------
# Convert absolute polygon → YOLO normalized
# ---------------------------------------------------------
def poly_abs_to_norm(poly, w, h):
    new_poly = []
    for (x, y) in poly:
        nx = max(0, min(x / w, 1))
        ny = max(0, min(y / h, 1))
        new_poly.append((nx, ny))
    return new_poly


# ---------------------------------------------------------
# Apply a geometric transform to image + polygons (SAFE)
# ---------------------------------------------------------
def apply_geo_aug(img, labels, aug):
    h, w = img.shape[:2]

    # Flatten all polygons into one long keypoint list
    keypoints = []
    poly_map = []   # (cls, start_idx, length)
    idx = 0

    for obj in labels:
        poly_abs = poly_norm_to_abs(obj["poly"], w, h)
        keypoints.extend(poly_abs)
        poly_map.append((obj["class"], idx, len(poly_abs)))
        idx += len(poly_abs)

    # Albumentations needs a Compose with keypoints enabled
    comp = A.Compose(
        [aug],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )

    out = comp(image=img, keypoints=keypoints)
    aug_img = out["image"]
    kp_aug = out["keypoints"]

    h2, w2 = aug_img.shape[:2]

    # reconstruct polygons
    final_labels = []
    for cls, start, length in poly_map:
        pts_abs = kp_aug[start:start+length]
        pts_norm = poly_abs_to_norm(pts_abs, w2, h2)
        final_labels.append({"class": cls, "poly": pts_norm})

    return aug_img, final_labels


# ---------------------------------------------------------
# Apply photo-only augmentation (image only)
# ---------------------------------------------------------
def apply_photo_aug(img, labels, aug):
    comp = A.Compose([aug])
    out = comp(image=img)
    return out["image"], labels   # labels unchanged


# =========================================================
#                 MAIN AUGMENTATION PIPELINE
# =========================================================

def augment_dataset(
    input_dir="dataset", 
    output_dir="augmented_dataset",
    splits=("train", "val", "test"),
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output structure
    for split in splits:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Define augmentations
    geo_augs = [
        ("flip", A.HorizontalFlip(p=1)),
        ("rotate90", A.Rotate(limit=(90,90), p=1)),
        ("rotate180", A.Rotate(limit=(180,180), p=1)),
        ("rotate270", A.Rotate(limit=(270,270), p=1)),
        ("affine", A.Affine(scale=1.0, translate_percent=0.1, rotate=10, shear=5, p=1)),
    ]

    photo_augs = [
        ("bright", A.RandomBrightnessContrast(p=1)),
        ("blur", A.Blur(blur_limit=3, p=1)),
        ("noise", A.GaussNoise(p=1)),
        ("clahe", A.CLAHE(p=1)),
    ]

    for split in splits:
        img_dir = input_dir / split / "images"
        lbl_dir = input_dir / split / "labels"

        all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

        for img_path in all_imgs:
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            labels = load_yolo_poly(lbl_path)

            # -------- Save original into output folder --------
            out_img0 = output_dir / split / "images" / f"{stem}.jpg"
            out_lbl0 = output_dir / split / "labels" / f"{stem}.txt"

            cv2.imwrite(str(out_img0), img)
            save_yolo_poly(out_lbl0, labels)

            # -------- Apply geometric augmentations --------
            for name, aug in geo_augs:
                aug_img, aug_lbl = apply_geo_aug(img, labels, aug)

                out_img = output_dir / split / "images" / f"{stem}_{name}.jpg"
                out_lbl = output_dir / split / "labels" / f"{stem}_{name}.txt"

                cv2.imwrite(str(out_img), aug_img)
                save_yolo_poly(out_lbl, aug_lbl)

            # -------- Apply photo-only augmentations --------
            for name, aug in photo_augs:
                aug_img, aug_lbl = apply_photo_aug(img, labels, aug)

                out_img = output_dir / split / "images" / f"{stem}_{name}.jpg"
                out_lbl = output_dir / split / "labels" / f"{stem}_{name}.txt"

                cv2.imwrite(str(out_img), aug_img)
                save_yolo_poly(out_lbl, aug_lbl)



# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    augment_dataset("./dataset/N_Project_v1-product-detection-1_yolov11", "augmented_dataset")
