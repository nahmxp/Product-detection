import streamlit as st
import os
import glob
import yaml
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ultralytics import YOLO
from PIL import Image
import json
import random
import shutil
import zipfile
import tempfile
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import cv2
import albumentations as A
import math

# Page configuration
st.set_page_config(
    page_title="YOLO Training Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #00ff00;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'conversion_complete' not in st.session_state:
    st.session_state.conversion_complete = False
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'total_epochs' not in st.session_state:
    st.session_state.total_epochs = 0
if 'best_model_path' not in st.session_state:
    st.session_state.best_model_path = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# Paths
MODEL_DIR = "./Model"
DATASET_DIR = "./dataset"

def get_available_models():
    """Get all .pt files from Model directory"""
    models = glob.glob(os.path.join(MODEL_DIR, "*.pt"))
    return [os.path.basename(m) for m in models]

def get_available_datasets():
    """Get all .yaml files from dataset directory recursively"""
    datasets = glob.glob(os.path.join(DATASET_DIR, "**/*.yaml"), recursive=True)
    # Return relative paths from dataset directory
    return [os.path.relpath(d, DATASET_DIR) for d in datasets]

def load_dataset_info(yaml_path):
    """Load dataset information from YAML file"""
    full_path = os.path.join(DATASET_DIR, yaml_path)
    try:
        with open(full_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        st.error(f"Error loading dataset info: {e}")
        return None

def read_training_results(results_path):
    """Read training results from CSV"""
    try:
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            return df
        return None
    except Exception as e:
        st.error(f"Error reading results: {e}")
        return None

def plot_training_metrics(df):
    """Create interactive plots for training metrics"""
    if df is None or df.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Box Loss', 'Segmentation Loss', 'Precision & Recall', 'mAP Scores'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Box Loss
    if 'train/box_loss' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['train/box_loss'], name='Box Loss (Train)', 
                      line=dict(color='blue')),
            row=1, col=1
        )
    if 'val/box_loss' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val/box_loss'], name='Box Loss (Val)', 
                      line=dict(color='red')),
            row=1, col=1
        )
    
    # Segmentation Loss
    if 'train/seg_loss' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['train/seg_loss'], name='Seg Loss (Train)', 
                      line=dict(color='green')),
            row=1, col=2
        )
    if 'val/seg_loss' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val/seg_loss'], name='Seg Loss (Val)', 
                      line=dict(color='orange')),
            row=1, col=2
        )
    
    # Precision & Recall
    if 'metrics/precision(B)' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['metrics/precision(B)'], name='Precision (Box)', 
                      line=dict(color='purple')),
            row=2, col=1
        )
    if 'metrics/recall(B)' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['metrics/recall(B)'], name='Recall (Box)', 
                      line=dict(color='brown')),
            row=2, col=1, secondary_y=False
        )
    
    # mAP Scores
    if 'metrics/mAP50(B)' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['metrics/mAP50(B)'], name='mAP50 (Box)', 
                      line=dict(color='darkblue')),
            row=2, col=2
        )
    if 'metrics/mAP50-95(B)' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['metrics/mAP50-95(B)'], name='mAP50-95 (Box)', 
                      line=dict(color='darkred')),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, title_text="Training Metrics Dashboard")
    fig.update_xaxes(title_text="Epoch")
    
    return fig

def train_model(model_path, data_yaml, params):
    """Train the YOLO model"""
    try:
        # Load model
        model = YOLO(model_path)
        
        # Update session state
        st.session_state.total_epochs = params['epochs']
        
        # Train the model
        results = model.train(
            data=data_yaml,
            imgsz=params['imgsz'],
            batch=params['batch'],
            epochs=params['epochs'],
            patience=params['patience'],
            workers=params['workers'],
            device=params['device'],
            optimizer=params['optimizer'],
            lr0=params['lr0'],
            lrf=params['lrf'],
            weight_decay=params['weight_decay'],
            dropout=params['dropout'],
            mosaic=params['mosaic'],
            mixup=params['mixup'],
            hsv_h=params['hsv_h'],
            hsv_s=params['hsv_s'],
            hsv_v=params['hsv_v'],
            project='./runs/segment',
            name='streamlit_train',
            exist_ok=True,
            verbose=True
        )
        
        # Get the path to the best model
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        st.session_state.best_model_path = best_model_path
        st.session_state.training_complete = True
        
        return True, best_model_path, results.save_dir
        
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return False, None, None

def get_trained_models():
    """Get all .pt files from runs/segment directory"""
    trained_models = []
    runs_dir = "./runs/segment"
    
    if os.path.exists(runs_dir):
        # Find all weights/best.pt and weights/last.pt files
        for root, dirs, files in os.walk(runs_dir):
            if 'weights' in root:
                for file in files:
                    if file.endswith('.pt'):
                        full_path = os.path.join(root, file)
                        # Create a readable name
                        rel_path = os.path.relpath(full_path, runs_dir)
                        trained_models.append({
                            'display': rel_path,
                            'path': full_path
                        })
    
    return trained_models

def convert_to_tflite_simple(model_path):
    """Simple TFLite conversion - just like tf_cnv.py"""
    try:
        st.info("🔄 Starting TFLite conversion...")
        
        # Load the YOLO model
        model = YOLO(model_path)
        
        # Export the model to TFLite format (simple, like tf_cnv.py)
        export_path = model.export(format="tflite")
        
        return True, export_path
        
    except Exception as e:
        st.error(f"Conversion error: {str(e)}")
        return False, None

def coco_to_yolo_streamlit(coco_json, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, progress_callback=None):
    """
    Convert COCO dataset to YOLO format with folder structure:
    output_dir/
      ├── classes.txt
      ├── data.yaml
      ├── train/
      │   ├── images/
      │   └── labels/
      ├── valid/
      │   ├── images/
      │   └── labels/
      └── test/
          ├── images/
          └── labels/
    """
    try:
        with open(coco_json, 'r') as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        annotations = coco["annotations"]
        categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
        category_mapping = {cat_id: idx for idx, cat_id in enumerate(categories.keys())}

        # Create YOLO dirs with the new structure
        for split in ["train", "valid", "test"]:
            os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

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

        total_images = len(img_to_anns)
        processed = 0

        for img_id, anns in img_to_anns.items():
            img_info = images[img_id]
            file_name = img_info["file_name"]

            # Get width/height from JSON or from image
            if "width" in img_info and "height" in img_info:
                width, height = img_info["width"], img_info["height"]
            else:
                img_path = Path(coco_json).parent / file_name
                with Image.open(img_path) as im:
                    width, height = im.size

            # Decide split (use "valid" instead of "val")
            if img_id in train_ids:
                split = "train"
            elif img_id in val_ids:
                split = "valid"
            else:
                split = "test"

            # Copy image to new structure: split/images/
            src_path = Path(coco_json).parent / file_name
            dst_path = Path(output_dir) / split / "images" / Path(file_name).name
            os.makedirs(dst_path.parent, exist_ok=True)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

            # Write YOLO label to new structure: split/labels/
            label_path = Path(output_dir) / split / "labels" / (Path(file_name).stem + ".txt")
            with open(label_path, "w") as f:
                for ann in anns:
                    cat_id = ann["category_id"]
                    class_id = category_mapping[cat_id]

                    # Handle polygons if available (for segmentation)
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

            processed += 1
            if progress_callback:
                progress_callback(processed / total_images)

        # Create classes.txt
        class_names = [categories[k] for k in sorted(categories.keys())]
        classes_path = Path(output_dir) / "classes.txt"
        with open(classes_path, "w") as cf:
            for class_name in class_names:
                cf.write(f"{class_name}\n")

        # Create data.yaml
        yaml_dict = {
            "path": str(Path(output_dir).absolute()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names
        }
        with open(Path(output_dir) / "data.yaml", "w") as yf:
            yaml.dump(yaml_dict, yf, default_flow_style=False)

        return True, len(train_ids), len(val_ids), len(test_ids), class_names
        
    except Exception as e:
        return False, str(e), 0, 0, []

def unzip_and_convert_streamlit(zip_path, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, progress_callback=None):
    """Unzip COCO dataset and convert to YOLO format"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find coco.json inside extracted folder
        coco_json = None
        for root, _, files in os.walk(tmpdir):
            if "coco.json" in files:
                coco_json = os.path.join(root, "coco.json")
                break
            # Also check for _annotations.coco.json (common in Roboflow exports)
            for file in files:
                if file.endswith(".json") and "coco" in file.lower():
                    coco_json = os.path.join(root, file)
                    break
            if coco_json:
                break

        if coco_json is None:
            return False, "coco.json not found inside ZIP!", 0, 0, []

        return coco_to_yolo_streamlit(coco_json, output_dir, train_ratio, val_ratio, test_ratio, progress_callback)

# ==================== Augmentation Helper Functions ====================
def parse_yolo_label(line):
    """Parse YOLO label line into dict with class, bbox, or poly"""
    parts = line.strip().split()
    cls = int(parts[0])
    coords = list(map(float, parts[1:]))
    if len(coords) == 4:
        return {"class": cls, "bbox": coords, "poly": None}
    else:
        poly = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        return {"class": cls, "bbox": None, "poly": poly}

def save_yolo_label(path: Path, labels):
    """Save YOLO labels to file"""
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
    """Convert polygon to keypoints"""
    return [(x, y) for x, y in poly_abs]

def reconstruct_polygons_from_keypoints(keypoints, poly_splits):
    """Reconstruct polygons from keypoints after augmentation"""
    polys = []
    for cls, start, length in poly_splits:
        pts = keypoints[start:start+length]
        polys.append((cls, pts))
    return polys

def polygon_to_bbox_norm(poly_pts, img_w, img_h):
    """Convert polygon to normalized bounding box"""
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return [cx, cy, bw, bh]

def low_res(img, **kwargs):
    """Low resolution simulation augmentation"""
    scale_factor = np.random.uniform(0.3, 0.5)  # 30%-50% of original
    h, w = img.shape[:2]
    new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_up = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img_up

def augment_dataset_streamlit(input_dir, output_dir, progress_callback=None, status_callback=None):
    """
    Augment YOLO dataset with geometric and photometric transforms.
    Preserves exact calculations from aug.py.
    
    Args:
        input_dir: Path to input YOLO dataset
        output_dir: Path to output augmented dataset
        progress_callback: Function to update progress (0.0 to 1.0)
        status_callback: Function to update status text
    
    Returns:
        success: bool
        stats: dict with augmentation statistics
    """
    try:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if output_dir.exists():
            shutil.rmtree(output_dir)

        # Create output directories
        for split in ["train", "valid", "test"]:
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # --- Define augmentations (exactly as in aug.py) ---
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
            ("low_light", A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4), contrast_limit=(-0.4, -0.2), p=1.0)),
            ("overexposed", A.RandomBrightnessContrast(brightness_limit=(0.4, 0.6), contrast_limit=(0.2, 0.4), p=1.0)),
            ("reflection_reduce", A.Sequential([
                A.MedianBlur(blur_limit=5, p=1.0),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)
            ], p=1.0)),
            ("low_res", A.Lambda(image=low_res, p=1.0)),
        ]

        stats = {
            "original_images": 0,
            "augmented_images": 0,
            "total_augmentations": len(geo_augs) + len(photo_augs),
            "splits": {}
        }

        # Calculate total work for progress
        total_work = 0
        for split in ["train", "valid", "test"]:
            img_dir = input_dir / split / "images"
            if img_dir.exists():
                img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
                total_work += len(img_files)
        
        completed_work = 0

        # --- Process dataset ---
        for split in ["train", "valid", "test"]:
            img_dir = input_dir / split / "images"
            lbl_dir = input_dir / split / "labels"
            out_img_dir = output_dir / split / "images"
            out_lbl_dir = output_dir / split / "labels"

            if not img_dir.exists():
                continue

            img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
            
            split_stats = {
                "original": len(img_files),
                "augmented": 0
            }

            for img_file in img_files:
                if status_callback:
                    status_callback(f"Processing {split}: {img_file.name}")
                
                label_file = lbl_dir / (img_file.stem + ".txt")
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                orig_h, orig_w = image.shape[:2]

                labels = []
                if label_file.exists():
                    with open(label_file, "r") as f:
                        for line in f:
                            if line.strip():
                                labels.append(parse_yolo_label(line))

                # Copy original
                shutil.copy(str(img_file), str(out_img_dir / img_file.name))
                if label_file.exists():
                    shutil.copy(str(label_file), str(out_lbl_dir / label_file.name))

                keypoints, keypoints_cls, poly_splits, bboxes_pascal, bboxes_cls = [], [], [], [], []

                for lab in labels:
                    if lab["bbox"] is not None:
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

                # 1) Geometric augmentations
                if len(bboxes_pascal) > 0 or len(keypoints) > 0:
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
                        new_h, new_w = aug_img.shape[:2]

                        new_labels = []
                        for bbox, cls in zip(transformed["bboxes"], transformed["bboxes_cls"]):
                            x1, y1, x2, y2 = bbox
                            cx = ((x1 + x2) / 2) / new_w
                            cy = ((y1 + y2) / 2) / new_h
                            bw = (x2 - x1) / new_w
                            bh = (y2 - y1) / new_h
                            new_labels.append({"class": cls, "bbox": [cx, cy, bw, bh], "poly": None})

                        if poly_splits:
                            polys = reconstruct_polygons_from_keypoints(transformed["keypoints"], poly_splits)
                            for cls, pts in polys:
                                norm = [(max(min(x / new_w, 1.0), 0.0), max(min(y / new_h, 1.0), 0.0)) for x, y in pts]
                                if len(norm) >= 3:
                                    new_labels.append({"class": cls, "bbox": None, "poly": norm})
                                else:
                                    bbox_norm = polygon_to_bbox_norm([(x*new_w, y*new_h) for x,y in norm], new_w, new_h)
                                    new_labels.append({"class": cls, "bbox": bbox_norm, "poly": None})

                        cv2.imwrite(str(out_img_dir / f"{img_file.stem}_{name}.jpg"), aug_img)
                        save_yolo_label(out_lbl_dir / f"{img_file.stem}_{name}.txt", new_labels)
                        split_stats["augmented"] += 1

                # 2) Photometric augmentations
                for name, aug in photo_augs:
                    transform = A.Compose([aug])
                    transformed = transform(image=image)
                    aug_img = transformed["image"]
                    new_h, new_w = aug_img.shape[:2]

                    cv2.imwrite(str(out_img_dir / f"{img_file.stem}_{name}.jpg"), aug_img)

                    new_labels = []
                    for bbox, cls in zip(bboxes_pascal, bboxes_cls):
                        x1, y1, x2, y2 = bbox
                        cx = ((x1 + x2) / 2) / new_w
                        cy = ((y1 + y2) / 2) / new_h
                        bw = (x2 - x1) / new_w
                        bh = (y2 - y1) / new_h
                        new_labels.append({"class": cls, "bbox": [cx, cy, bw, bh], "poly": None})

                    for cls, start, length in poly_splits:
                        pts = keypoints[start:start+length]
                        norm = [(x / new_w, y / new_h) for x, y in pts]
                        if len(norm) >= 3:
                            new_labels.append({"class": cls, "bbox": None, "poly": norm})
                        else:
                            bbox_norm = polygon_to_bbox_norm(pts, new_w, new_h)
                            new_labels.append({"class": cls, "bbox": bbox_norm, "poly": None})

                    save_yolo_label(out_lbl_dir / f"{img_file.stem}_{name}.txt", new_labels)
                    split_stats["augmented"] += 1

                completed_work += 1
                if progress_callback:
                    progress_callback(completed_work / total_work)
            
            stats["splits"][split] = split_stats
            stats["original_images"] += split_stats["original"]
            stats["augmented_images"] += split_stats["augmented"]

        # Update YAML
        input_yaml_candidates = ["dataset.yaml", "data.yaml"]
        input_yaml_path = None
        for yaml_name in input_yaml_candidates:
            candidate = input_dir / yaml_name
            if candidate.exists():
                input_yaml_path = candidate
                break
        
        output_yaml_path = output_dir / "data.yaml"
        if input_yaml_path:
            with open(input_yaml_path, "r") as f:
                data_cfg = yaml.safe_load(f)
            data_cfg["path"] = str(output_dir.resolve())
            data_cfg["train"] = "train/images"
            data_cfg["val"] = "valid/images"
            data_cfg["test"] = "test/images"
            with open(output_yaml_path, "w") as f:
                yaml.safe_dump(data_cfg, f, sort_keys=False)
            
            # Also create classes.txt if names exist
            if "names" in data_cfg:
                classes_path = output_dir / "classes.txt"
                with open(classes_path, "w") as cf:
                    for name in data_cfg["names"]:
                        cf.write(f"{name}\n")
        
        return True, stats
        
    except Exception as e:
        return False, {"error": str(e)}

def visualize_yolo_annotations(image_path, label_path, class_names=None):
    """
    Visualize YOLO polygon/bbox annotations on an image.
    Returns the annotated image as numpy array for Streamlit display.
    
    Args:
        image_path: path to image file
        label_path: path to YOLO label file
        class_names: list of class names (optional)
    
    Returns:
        annotated_img: numpy array (BGR format)
        stats: dict with annotation statistics
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None, {"error": f"Image not found: {image_path}"}
    
    h, w, _ = img.shape
    
    if not os.path.exists(label_path):
        return img, {"error": f"Label file not found: {label_path}", "objects": 0}
    
    with open(label_path, "r") as f:
        annotations = f.readlines()
    
    stats = {
        "total_objects": len(annotations),
        "classes": {},
        "error": None
    }
    
    # Generate random colors for each class
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) < 3:
            continue
        
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        
        # Track class statistics
        class_name = str(cls_id)
        if class_names and cls_id < len(class_names):
            class_name = class_names[cls_id]
        stats["classes"][class_name] = stats["classes"].get(class_name, 0) + 1
        
        # Choose color
        color = colors[cls_id % len(colors)]
        
        # Check if it's a polygon (more than 4 values) or bbox (4 values)
        if len(coords) > 4:
            # Polygon format
            polygon = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                polygon.append([x, y])
            
            polygon = np.array(polygon, np.int32).reshape((-1, 1, 2))
            
            # Draw filled polygon with transparency
            overlay = img.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            # Draw polygon outline
            cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=2)
            
            # Put class label near first point
            label_pos = tuple(polygon[0][0])
        else:
            # Bounding box format (cx, cy, w, h)
            cx, cy, bw, bh = coords
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)
        
        # Put class label
        cv2.putText(img, class_name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
    
    return img, stats

# Main App Layout
st.title("🚀 YOLO Training & Conversion Dashboard")
st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📦 COCO to YOLO Converter", "🎨 Dataset Augmentation", "🔍 Annotation Checker", "🔄 TFLite Conversion", "🎯 Model Training"])

# ==================== TAB 1: COCO to YOLO Converter ====================
with tab1:
    st.header("📦 Convert COCO Dataset to YOLO Format")
    st.markdown("Upload or select a COCO format dataset (.zip) and convert it to YOLO format with the correct folder structure")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset Selection")
        
        # Option to upload ZIP or select existing
        conversion_option = st.radio(
            "Choose dataset source:",
            ["Upload ZIP file", "Select from dataset folder"],
            key="conversion_option"
        )
        
        zip_file_path = None
        
        if conversion_option == "Upload ZIP file":
            uploaded_file = st.file_uploader(
                "Upload COCO dataset ZIP file",
                type=['zip'],
                help="Upload a ZIP file containing COCO format dataset with coco.json"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_zip_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_zip_path, 'wb') as f:
                    f.write(uploaded_file.read())
                zip_file_path = temp_zip_path
                st.success(f"✅ Uploaded: {uploaded_file.name}")
        else:
            # Find ZIP files in dataset folder
            zip_files = glob.glob(os.path.join(DATASET_DIR, "**/*.zip"), recursive=True)
            
            if zip_files:
                zip_options = [os.path.relpath(z, DATASET_DIR) for z in zip_files]
                selected_zip = st.selectbox(
                    "Select ZIP file:",
                    zip_options,
                    key="existing_zip"
                )
                zip_file_path = os.path.join(DATASET_DIR, selected_zip)
                st.info(f"📁 Selected: {selected_zip}")
            else:
                st.warning("⚠️ No ZIP files found in dataset folder")
        
        # Output directory
        st.subheader("Output Configuration")
        output_name = st.text_input(
            "Output folder name:",
            value="converted_yolo_dataset",
            help="Name for the output YOLO dataset folder"
        )
        
        output_path = os.path.join(DATASET_DIR, "YOLO", output_name)
        st.text(f"Output path: {output_path}")
    
    with col2:
        st.subheader("Split Ratios")
        train_ratio = st.slider("Train ratio", 0.5, 0.9, 0.7, 0.05, key="train_ratio")
        val_ratio = st.slider("Validation ratio", 0.1, 0.3, 0.2, 0.05, key="val_ratio")
        test_ratio = st.slider("Test ratio", 0.05, 0.3, 0.1, 0.05, key="test_ratio")
        
        # Check if ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            st.error(f"⚠️ Ratios must sum to 1.0 (current: {total_ratio:.2f})")
        else:
            st.success(f"✅ Ratios valid (sum: {total_ratio:.2f})")
        
        st.markdown("---")
        st.markdown("### Output Structure")
        st.code("""
output_folder/
├── classes.txt
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
        """, language="text")
    
    st.markdown("---")
    
    # Convert button
    if zip_file_path and abs(total_ratio - 1.0) <= 0.01:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            if st.button("🚀 Convert to YOLO Format", type="primary", use_container_width=True, key="convert_coco_btn"):
                if os.path.exists(output_path):
                    st.warning(f"⚠️ Output folder already exists: {output_path}")
                    if st.button("⚠️ Overwrite existing folder?", key="overwrite_btn"):
                        shutil.rmtree(output_path)
                    else:
                        st.stop()
                
                with st.spinner("🔄 Converting COCO to YOLO format..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(val):
                        progress_bar.progress(val)
                        status_text.text(f"Processing images... {int(val * 100)}%")
                    
                    status_text.text("📂 Extracting ZIP file...")
                    
                    success, *result = unzip_and_convert_streamlit(
                        zip_path=zip_file_path,
                        output_dir=output_path,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    if success:
                        n_train, n_val, n_test, class_names = result
                        st.success("✅ Conversion completed successfully!")
                        
                        # Show statistics
                        st.subheader("📊 Conversion Statistics")
                        
                        stat_cols = st.columns(4)
                        with stat_cols[0]:
                            st.metric("Training Images", n_train)
                        with stat_cols[1]:
                            st.metric("Validation Images", n_val)
                        with stat_cols[2]:
                            st.metric("Test Images", n_test)
                        with stat_cols[3]:
                            st.metric("Total Classes", len(class_names))
                        
                        # Show classes
                        st.subheader("🏷️ Classes")
                        st.write(", ".join(class_names))
                        
                        # Show file paths
                        st.subheader("📁 Generated Files")
                        st.code(f"""
✅ {output_path}/data.yaml
✅ {output_path}/classes.txt
✅ {output_path}/train/images/ ({n_train} images)
✅ {output_path}/train/labels/ ({n_train} labels)
✅ {output_path}/valid/images/ ({n_val} images)
✅ {output_path}/valid/labels/ ({n_val} labels)
✅ {output_path}/test/images/ ({n_test} images)
✅ {output_path}/test/labels/ ({n_test} labels)
                        """, language="text")
                        
                        st.info(f"💡 **data.yaml path**: `{os.path.join(output_path, 'data.yaml')}`")
                        st.markdown("You can now use this dataset for training in the **Model Training** tab!")
                    else:
                        error_msg = result[0]
                        st.error(f"❌ Conversion failed: {error_msg}")
    elif not zip_file_path:
        st.info("📤 Please upload or select a ZIP file to start conversion")
    else:
        st.error("⚠️ Please adjust the split ratios to sum to 1.0")

st.markdown("---")

# ==================== TAB 2: Dataset Augmentation ====================
with tab2:
    st.header("🎨 YOLO Dataset Augmentation")
    st.markdown("Apply geometric and photometric augmentations to expand your YOLO dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Dataset Selection")
        
        # Find all YOLO datasets
        yolo_datasets = []
        for root, dirs, files in os.walk(DATASET_DIR):
            if 'data.yaml' in files or 'dataset.yaml' in files:
                # Check if it has proper structure
                has_structure = False
                for split in ['train', 'valid', 'val', 'test']:
                    if os.path.exists(os.path.join(root, split, 'images')):
                        has_structure = True
                        break
                if has_structure:
                    yolo_datasets.append(root)
        
        if yolo_datasets:
            dataset_options = [os.path.relpath(d, DATASET_DIR) for d in yolo_datasets]
            selected_aug_dataset = st.selectbox(
                "Select Dataset to Augment:",
                dataset_options,
                key="aug_dataset"
            )
            
            full_dataset_path = os.path.join(DATASET_DIR, selected_aug_dataset)
            
            # Load dataset info
            yaml_files = ['data.yaml', 'dataset.yaml']
            dataset_yaml = None
            class_names = None
            
            for yaml_file in yaml_files:
                yaml_path = os.path.join(full_dataset_path, yaml_file)
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        dataset_yaml = yaml.safe_load(f)
                    break
            
            # Show dataset info
            if dataset_yaml:
                if 'names' in dataset_yaml:
                    class_names = dataset_yaml['names']
                    st.info(f"📊 Dataset: {len(class_names)} classes")
                if 'nc' in dataset_yaml:
                    st.info(f"📊 Classes: {dataset_yaml['nc']}")
            
            # Count current images
            st.subheader("📈 Current Dataset Statistics")
            current_stats = {}
            total_images = 0
            
            for split in ['train', 'valid', 'val', 'test']:
                split_dir = os.path.join(full_dataset_path, split, 'images')
                if os.path.exists(split_dir):
                    img_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        img_files.extend(glob.glob(os.path.join(split_dir, ext)))
                    current_stats[split] = len(img_files)
                    total_images += len(img_files)
            
            if current_stats:
                stat_cols = st.columns(len(current_stats))
                for idx, (split, count) in enumerate(current_stats.items()):
                    with stat_cols[idx]:
                        st.metric(split.capitalize(), count)
                st.metric("**Total Original Images**", total_images)
            
            # Output directory
            st.subheader("💾 Output Configuration")
            output_name = st.text_input(
                "Augmented dataset folder name:",
                value=f"{selected_aug_dataset.split('/')[-1]}_augmented",
                help="Name for the augmented dataset folder",
                key="aug_output_name"
            )
            
            output_path = os.path.join(DATASET_DIR, output_name)
            st.text(f"Output path: {output_path}")
            
            if os.path.exists(output_path):
                st.warning("⚠️ Output folder already exists and will be overwritten!")
        else:
            st.warning("⚠️ No YOLO datasets found")
            st.info("💡 Convert a COCO dataset first or ensure your dataset has the proper structure")
    
    with col2:
        st.subheader("🎨 Augmentations Applied")
        
        st.markdown("### Geometric (40+ variants)")
        st.markdown("""
        - ✅ Horizontal/Vertical Flip
        - ✅ Rotations (90°, 180°, 270°)
        - ✅ Fine rotations (every 12°)
        - ✅ Shear transforms (X/Y axis)
        - ✅ Zoom out (90% to 30%)
        """)
        
        st.markdown("### Photometric (11 variants)")
        st.markdown("""
        - ✅ Brightness & Contrast
        - ✅ HSV Shifts
        - ✅ RGB Shifts
        - ✅ CLAHE (histogram equalization)
        - ✅ Gamma correction
        - ✅ Motion blur
        - ✅ Gaussian noise
        - ✅ Low light simulation 🌙
        - ✅ Overexposed simulation ☀️
        - ✅ Reflection reduction ✨
        - ✅ Low resolution simulation 📱
        """)
        
        st.markdown("---")
        st.info("📊 **Total**: ~50+ augmented versions per image")
    
    st.markdown("---")
    
    # Augmentation button
    if yolo_datasets:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            if st.button("🚀 Start Augmentation", type="primary", use_container_width=True, key="aug_start_btn"):
                if os.path.exists(output_path):
                    st.warning(f"⚠️ This will delete and recreate: {output_path}")
                
                with st.spinner("🎨 Augmenting dataset... This may take several minutes."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(val):
                        progress_bar.progress(val)
                    
                    def update_status(text):
                        status_text.text(text)
                    
                    update_status("🔄 Initializing augmentation pipeline...")
                    
                    success, stats = augment_dataset_streamlit(
                        input_dir=full_dataset_path,
                        output_dir=output_path,
                        progress_callback=update_progress,
                        status_callback=update_status
                    )
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    if success:
                        st.success("✅ Augmentation completed successfully!")
                        
                        # Show statistics
                        st.subheader("📊 Augmentation Results")
                        
                        result_cols = st.columns(3)
                        with result_cols[0]:
                            st.metric("Original Images", stats["original_images"])
                        with result_cols[1]:
                            st.metric("Augmented Images", stats["augmented_images"])
                        with result_cols[2]:
                            total = stats["original_images"] + stats["augmented_images"]
                            st.metric("Total Images", total)
                        
                        # Per-split breakdown
                        st.markdown("### 📂 Split Breakdown")
                        if "splits" in stats:
                            for split, split_stats in stats["splits"].items():
                                with st.expander(f"📁 {split.upper()}", expanded=True):
                                    cols = st.columns(3)
                                    with cols[0]:
                                        st.metric("Original", split_stats["original"])
                                    with cols[1]:
                                        st.metric("Augmented", split_stats["augmented"])
                                    with cols[2]:
                                        st.metric("Total", split_stats["original"] + split_stats["augmented"])
                        
                        # Show files created
                        st.markdown("### 📁 Generated Files")
                        st.code(f"""
✅ {output_path}/data.yaml
✅ {output_path}/classes.txt
✅ {output_path}/train/ (images + labels)
✅ {output_path}/valid/ (images + labels)
✅ {output_path}/test/ (images + labels)
                        """, language="text")
                        
                        st.info(f"💡 **data.yaml path**: `{os.path.join(output_path, 'data.yaml')}`")
                        st.markdown("You can now use this augmented dataset for training in the **Model Training** tab!")
                        
                        # Multiplier info
                        if stats["original_images"] > 0:
                            multiplier = (stats["original_images"] + stats["augmented_images"]) / stats["original_images"]
                            st.success(f"🎉 Dataset expanded by **{multiplier:.1f}x** (from {stats['original_images']} to {stats['original_images'] + stats['augmented_images']} images)")
                    else:
                        error_msg = stats.get("error", "Unknown error")
                        st.error(f"❌ Augmentation failed: {error_msg}")
                        st.info("💡 Check that your dataset has the proper YOLO structure with images and labels folders")
    else:
        st.info("📤 Please ensure you have a YOLO dataset in the dataset folder")
        
        st.markdown("### 📂 Expected Dataset Structure:")
        st.code("""
dataset/
└── your_dataset/
    ├── data.yaml (or dataset.yaml)
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/ (or val/)
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
        """, language="text")

st.markdown("---")

# ==================== TAB 3: Annotation Checker ====================
with tab3:
    st.header("🔍 YOLO Annotation Checker")
    st.markdown("Visualize YOLO format annotations (polygons & bounding boxes) on your images")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Dataset Selection")
        
        # Find all YOLO datasets with proper structure
        yolo_datasets = []
        for root, dirs, files in os.walk(DATASET_DIR):
            if 'data.yaml' in files or 'dataset.yaml' in files:
                yolo_datasets.append(root)
        
        if yolo_datasets:
            dataset_options = [os.path.relpath(d, DATASET_DIR) for d in yolo_datasets]
            selected_dataset_path = st.selectbox(
                "Select YOLO Dataset:",
                dataset_options,
                key="anno_dataset"
            )
            
            full_dataset_path = os.path.join(DATASET_DIR, selected_dataset_path)
            
            # Load dataset info
            yaml_files = ['data.yaml', 'dataset.yaml']
            dataset_yaml = None
            class_names = None
            
            for yaml_file in yaml_files:
                yaml_path = os.path.join(full_dataset_path, yaml_file)
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        dataset_yaml = yaml.safe_load(f)
                    break
            
            # Get class names
            if dataset_yaml and 'names' in dataset_yaml:
                class_names = dataset_yaml['names']
                st.success(f"✅ Loaded {len(class_names)} classes")
            else:
                # Try to load from classes.txt
                classes_txt = os.path.join(full_dataset_path, 'classes.txt')
                if os.path.exists(classes_txt):
                    with open(classes_txt, 'r') as f:
                        class_names = [line.strip() for line in f.readlines()]
                    st.success(f"✅ Loaded {len(class_names)} classes from classes.txt")
            
            # Select split
            st.subheader("📂 Split Selection")
            available_splits = []
            for split in ['train', 'valid', 'val', 'test']:
                split_images = os.path.join(full_dataset_path, split, 'images')
                if os.path.exists(split_images):
                    available_splits.append(split)
            
            if available_splits:
                selected_split = st.selectbox(
                    "Select Split:",
                    available_splits,
                    key="anno_split"
                )
                
                # Get all images in the selected split
                images_dir = os.path.join(full_dataset_path, selected_split, 'images')
                labels_dir = os.path.join(full_dataset_path, selected_split, 'labels')
                
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(glob.glob(os.path.join(images_dir, ext)))
                
                if image_files:
                    st.info(f"📸 Found {len(image_files)} images")
                    
                    # Select image
                    st.subheader("🖼️ Image Selection")
                    image_names = [os.path.basename(img) for img in image_files]
                    
                    # Add a search box
                    search_term = st.text_input("🔎 Search images:", "", key="image_search")
                    
                    if search_term:
                        filtered_images = [img for img in image_names if search_term.lower() in img.lower()]
                    else:
                        filtered_images = image_names
                    
                    if filtered_images:
                        selected_image_name = st.selectbox(
                            f"Select Image ({len(filtered_images)} images):",
                            filtered_images,
                            key="selected_image"
                        )
                        
                        selected_image_path = os.path.join(images_dir, selected_image_name)
                        
                        # Automatically find corresponding label
                        image_stem = Path(selected_image_name).stem
                        label_path = os.path.join(labels_dir, f"{image_stem}.txt")
                        
                        # Display annotation button
                        st.markdown("---")
                        if st.button("👁️ View Annotations", type="primary", use_container_width=True):
                            st.session_state.show_annotations = True
                    else:
                        st.warning("No images match your search")
                else:
                    st.warning(f"⚠️ No images found in {images_dir}")
            else:
                st.warning("⚠️ No valid splits found (train/valid/test)")
        else:
            st.warning("⚠️ No YOLO datasets found in dataset folder")
            st.info("💡 Convert a COCO dataset first using the 'COCO to YOLO Converter' tab")
    
    with col2:
        st.subheader("📊 Annotation Visualization")
        
        if 'show_annotations' in st.session_state and st.session_state.show_annotations:
            if 'selected_image_path' in locals() and 'label_path' in locals():
                with st.spinner("🎨 Rendering annotations..."):
                    # Visualize annotations
                    annotated_img, stats = visualize_yolo_annotations(
                        selected_image_path,
                        label_path,
                        class_names
                    )
                    
                    if annotated_img is not None:
                        # Convert BGR to RGB for display
                        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        
                        # Display the annotated image
                        st.image(annotated_img_rgb, use_container_width=True, caption=selected_image_name)
                        
                        # Display statistics
                        if 'error' in stats and stats['error']:
                            st.warning(stats['error'])
                        else:
                            st.success(f"✅ Found {stats['total_objects']} annotated objects")
                            
                            # Show class distribution
                            if stats['classes']:
                                st.markdown("### 📋 Class Distribution")
                                
                                # Create columns for class stats
                                class_cols = st.columns(min(len(stats['classes']), 4))
                                for idx, (cls_name, count) in enumerate(stats['classes'].items()):
                                    with class_cols[idx % len(class_cols)]:
                                        st.metric(cls_name, count)
                                
                                # Create a bar chart
                                import plotly.express as px
                                df_classes = pd.DataFrame(
                                    list(stats['classes'].items()),
                                    columns=['Class', 'Count']
                                )
                                fig = px.bar(df_classes, x='Class', y='Count', 
                                            title='Object Distribution',
                                            color='Count',
                                            color_continuous_scale='Viridis')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Image info
                        st.markdown("---")
                        st.markdown("### 📄 File Information")
                        col_info1, col_info2 = st.columns(2)
                        
                        with col_info1:
                            st.text(f"Image: {selected_image_name}")
                            st.text(f"Label: {os.path.basename(label_path)}")
                        
                        with col_info2:
                            # Get image size
                            img_pil = Image.open(selected_image_path)
                            st.text(f"Resolution: {img_pil.size[0]}x{img_pil.size[1]}")
                            img_size_mb = os.path.getsize(selected_image_path) / (1024 * 1024)
                            st.text(f"Size: {img_size_mb:.2f} MB")
                        
                        # Navigation buttons
                        st.markdown("---")
                        col_nav1, col_nav2, col_nav3 = st.columns(3)
                        
                        with col_nav1:
                            if st.button("⬅️ Previous Image", use_container_width=True):
                                if 'filtered_images' in locals():
                                    current_idx = filtered_images.index(selected_image_name)
                                    if current_idx > 0:
                                        st.session_state.selected_image = filtered_images[current_idx - 1]
                                        st.rerun()
                        
                        with col_nav2:
                            if st.button("🔄 Reload", use_container_width=True):
                                st.rerun()
                        
                        with col_nav3:
                            if st.button("➡️ Next Image", use_container_width=True):
                                if 'filtered_images' in locals():
                                    current_idx = filtered_images.index(selected_image_name)
                                    if current_idx < len(filtered_images) - 1:
                                        st.session_state.selected_image = filtered_images[current_idx + 1]
                                        st.rerun()
                    else:
                        st.error("❌ Failed to load image")
        else:
            # Show instructions
            st.info("👈 Select a dataset, split, and image from the left panel, then click 'View Annotations'")
            
            st.markdown("### 🎯 Features:")
            st.markdown("""
            - ✅ Visualize polygon and bounding box annotations
            - ✅ Automatic label file detection
            - ✅ Color-coded classes
            - ✅ Class distribution statistics
            - ✅ Navigate between images
            - ✅ Search functionality
            - ✅ Support for train/valid/test splits
            """)
            
            # Show example structure
            st.markdown("### 📂 Expected Dataset Structure:")
            st.code("""
dataset/
└── your_dataset/
    ├── data.yaml (or classes.txt)
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
            """, language="text")

st.markdown("---")

# ==================== TAB 4: TFLite Conversion ====================
with tab4:
    st.header("📦 Convert Model to TFLite")
    st.markdown("Select any trained `.pt` model and convert it to TFLite format")
    
    # Get all trained models
    trained_models = get_trained_models()
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        if trained_models:
            model_options = [m['display'] for m in trained_models]
            selected_model_idx = st.selectbox(
                "Select Model to Convert:",
                range(len(model_options)),
                format_func=lambda i: model_options[i],
                key="conversion_model"
            )
            
            selected_model_path = trained_models[selected_model_idx]['path']
            
            # Show model info
            st.text(f"📁 Model Path: {selected_model_path}")
            
            # Get model file size
            if os.path.exists(selected_model_path):
                size_mb = os.path.getsize(selected_model_path) / (1024 * 1024)
                st.text(f"📊 Model Size: {size_mb:.2f} MB")
        else:
            st.warning("⚠️ No trained models found in ./runs/segment/")
            st.info("Train a model first using the Training tab, or place your .pt files in ./runs/segment/*/weights/")
    
    with col_b:
        if trained_models:
            st.markdown("### Quick Info")
            st.markdown("- Simple conversion")
            st.markdown("- Like `tf_cnv.py`")
            st.markdown("- One-click export")
    
    st.markdown("---")
    
    if trained_models:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Convert to TFLite", type="primary", use_container_width=True, key="convert_btn"):
                with st.spinner("Converting to TFLite... Please wait."):
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)
                    
                    success, export_path = convert_to_tflite_simple(selected_model_path)
                    
                    progress_bar.progress(100)
                    
                    if success:
                        st.success(f"✅ Model successfully converted to TFLite!")
                        st.code(f"Export Path: {export_path}", language="text")
                        
                        # Show export file info
                        if os.path.exists(export_path):
                            tflite_size_mb = os.path.getsize(export_path) / (1024 * 1024)
                            st.metric("TFLite Model Size", f"{tflite_size_mb:.2f} MB")
                            
                            # Show the directory
                            export_dir = os.path.dirname(export_path)
                            st.info(f"📁 Find your TFLite model at: `{export_dir}`")
                    else:
                        st.error("❌ Conversion failed. Check the error message above.")
                        st.info("💡 **Tip**: Segmentation models (`-seg.pt`) may have conversion issues. Try using detection models for better TFLite compatibility.")

st.markdown("---")

# ==================== TAB 5: Model Training ====================
with tab5:

    st.header("🎯 Train YOLO Model")
    st.markdown("Configure training parameters and train your model")
    
    # Prominent Epochs and Patience Configuration
    st.markdown("### 🔢 Essential Training Parameters")
    
    col_e1, col_e2, col_e3 = st.columns([2, 2, 3])
    
    with col_e1:
        epochs_main = st.number_input(
            "🔄 Epochs",
            min_value=1,
            max_value=2000,
            value=100,
            step=10,
            help="Number of complete passes through the training dataset",
            key="epochs_main"
        )
        st.caption("💡 More epochs = longer training")
    
    with col_e2:
        patience_main = st.number_input(
            "⏱️ Patience",
            min_value=0,
            max_value=200,
            value=50,
            step=5,
            help="Stop training if no improvement for N epochs",
            key="patience_main"
        )
        st.caption("💡 Prevents overfitting")
    
    with col_e3:
        st.markdown("#### 📊 Training Duration")
        st.info(f"**Max Epochs**: {epochs_main}")
        st.info(f"**Early Stop**: After {patience_main} epochs without improvement")
        estimated_time = epochs_main * 0.5  # Rough estimate: 30 seconds per epoch
        st.caption(f"⏰ Estimated: ~{estimated_time:.0f} minutes (may vary)")
    
    st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Training Configuration")
    
    # Model Selection
    st.subheader("1️⃣ Select Model")
    available_models = get_available_models()
    if not available_models:
        st.error("No models found in Model directory!")
        st.stop()
    
    selected_model = st.selectbox(
        "Choose a pretrained model:",
        available_models,
        help="Select a YOLO model from the Model folder"
    )
    
    # Dataset Selection
    st.subheader("2️⃣ Select Dataset")
    available_datasets = get_available_datasets()
    if not available_datasets:
        st.error("No datasets found in dataset directory!")
        st.stop()
    
    selected_dataset = st.selectbox(
        "Choose a dataset config:",
        available_datasets,
        help="Select a dataset YAML file"
    )
    
    # Show dataset info
    dataset_info = load_dataset_info(selected_dataset)
    if dataset_info:
        st.info(f"📊 Classes: {dataset_info.get('nc', 'N/A')}")
        if 'names' in dataset_info:
            st.write("Classes:", ', '.join(dataset_info['names']))
    
    st.markdown("---")
    
    # Training Parameters
    st.subheader("3️⃣ Training Parameters")
    
    with st.expander("Basic Settings", expanded=True):
        imgsz = st.number_input("Image Size", min_value=320, max_value=1280, value=640, step=32)
        batch = st.number_input("Batch Size", min_value=1, max_value=64, value=24, step=1)
        
        # Note about epochs and patience being in main tab
        st.info("⬆️ **Epochs & Patience** are configured in the main Training tab above")
        
        device = st.selectbox("Device", [0, "cpu"], index=0, help="0 for GPU, cpu for CPU")
    
    with st.expander("Advanced Settings"):
        optimizer = st.selectbox("Optimizer", ["AdamW", "SGD", "Adam"], index=0)
        lr0 = st.number_input("Initial Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        lrf = st.number_input("Final LR Fraction", min_value=0.001, max_value=0.5, value=0.01, format="%.3f")
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.01, value=0.0005, format="%.4f")
        dropout = st.number_input("Dropout", min_value=0.0, max_value=0.5, value=0.2, format="%.2f")
        workers = st.number_input("Workers", min_value=0, max_value=16, value=0, step=1)
    
    with st.expander("Augmentation"):
        mosaic = st.slider("Mosaic", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        mixup = st.slider("Mixup", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        hsv_h = st.slider("HSV-H", min_value=0.0, max_value=0.1, value=0.015, step=0.005)
        hsv_s = st.slider("HSV-S", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        hsv_v = st.slider("HSV-V", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
    
    st.markdown("---")
    
    # Convert to TFLite option
    convert_after_training = st.checkbox("Convert to TFLite after training", value=True)

# Main content area - Model Selection Overview
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    st.metric("Selected Model", selected_model)
with col2:
    st.metric("Dataset", selected_dataset.split('/')[-2] if '/' in selected_dataset else 'Dataset')
with col3:
    st.metric("Epochs", epochs_main)
with col4:
    st.metric("Patience", patience_main)

st.markdown("---")

# Training Control
if not st.session_state.training_started:
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        st.session_state.training_started = True
        st.session_state.training_complete = False
        st.session_state.conversion_complete = False
        st.rerun()

# Training Progress Section
if st.session_state.training_started:
    st.header("📈 Training Progress")
    
    # Prepare training parameters
    params = {
        'imgsz': imgsz,
        'batch': batch,
        'epochs': epochs_main,  # Use value from main tab
        'patience': patience_main,  # Use value from main tab
        'workers': workers,
        'device': device,
        'optimizer': optimizer,
        'lr0': lr0,
        'lrf': lrf,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'mosaic': mosaic,
        'mixup': mixup,
        'hsv_h': hsv_h,
        'hsv_s': hsv_s,
        'hsv_v': hsv_v
    }
    
    model_path = os.path.join(MODEL_DIR, selected_model)
    data_yaml = os.path.join(DATASET_DIR, selected_dataset)
    
    if not st.session_state.training_complete:
        with st.spinner("🔥 Training in progress... This may take a while."):
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            status_placeholder.info("⏳ Initializing training...")
            
            # Train the model
            success, best_model_path, save_dir = train_model(model_path, data_yaml, params)
            
            if success:
                st.success(f"✅ Training completed successfully!")
                st.session_state.training_complete = True
                
                # Read results
                results_csv = os.path.join(save_dir, 'results.csv')
                if os.path.exists(results_csv):
                    st.session_state.results_df = read_training_results(results_csv)
                
                st.rerun()
            else:
                st.error("❌ Training failed. Please check the logs.")
                st.session_state.training_started = False

# Results Visualization
if st.session_state.training_complete:
    st.header("📊 Training Results")
    
    # Get the latest training directory
    latest_train_dir = os.path.join('./runs/segment/streamlit_train')
    
    if os.path.exists(latest_train_dir):
        # Display metrics
        results_csv = os.path.join(latest_train_dir, 'results.csv')
        
        if st.session_state.results_df is not None:
            df = st.session_state.results_df
            
            # Show latest metrics
            st.subheader("Final Metrics")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                if 'metrics/mAP50(B)' in df.columns:
                    st.metric("mAP50 (Box)", f"{df['metrics/mAP50(B)'].iloc[-1]:.4f}")
            with metric_cols[1]:
                if 'metrics/mAP50-95(B)' in df.columns:
                    st.metric("mAP50-95 (Box)", f"{df['metrics/mAP50-95(B)'].iloc[-1]:.4f}")
            with metric_cols[2]:
                if 'metrics/precision(B)' in df.columns:
                    st.metric("Precision", f"{df['metrics/precision(B)'].iloc[-1]:.4f}")
            with metric_cols[3]:
                if 'metrics/recall(B)' in df.columns:
                    st.metric("Recall", f"{df['metrics/recall(B)'].iloc[-1]:.4f}")
            
            # Plot training curves
            st.subheader("Training Curves")
            fig = plot_training_metrics(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show results dataframe
            with st.expander("📋 View Full Results"):
                st.dataframe(df, use_container_width=True)
        
        # Display training images
        st.subheader("Training Visualizations")
        
        viz_cols = st.columns(2)
        
        # Results plot
        results_img = os.path.join(latest_train_dir, 'results.png')
        if os.path.exists(results_img):
            with viz_cols[0]:
                st.image(results_img, caption="Training Results", use_container_width=True)
        
        # Confusion matrix
        confusion_img = os.path.join(latest_train_dir, 'confusion_matrix.png')
        if os.path.exists(confusion_img):
            with viz_cols[1]:
                st.image(confusion_img, caption="Confusion Matrix", use_container_width=True)
        
        # PR Curves
        st.subheader("Precision-Recall Curves")
        pr_cols = st.columns(2)
        
        box_pr = os.path.join(latest_train_dir, 'BoxPR_curve.png')
        if os.path.exists(box_pr):
            with pr_cols[0]:
                st.image(box_pr, caption="Box PR Curve", use_container_width=True)
        
        mask_pr = os.path.join(latest_train_dir, 'MaskPR_curve.png')
        if os.path.exists(mask_pr):
            with pr_cols[1]:
                st.image(mask_pr, caption="Mask PR Curve", use_container_width=True)
        
        # Training batches
        with st.expander("🖼️ View Training Batches"):
            batch_cols = st.columns(3)
            for i, col in enumerate(batch_cols):
                batch_img = os.path.join(latest_train_dir, f'train_batch{i}.jpg')
                if os.path.exists(batch_img):
                    with col:
                        st.image(batch_img, caption=f"Training Batch {i}", use_container_width=True)
    
    # TFLite Conversion Section
    st.markdown("---")
    st.header("🔄 Model Conversion")
    
    if convert_after_training and not st.session_state.conversion_complete:
        if st.button("Convert to TFLite", type="primary", key="train_convert_btn"):
            with st.spinner("Converting to TFLite..."):
                success, export_path = convert_to_tflite_simple(st.session_state.best_model_path)
                
                if success:
                    st.success(f"✅ Model successfully converted to TFLite!")
                    st.info(f"📁 Exported to: {export_path}")
                    st.session_state.conversion_complete = True
                else:
                    st.error("❌ Conversion failed. Try using the 'TFLite Conversion' tab for more options.")
    
    if st.session_state.conversion_complete:
        st.success("✅ TFLite conversion completed!")
        st.info("💡 You can also convert other models using the 'TFLite Conversion' tab.")
    
    # Reset button
    st.markdown("---")
    if st.button("🔄 Start New Training", type="secondary"):
        st.session_state.training_started = False
        st.session_state.training_complete = False
        st.session_state.conversion_complete = False
        st.session_state.best_model_path = None
        st.session_state.results_df = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>YOLO Training Dashboard | Built with Streamlit & Ultralytics</p>
    </div>
    """, 
    unsafe_allow_html=True
)

