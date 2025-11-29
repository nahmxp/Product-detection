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

# Main App Layout
st.title("🚀 YOLO Training & Conversion Dashboard")
st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📦 COCO to YOLO Converter", "🔄 TFLite Conversion", "🎯 Model Training"])

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

# ==================== TAB 2: TFLite Conversion ====================
with tab2:
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

# ==================== TAB 3: Model Training ====================
with tab3:

    st.header("🎯 Train YOLO Model")
    st.markdown("Configure training parameters and train your model")

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
        epochs = st.number_input("Epochs", min_value=1, max_value=2000, value=100, step=10)
        patience = st.number_input("Patience (Early Stopping)", min_value=0, max_value=100, value=50, step=5)
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

# Main content area
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("Selected Model", selected_model)
with col2:
    st.metric("Dataset", selected_dataset.split('/')[-2] if '/' in selected_dataset else 'Dataset')
with col3:
    st.metric("Epochs", epochs)

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
        'epochs': epochs,
        'patience': patience,
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

