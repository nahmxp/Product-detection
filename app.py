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

# Main App Layout
st.title("🚀 YOLO Training & Conversion Dashboard")
st.markdown("---")

# Create tabs for different sections
tab1, tab2 = st.tabs(["🔄 TFLite Conversion", "🎯 Model Training"])

# ==================== TAB 1: TFLite Conversion ====================
with tab1:
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

# ==================== TAB 2: Model Training ====================
with tab2:

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

