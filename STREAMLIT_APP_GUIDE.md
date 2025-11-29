# 🚀 YOLO Training & Conversion Dashboard Guide

## Overview
This Streamlit app provides two main functionalities:
1. **TFLite Conversion** - Simple conversion of trained models to TFLite format
2. **Model Training** - Full training pipeline with visualization

---

## 🔄 Tab 1: TFLite Conversion

### What it does:
- Simple, straightforward model conversion (like `tf_cnv.py`)
- Select any `.pt` model from your trained models
- One-click conversion to TFLite format

### How to use:
1. The app automatically finds all `.pt` files in `./runs/segment/*/weights/`
2. Select the model you want to convert from the dropdown
3. View the model path and size
4. Click **"Convert to TFLite"**
5. Wait for the conversion to complete
6. Find your `.tflite` file in the same directory as the `.pt` file

### Features:
- ✅ Shows model file path
- ✅ Displays model size
- ✅ Shows TFLite output size after conversion
- ✅ Progress indicator
- ✅ Clear success/error messages

### Note:
- Segmentation models (`-seg.pt`) may have conversion issues due to complex architecture
- Detection models (non-seg) convert more reliably to TFLite
- If conversion fails, an ONNX file is usually created successfully as an intermediate format

---

## 🎯 Tab 2: Model Training

### What it does:
- Full YOLO model training pipeline
- Real-time visualization of training metrics
- Automatic result visualization
- Optional TFLite conversion after training

### Configuration (Sidebar):

#### 1️⃣ Select Model
- Choose from pretrained YOLO models in `./Model/` folder
- Available: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x (regular and segmentation)

#### 2️⃣ Select Dataset
- Choose dataset configuration from `./dataset/` folder
- Shows number of classes and class names
- Currently available:
  - `YOLO/yolov11/data.yaml` (4 classes: Fogg products)
  - `N_Project_v1-product-detection-1_yolov11/data.yaml` (1 class: Soap-Box)

#### 3️⃣ Training Parameters

**Basic Settings:**
- Image Size: 320-1280 (default: 640)
- Batch Size: 1-64 (default: 24)
- Epochs: 1-2000 (default: 100)
- Patience: 0-100 (default: 50)
- Device: GPU (0) or CPU

**Advanced Settings:**
- Optimizer: AdamW, SGD, Adam
- Learning rates (initial and final)
- Weight decay
- Dropout
- Workers

**Augmentation:**
- Mosaic
- Mixup
- HSV adjustments (Hue, Saturation, Value)

### Training Workflow:
1. Configure all parameters in the sidebar
2. Click **"Start Training"** button
3. Wait for training to complete (progress shown)
4. View comprehensive results:
   - Final metrics (mAP50, mAP50-95, Precision, Recall)
   - Interactive training curves (Plotly charts)
   - Confusion matrix
   - PR curves (Box and Mask)
   - Training batch visualizations
5. Optionally convert the trained model to TFLite

### Visualizations:
- **Training Curves**: Interactive plots showing loss, precision, recall, mAP over epochs
- **Confusion Matrix**: Visual representation of prediction accuracy
- **PR Curves**: Precision-Recall curves for boxes and masks
- **Training Batches**: Sample images from training process
- **Results CSV**: Full results table with all metrics

---

## 🚀 How to Run

### Method 1: Using the launch script
```bash
./run_app.sh
```

### Method 2: Direct command
```bash
streamlit run app.py
```

### Method 3: With virtual environment
```bash
source venv/bin/activate
streamlit run app.py
```

---

## 📁 File Structure

```
.
├── app.py                      # Main Streamlit application
├── tf_cnv.py                   # Simple conversion script (reference)
├── train.py                    # Training script (reference)
├── Model/                      # Pretrained YOLO models (.pt files)
├── dataset/                    # Dataset configurations (.yaml files)
│   ├── YOLO/yolov11/
│   └── N_Project_v1-product-detection-1_yolov11/
└── runs/segment/              # Training outputs
    └── streamlit_train/       # Latest training run
        └── weights/
            ├── best.pt        # Best model checkpoint
            └── last.pt        # Last model checkpoint
```

---

## 💡 Tips

1. **For TFLite Conversion:**
   - Use detection models (non-seg) for better compatibility
   - Segmentation models have complex architectures that may fail
   - Check the exported directory for ONNX files if TFLite fails

2. **For Training:**
   - Start with fewer epochs (50-100) to test
   - Use patience parameter for early stopping
   - Monitor the training curves for overfitting
   - Batch size depends on your GPU memory (RTX 4060 Ti: 24 is good)

3. **Performance:**
   - Use GPU (device=0) for faster training
   - Adjust workers based on CPU cores (0 is safe default)
   - Higher image size = more memory but better accuracy

---

## 🐛 Troubleshooting

### TFLite Conversion Fails
- **Error**: `'NoneType' object has no attribute 'shape'`
  - **Cause**: Complex model architecture (attention layers, segmentation heads)
  - **Solution**: Use detection models instead of segmentation models

### Training is Slow
- **Check**: Device is set to GPU (0)
- **Check**: Batch size is not too small
- **Check**: Workers count (try 0 or 4)

### Out of Memory
- **Reduce**: Batch size
- **Reduce**: Image size
- **Use**: Smaller model (yolo11n instead of yolo11l)

---

## 📊 Understanding Metrics

- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds 0.5-0.95
- **Precision**: How many detections are correct
- **Recall**: How many ground truth objects are detected
- **Box Loss**: Bounding box regression loss
- **Seg Loss**: Segmentation mask loss (for segmentation models)

---

## 🎨 UI Features

- **Two-tab interface** for clear separation of tasks
- **Sidebar configuration** for training parameters
- **Interactive charts** with Plotly for better exploration
- **Progress indicators** for long-running tasks
- **Color-coded metrics** for quick understanding
- **Expandable sections** to reduce clutter
- **Responsive layout** that adapts to screen size

---

## 📝 Notes

- The app maintains session state for training progress
- You can reset and start new training anytime
- All training outputs are saved in `./runs/segment/streamlit_train/`
- TFLite conversion can be done independently of training
- The app automatically finds and lists all available models and datasets

---

## 🔗 Related Files

- `tf_cnv.py` - Simple conversion reference script
- `train.py` - Training configuration reference
- `requirements.txt` - Python dependencies

---

**Built with**: Streamlit, Ultralytics YOLO, Plotly, TensorFlow

