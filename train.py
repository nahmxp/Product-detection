from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("./Model/yolo11m-seg.pt")

# Train the model
results = model.train(
    data="./dataset/YOLO/yolov11/data.yaml",   # dataset config
    imgsz=640,                        # image size
    batch=24,                         # you can increase since you have 16GB VRAM
    epochs=10,                       # start with 100; avoid 350 for small dataset
    patience=10,                      # early stopping: stop if no improvement in 20 epochs
    workers=0,                        # safe for 4060 Ti
    device=0,                         # use GPU
    optimizer="AdamW",                # better generalization vs. SGD
    lr0=0.001,                        # initial learning rate
    lrf=0.01,                         # final learning rate fraction
    weight_decay=0.0005,              # helps reduce overfitting
    dropout=0.2,                      # extra regularization
    mosaic=1.0,                       # strong augmentation
    mixup=0.15,                       # combine images to improve generalization
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
    )