from ultralytics import YOLO

# # Load the YOLO11 model
# model = YOLO("yolo11n.pt")

# # Export the model to TFLite format
# model.export(format="tflite")  # creates 'yolo11n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("./best_float32.tflite")

# Run inference
results = tflite_model("./image.png")

# Save the output image with predictions
output_path = "./output_inference.jpg"
results[0].save(filename=output_path)
print(f"✅ Output image saved to {output_path}")