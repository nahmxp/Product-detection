from ultralytics import YOLO
import cv2
import os
import argparse

# Configurable paths / threshold (defaults)
MODEL_PATH = "./runs/segment/train8/weights/best_saved_model/best_float32.tflite"
IMAGE_PATH = "./WhatsApp Image 2025-11-12 at 13.43.02.jpeg"
OUTPUT_PATH = "./output_inference_annotated.jpg"
# Minimum confidence to display (default 0.75 = 75%)
MIN_CONF = 0.5

def main(model_path: str, image_path: str, out_path: str, min_conf: float, show_all: bool = False):
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model not found: {model_path}")
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Image not found: {image_path}")

	model = YOLO(model_path)

	# Load image with OpenCV (BGR) then convert to RGB for model
	img_bgr = cv2.imread(image_path)
	if img_bgr is None:
		raise RuntimeError(f"Failed to load image: {image_path}")
	img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	# Run inference
	results = model(img)
	r = results[0]

	# Annotate image (work on RGB then convert back to BGR to save)
	annotated = img.copy()

	# Collect all detections (no threshold) for debugging
	all_detections = []
	for box in r.boxes:
		try:
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			conf = float(box.conf[0])
			cls = int(box.cls[0])
		except Exception:
			continue
		name = model.names.get(cls, str(cls)) if hasattr(model, "names") else str(cls)
		all_detections.append((name, conf, (x1, y1, x2, y2)))

	# Print raw detections for diagnosis
	if all_detections:
		print(f"Raw detections (no threshold): {len(all_detections)}")
		# print top 10 by confidence
		for name, conf, (x1, y1, x2, y2) in sorted(all_detections, key=lambda x: x[1], reverse=True)[:20]:
			print(f" - {name}: {conf:.3f} at [{x1},{y1},{x2},{y2}]")
	else:
		print("No raw detections found by the model.")

	# If show_all is requested, annotate and save all detections (regardless of min_conf)
	if show_all:
		ann_all = annotated.copy()
		for name, conf, (x1, y1, x2, y2) in all_detections:
			label = f"{name} {conf:.2f}"
			cv2.rectangle(ann_all, (x1, y1), (x2, y2), (255, 165, 0), 2)  # orange boxes
			cv2.putText(ann_all, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
		all_out = os.path.splitext(out_path)[0] + "_all.jpg"
		cv2.imwrite(all_out, cv2.cvtColor(ann_all, cv2.COLOR_RGB2BGR))
		print(f"✅ Saved all detections image to {all_out}")

	# Now filter by min_conf and annotate
	detections = []
	for name, conf, (x1, y1, x2, y2) in all_detections:
		if conf < min_conf:
			continue
		detections.append((name, conf, (x1, y1, x2, y2)))
		label = f"{name} {conf:.2f}"
		cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

	# Print detections summary for filtered results
	if detections:
		print(f"✅ Detections (conf >= {min_conf:.2f}): {len(detections)}")
		for name, conf, (x1, y1, x2, y2) in detections:
			print(f" - {name}: {conf:.3f} at [{x1},{y1},{x2},{y2}]")
	else:
		print(f"No detections with confidence >= {min_conf:.2f}")

	# Save annotated image for filtered detections
	out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
	cv2.imwrite(out_path, out_bgr)
	print(f"✅ Output image saved to {out_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run TFLite YOLO inference and show confidences")
	parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to TFLite model")
	parser.add_argument("--image", type=str, default=IMAGE_PATH, help="Path to input image")
	parser.add_argument("--out", type=str, default=OUTPUT_PATH, help="Path to save annotated output")
	parser.add_argument("--min-conf", type=float, default=MIN_CONF, help="Minimum confidence to show (0-1)")
	parser.add_argument("--show-all", action="store_true", help="Save an image with all raw detections regardless of confidence")
	args = parser.parse_args()
	main(args.model, args.image, args.out, args.min_conf, args.show_all)