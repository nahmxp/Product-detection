import cv2
import os
import numpy as np

def visualize_yolo_polygons(image_path, label_path, output_path=None, class_names=None):
    """
    Visualize YOLO polygon annotations.
    
    image_path: path to image file
    label_path: path to YOLO polygon label file
    output_path: save path (if None, show in a window)
    class_names: list of class names (optional)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w, _ = img.shape

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, "r") as f:
        annotations = f.readlines()

    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) < 3:
            continue

        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # Convert normalized coords -> pixel coords
        polygon = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            polygon.append([x, y])

        polygon = np.array(polygon, np.int32).reshape((-1, 1, 2))

        # Draw polygon
        cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        # Put class label near first point
        label = str(cls_id)
        if class_names and cls_id < len(class_names):
            label = class_names[cls_id]

        cv2.putText(img, label, tuple(polygon[0][0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image at {output_path}")
    else:
        cv2.imshow("YOLO Polygon Annotation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    image_file = "yolo_dataset_aug/images/test/1758522760949-WhatsApp_Image_2025-09-20_at_3.05.49_PM_rot270.jpg"     # your test image
    label_file = "yolo_dataset_aug/labels/test/1758522760949-WhatsApp_Image_2025-09-20_at_3.05.49_PM_rot270.txt"     # polygon labels
    output_file = "output.jpg"

    # optional class names
    class_list = ["person", "car", "dog"]

    visualize_yolo_polygons(image_file, label_file, output_file, class_list)

