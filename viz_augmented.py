#!/usr/bin/env python3
"""Visualize augmented results to verify polygon correctness"""

import cv2
import numpy as np
from pathlib import Path

def visualize_annotations(img_path, label_path, output_path):
    """Visualize YOLO polygon annotations on image"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read {img_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Read labels
    if not Path(label_path).exists():
        print(f"Label file not found: {label_path}")
        return False
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            if len(coords) == 4:
                # Bounding box
                cx, cy, bw, bh = coords
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"cls:{cls}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Polygon
                poly = [(coords[i] * w, coords[i+1] * h) for i in range(0, len(coords), 2)]
                poly_np = np.array(poly, dtype=np.int32)
                cv2.polylines(img, [poly_np], True, (0, 255, 0), 2)
                
                # Draw vertices
                for pt in poly:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)
                
                # Draw centroid
                cx = int(sum([p[0] for p in poly]) / len(poly))
                cy = int(sum([p[1] for p in poly]) / len(poly))
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(img, f"cls:{cls} pts:{len(poly)}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imwrite(str(output_path), img)
    print(f"✓ {output_path.name}")
    return True

def main():
    aug_dir = Path("./dataset/YOLO/yolov11/valid")
    img_dir = aug_dir / "images"
    lbl_dir = aug_dir / "labels"
    
    output_dir = Path("/media/xpert-ai/Documents/NDEV/Product detection/viz_valid_output")
    output_dir.mkdir(exist_ok=True)
    
    img_files = sorted(list(img_dir.glob("*.jpg")))
    
    print(f"Visualizing {len(img_files)} augmented images...")
    for img_file in img_files:
        label_file = lbl_dir / (img_file.stem + ".txt")
        output_file = output_dir / f"viz_{img_file.name}"
        visualize_annotations(img_file, label_file, output_file)
    
    print(f"\n✅ Visualization complete!")
    print(f"Check results in: {output_dir}")
    print("\nFiles to review:")
    for f in sorted(output_dir.glob("*.jpg")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
