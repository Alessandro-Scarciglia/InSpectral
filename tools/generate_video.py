"""
This script compiles a sequenced directory of input frames into a standard AVI video.
It utilizes regex filters to parse specific image subsets (e.g., RGB, Depth, or Masks) 
and handles automatic on-the-fly reshaping to preserve dimension consistency across frames.
"""

import os
import re
from pathlib import Path
import cv2

# =======================================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =======================================================================================
CONFIG = {
    "image_dir": "/home/vision/Desktop/Datasets/Landsat8_v4/images/ir/",
    "output_path": "ir_video.avi",
    "fps": 25,
    "series": "",         # Dynamic suffix: "" for RGB, "_m" for mask, "_d" for depth
    "extension": ".png"
}


def images_to_avi(
    image_dir: str,
    output_path: str,
    fps: int = 25,
    series: str = "",
    ext: str = ".png"
) -> None:
    """
    Reads structured image files sequentially and compiles them into an AVI video stream.
    Automatically resizes mismatched frames to match the base template dimensions.
    """
    image_dir = Path(image_dir)

    # -----------------------------------------------------------------------------------
    # REGEX FILTERS CONFIGURATION
    # -----------------------------------------------------------------------------------
    if series == "":
        # Matches pure sequential digits + extension (e.g., "000.png")
        regex = re.compile(rf"^\d+{ext}$")
    else:
        # Matches sequential digits + custom experimental suffix + extension (e.g., "000_d.png")
        regex = re.compile(rf"^\d+{re.escape(series)}{ext}$")

    # Filter and sort tracking names sequentially 
    images = sorted(
        p for p in image_dir.iterdir()
        if regex.match(p.name)
    )

    if not images:
        raise RuntimeError(f"No valid frames found matching criteria for the sequence series: '{series}'")

    # -----------------------------------------------------------------------------------
    # BASE VIDEO INITIALIZATION
    # -----------------------------------------------------------------------------------
    # Load first sample template frame to establish standard video height and width boundaries
    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        raise IOError(f"Could not read the primary anchor template frame: {images[0]}")
        
    height, width, _ = first_frame.shape

    # Define standard XVID codec properties for AVI encoding
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # -----------------------------------------------------------------------------------
    # FRAME PROCESSING AND WRITING PIPELINE
    # -----------------------------------------------------------------------------------
    print(f"[INFO] Compiling {len(images)} frames into the final video output...")
    
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARNING] Skipping corrupted or unreadable path: {img_path}")
            continue

        # Force geometric reshape validation if internal frame dimensions mismatch the anchor template
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        # Write sequential frame straight to disk storage video stream
        writer.write(frame)

    # Close active streaming pipes safely
    writer.release()
    print(f"[SUCCESS] Video successfully created using series '{series}': {output_path}")


if __name__ == "__main__":
    # Execute image sequence conversion pipeline using configured dictionary parameters
    images_to_avi(
        image_dir=CONFIG["image_dir"],
        output_path=CONFIG["output_path"],
        fps=CONFIG["fps"],
        series=CONFIG["series"],
        ext=CONFIG["extension"]
    )