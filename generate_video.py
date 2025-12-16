import cv2
from pathlib import Path
import re

def images_to_avi(
    image_dir: str,
    output_path: str,
    fps: int = 25,
    series: str = "",        # "", "_m", "_d"
    ext: str = ".png"
):
    image_dir = Path(image_dir)

    if series == "":
        # accetta solo numeri + estensione (es: 000.png)
        regex = re.compile(rf"^\d+{ext}$")
    else:
        # accetta numeri + suffisso + estensione (es: 000_m.png)
        regex = re.compile(rf"^\d+{re.escape(series)}{ext}$")

    images = sorted(
        p for p in image_dir.iterdir()
        if regex.match(p.name)
    )

    if not images:
        raise RuntimeError(f"Nessuna immagine trovata per la serie '{series}'")

    first = cv2.imread(str(images[0]))
    h, w, _ = first.shape

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for img in images:
        frame = cv2.imread(str(img))
        if frame is None:
            continue

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)

    writer.release()
    print(f"Video creato usando la serie '{series}': {output_path}")


if __name__ == "__main__":
    images_to_avi(
        image_dir="/home/vision/Desktop/Datasets/Landsat8/colour_256_radialboost_20_training",
        output_path="depth_video.avi",
        fps=25,
        series="_d"
    )
