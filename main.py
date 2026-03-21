import pymupdf
import os
import numpy as np
import config
from PIL import Image


def crop_to_content(pix, margin=5):
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)

    if pix.n >= 3:
        gray = img[:, :, :3].mean(axis=2)
    else:
        gray = img

    mask = gray < 245
    coords = np.argwhere(mask)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min_new = y_min - margin
    x_min_new = x_min - margin
    y_max_new = y_max + margin
    x_max_new = x_max + margin

    y_min_clamped = max(0, y_min_new)
    x_min_clamped = max(0, x_min_new)
    y_max_clamped = min(pix.height, y_max_new)
    x_max_clamped = min(pix.width, x_max_new)

    cropped = img[y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped]

    pad_top = y_min_clamped - y_min_new
    pad_left = x_min_clamped - x_min_new
    pad_bottom = y_max_new - y_max_clamped
    pad_right = x_max_new - x_max_clamped

    cropped = np.pad(
        cropped,
        ((int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right)), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    return cropped


doc = pymupdf.open(config.INPUT_PDF)

if not os.path.exists(config.IMAGE_FOLDER):
    os.makedirs(config.IMAGE_FOLDER)

output_index = 1

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=300)

    if not page.get_text("text").strip():
        print(f"Skipping blank page {i+1}")
        continue

    cropped = crop_to_content(pix, margin=5)

    img = Image.fromarray(cropped)
    img.save(f"{config.IMAGE_FOLDER}question_{output_index}.png")
    output_index += 1
