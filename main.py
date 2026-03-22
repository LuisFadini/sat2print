import pymupdf
import os
import numpy as np
import config
import re
import tempfile
import pytesseract
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


def extract_questions_to_image(input_file):
    doc = pymupdf.open(input_file)

    output_index = 1

    for i, page in enumerate(doc):
        if not page.get_text("text").strip():
            print(f"Skipping blank page {i+1}")
            continue

        pix = page.get_pixmap(dpi=config.DPI)

        cropped = crop_to_content(pix, margin=5)

        img = Image.fromarray(cropped)
        img.save(f"{config.IMAGE_FOLDER}question_{output_index}.png")
        output_index += 1
    doc.close()


def generate_final_pdf(output_file):
    doc = pymupdf.open()

    a4_w, a4_h = pymupdf.paper_size("a4")
    SCALE = 72 / config.DPI

    def extract_number(filename):
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else 0

    images = sorted(
        [f for f in os.listdir(config.IMAGE_FOLDER) if f.endswith(".png")],
        key=extract_number,
    )

    page = doc.new_page(width=a4_w, height=a4_h)
    y_cursor = config.MARGIN

    for image in images:
        path = os.path.join(config.IMAGE_FOLDER, image)

        with Image.open(path) as im:
            px_w, px_h = im.size

        width = px_w * SCALE
        height = px_h * SCALE

        max_width = a4_w - 2 * config.MARGIN
        if width > max_width:
            scale = max_width / width
            width *= scale
            height *= scale

        if y_cursor + height > a4_h - config.MARGIN:
            page = doc.new_page(width=a4_w, height=a4_h)
            y_cursor = config.MARGIN

        x = config.MARGIN
        rect = pymupdf.Rect(x, y_cursor, x + width, y_cursor + height)

        page.insert_image(rect, filename=path)

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                ocr_pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                    Image.open(path), extension="pdf"
                )
                tmp.write(ocr_pdf_bytes)
                tmp.flush()

                ocr_doc = pymupdf.open(tmp.name)

                page.show_pdf_page(rect, ocr_doc, 0)

                ocr_doc.close()
                os.unlink(tmp.name)

        except Exception as e:
            print(f"OCR failed for {image}: {e}")

        y_cursor += height + config.SPACING

    doc.save(output_file)
    doc.close()


if __name__ == "__main__":
    if not os.path.exists(config.IMAGE_FOLDER):
        os.makedirs(config.IMAGE_FOLDER)

    extract_questions_to_image(config.INPUT_PDF)
    generate_final_pdf(config.OUTPUT_PDF)
