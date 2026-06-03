import pymupdf
import os
import numpy as np
import config
import re
import tempfile
import pytesseract
import shutil
import argparse
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


def add_ocr_layer(page, rect, image_path):
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as tmp:
            with Image.open(image_path) as pil_img:
                ocr_pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                    pil_img,
                    extension="pdf",
                )

            tmp.write(ocr_pdf_bytes)
            tmp.flush()

            ocr_doc = pymupdf.open(tmp.name)

            page.show_pdf_page(rect, ocr_doc, 0)

            ocr_doc.close()
            os.unlink(tmp.name)

    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")


def generate_final_pdf(output_file, auto_spacing=False, max_images_per_page=None):
    doc = pymupdf.open()

    a4_w, a4_h = pymupdf.paper_size("a4")
    usable_height = a4_h - (2 * config.MARGIN)
    scale = 72 / config.DPI

    images = sorted(
        (f for f in os.listdir(config.IMAGE_FOLDER) if f.endswith(".png")),
        key=lambda f: int(re.search(r"\d+", f).group()),
    )

    image_data = []

    for image in images:
        path = os.path.join(config.IMAGE_FOLDER, image)

        with Image.open(path) as im:
            width, height = im.size

        width *= scale
        height *= scale

        max_width = a4_w - (2 * config.MARGIN)

        if width > max_width:
            factor = max_width / width
            width *= factor
            height *= factor

        image_data.append(
            {
                "path": path,
                "width": width,
                "height": height,
            }
        )

    pages = []
    current_page = []
    current_height = 0

    for img in image_data:
        required = img["height"]

        if current_page:
            required += config.SPACING

        height_limit_hit = (
            current_height + required > usable_height
        )

        image_limit_hit = (
            max_images_per_page is not None
            and len(current_page) >= max_images_per_page
        )

        if current_page and (
            height_limit_hit or image_limit_hit
        ):
            pages.append(current_page)

            current_page = [img]
            current_height = img["height"]
        else:
            current_page.append(img)
            current_height += required

    if current_page:
        pages.append(current_page)

    for page_images in pages:
        page = doc.new_page(width=a4_w, height=a4_h)

        if auto_spacing:
            used_height = sum(img["height"] for img in page_images)

            extra_spacing = max(0, usable_height - used_height) / len(page_images)
        else:
            extra_spacing = config.SPACING

        y = config.MARGIN

        for img in page_images:
            rect = pymupdf.Rect(
                config.MARGIN,
                y,
                config.MARGIN + img["width"],
                y + img["height"],
            )

            page.insert_image(
                rect,
                filename=img["path"],
            )

            add_ocr_layer(
                page,
                rect,
                img["path"],
            )

            y += img["height"] + extra_spacing

    doc.save(output_file)
    doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input PDF file")
    parser.add_argument(
        "-o", "--output", help="Output PDF file (overrides default)", default=None
    )
    parser.add_argument(
        "--auto-spacing",
        action="store_true",
        help="Expand images on each page to fill available vertical space",
    )
    parser.add_argument(
        "--max-images-per-page",
        type=int,
        default=None,
        help="Maximum number of images allowed on a page",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output if args.output else config.OUTPUT_PDF

    if not os.path.exists(input_path):
        parser.error(f"File does not exist: {input_path}")

    if not os.path.exists(config.IMAGE_FOLDER):
        os.makedirs(config.IMAGE_FOLDER)

    extract_questions_to_image(input_path)
    generate_final_pdf(
        output_path,
        auto_spacing=args.auto_spacing,
        max_images_per_page=args.max_images_per_page,
    )

    shutil.rmtree(config.IMAGE_FOLDER)
