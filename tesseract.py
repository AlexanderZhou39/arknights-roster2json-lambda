from pytesseract import pytesseract

def tesseract_ocr(img_pil, config):
    return pytesseract.image_to_string(img_pil, config=config)