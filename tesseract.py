from pytesseract import pytesseract

pytesseract.tesseract_cmd = './amazonlinux-2/bin/tesseract'

def tesseract_ocr(img_pil, config):
    return pytesseract.image_to_string(img_pil, config=config)