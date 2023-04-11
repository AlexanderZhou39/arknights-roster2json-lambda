from PIL import Image, ImageOps
from tesseract import tesseract_ocr
from potclassifier import classify_potential
from modclassifier import classify_module
from settings import (
    NAME_MIN_CROP,
    NAME_MAX_PAD,
    LEVEL_MIN_CROP,
    LEVEL_MAX_CROP,
    LEVEL_WIDTH_CROP,
    GRAY_THRESHOLD,
    BRIGHTNESS_BUMP,
    ACCEPTED_PREPOSITIONS
)
import os

DEBUG = os.environ.get('DEBUG') == 'True'

def resize(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using black background.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB'), offset, resize_width, resize_height

def get_original_rect(offset, resize_w, resize_h, xyxy, width, height):
    w_proportion = width / (resize_w)
    y_proportion = height / (resize_h)
    ori_xmin = (xyxy[0] - offset[0]) * w_proportion
    ori_xmax = (xyxy[2] - offset[0]) * w_proportion
    ori_ymin = (xyxy[1] - offset[1]) * y_proportion
    ori_ymax = (xyxy[3] - offset[1]) * y_proportion
    return (
        round(ori_xmin),
        round(ori_ymin),
        round(ori_xmax),
        round(ori_ymax)
    )


def get_class_or_none(df, xmin, xmax, ymin, ymax):
    filtered = df[
        (df['xcenter'] < xmax) & (df['xcenter'] > xmin) & 
        (df['ycenter'] < ymax) & (df['ycenter'] > ymin)
    ]
    if len(filtered) > 0:
        return int(filtered.iloc[0]['class'])
    return None

def get_object_or_none(df, xmin, xmax, ymin, ymax):
    filtered = df[
        (df['xcenter'] < xmax) & (df['xcenter'] > xmin) & 
        (df['ycenter'] < ymax) & (df['ycenter'] > ymin)
    ]
    if len(filtered) > 0:
        return filtered.iloc[0]
    return None


def get_potential(df, xmin, xmax, ymin, ymax, img, index):
    potential = get_object_or_none(
        df=df, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )
    if potential is None:
        return 1
    
    cropped = img.crop((
        potential.xmin,
        potential.ymin,
        potential.xmax,
        potential.ymax
    ))
    if DEBUG:
        cropped.save(f'pots/pot_{index}.jpg')

    return classify_potential(cropped)

def get_module(df, xmin, xmax, ymin, ymax, img):
    module = get_object_or_none(
        df=df, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )
    if module is None:
        return None
    
    cropped = img.crop((
        module.xmin,
        module.ymin,
        module.xmax,
        module.ymax
    ))

    return classify_module(cropped)

def get_promotion(df, xmin, xmax, ymin, ymax):
    promotion = get_class_or_none(
        df=df, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )

    if promotion is None:
        return 0
    return promotion + 1

def get_favorite(df, xmin, xmax, ymin, ymax):
    favorite = get_class_or_none(
        df=df, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )

    if favorite is None:
        return False
    return True


def filter_yellow(img_pil):
    width, height = img_pil.size
    image_data = img_pil.load()
    for y in range(height):
        for x in range(width):
            r, g, b = image_data[x, y]
            if g > 100 and r > 100 and b < 130:
                image_data[x, y] = 0, 0, 0
            else:
                image_data[x, y] = r, g, b

def preprocess_text_image(
    xyxy, resize_w, resize_h, offset, image_pil_raw, threshold=GRAY_THRESHOLD
):
    # crop original image
    cropped = image_pil_raw.crop(get_original_rect(
        offset=offset,
        resize_w=resize_w,
        resize_h=resize_h,
        xyxy=xyxy,
        width=image_pil_raw.width,
        height=image_pil_raw.height
    ))
    # convert to rgb
    cropped = cropped.convert('RGB')
    # filter yellow (level bar and favorite)
    filter_yellow(cropped)
    # apply grayscale threshold
    cropped = cropped.convert('L').point(
        lambda p: min(p + BRIGHTNESS_BUMP, 255) if p > threshold else 0
    )
    return cropped

def process_name_prediction(txt):
    # remove unicode
    new_txt = txt.encode('ascii', 'ignore')
    new_txt = new_txt.decode()

    # replace ! with l
    new_txt = new_txt.replace('!', 'l')

    # remove newlines and spaces
    new_txt = new_txt.split()
    # remove noise
    new_txt = [
        word for word in new_txt if (len(word) > 2 or word == 'W' or word in ACCEPTED_PREPOSITIONS)
    ]
    # rejoin
    new_txt = " ".join(new_txt)

    # remove nonalphanumeric characters
    new_txt = "".join([ c for c in new_txt if c.isalpha() or c == " " ])

    return new_txt.strip().lower()

def process_level_prediction(txt):
    new_txt = "".join([ char for char in txt if char.isnumeric() ])
    if len(new_txt) == 0:
        return 1
    return int(new_txt)

def add_margin(pil_img, m_width, color):
    width, height = pil_img.size
    new_width = width + (m_width * 2)
    new_height = height
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (m_width, 0))
    return result

def add_margin_top(pil_img, color):
    width, height = pil_img.size
    new_width = width
    new_height = height + (50 * 2)
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (0, 50))
    return result

def get_operator_name(
    df, xmin, xmax, ymin, ymax, resize_w, resize_h, offset, resize_full_height, image_pil_raw, index
):
    name = get_object_or_none(
        df=df,xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )

    if name is not None:
        # do ocr on model crop
        xyxy = (
            name.xmin,
            name.ymin,
            name.xmax,
            name.ymax
        )
        processed_image = preprocess_text_image(
            xyxy=xyxy,
            resize_w=resize_w,
            resize_h=resize_h,
            offset=offset,
            image_pil_raw=image_pil_raw
        )
    else:
        # attempt dumb crop ocr
        card_height = ymax - ymin
        xyxy = (
            xmin, 
            ymin + (card_height * NAME_MIN_CROP), 
            xmax, 
            min(ymax + (card_height * NAME_MAX_PAD), resize_full_height)
        )
        
        processed_image = preprocess_text_image(
            xyxy=xyxy,
            resize_w=resize_w,
            resize_h=resize_h,
            offset=offset,
            image_pil_raw=image_pil_raw
        )
    processed_image = add_margin(processed_image, 50, 0)
    if DEBUG:
        processed_image.save(f'results/name_{index}.jpg')
    return processed_image

def get_operator_level(
    df, xmin, xmax, ymin, ymax, resize_w, resize_h, offset, image_pil_raw, index
):
    level = get_object_or_none(
        df=df,xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )
    if level is not None:
        # do ocr on model crop
        xyxy = (
            level.xmin,
            level.ymin,
            level.xmax,
            level.ymax
        )
        processed_image = preprocess_text_image(
            xyxy=xyxy,
            resize_w=resize_w,
            resize_h=resize_h,
            offset=offset,
            image_pil_raw=image_pil_raw,
            threshold=240
        )
    else:
        # attempt dumb crop ocr
        card_height = ymax - ymin
        card_width = xmax - xmin
        xyxy = (
            xmin, 
            ymin + (card_height * LEVEL_MIN_CROP), 
            xmin + (card_width * LEVEL_WIDTH_CROP), 
            ymin + (card_height * LEVEL_MAX_CROP)
        )
        
        processed_image = preprocess_text_image(
            xyxy=xyxy,
            resize_w=resize_w,
            resize_h=resize_h,
            offset=offset,
            image_pil_raw=image_pil_raw,
            threshold=240
        )
    processed_image = add_margin(processed_image, 30, 0)
    if DEBUG:
        processed_image.save(f'results/level_{index}.jpg')
    return processed_image

PADDING = add_margin(Image.open('padding.png'), 50, 0)

def get_concat_v(im1, im2, border=False, spacing=30):
    padding = PADDING
    if border:
        im2 = ImageOps.expand(im2, border=1, fill='white')
        padding = ImageOps.expand(padding, border=1, fill='white')
    dst = Image.new('RGB', (
        max(im1.width, im2.width, padding.width), im1.height + im2.height + padding.height + spacing * 2
    ))
    dst.paste(im1, (0, 0))
    dst.paste(padding, (0, im1.height + spacing))
    dst.paste(im2, (0, im1.height + padding.height + spacing * 2))
    return dst

def get_concat_v_no_padding(im1, im2, spacing=30):
    dst = Image.new('RGB', (
        max(im1.width, im2.width), im1.height + im2.height + spacing
    ))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height + spacing))
    return dst

def name_mosaic(images):
    if len(images) < 2:
        return images[0]
    # mosaic = ImageOps.expand(PADDING, border=1, fill='white')
    mosaic = ImageOps.expand(images[0], border=1, fill='white')
    for i, image in enumerate(images):
        if i == 0:
            continue
        mosaic = get_concat_v(mosaic, image, border=True)
    return mosaic

def level_mosaic(images):
    if len(images) < 2:
        return images[0]
    mosaic = images[0]
    for i, image in enumerate(images):
        if i == 0:
            continue
        mosaic = get_concat_v_no_padding(mosaic, image, spacing=20)
    width, height = mosaic.size
    mosaic = mosaic.resize((round(width * 1.3), round(height * 1.3)))
    mosaic = add_margin_top(mosaic, 0)
    return mosaic
    

def mosaic_to_names(mosaic):
    raw_txt = tesseract_ocr(mosaic, config='--psm 4')

    # remove unicode
    raw_txt = raw_txt.encode('ascii', 'ignore')
    raw_txt = raw_txt.decode()

    # replace and split predictions
    raw_txt = raw_txt.replace('!', 'l')
    raw_txt = raw_txt.split('dumb')
    raw_txt = [ slug for slug in raw_txt if slug != '' ]

    names = []

    for slug in raw_txt:
        new_txt = slug.split()
        # remove noise
        new_txt = [
            word for word in new_txt if (len(word) > 2 or word == 'W' or word.lower() in ACCEPTED_PREPOSITIONS)
        ]
        # rejoin
        new_txt = " ".join(new_txt)

        # remove nonalphanumeric characters
        new_txt = "".join([ c for c in new_txt if c.isalpha() or c == " " ])

        names.append(new_txt.strip().lower())
    
    return names

def mosaic_to_levels(mosaic):
    raw_txt = tesseract_ocr(mosaic, config='--psm 6')
    raw_txt = raw_txt.replace('O', '0')
    raw_txt = raw_txt.replace('S', '5')
    raw_txt = raw_txt.replace('OQ', '0')
    raw_txt = raw_txt.replace('A', '4')
    raw_txt = raw_txt.split()
    raw_txt = [ slug for slug in raw_txt if slug != '' ]

    levels = []
    for slug in raw_txt:
        new_txt = "".join([ char for char in slug if char.isnumeric() ])
        if len(new_txt) == 0:
            levels.append(1)
        else:
            levels.append(max(min(int(new_txt), 90), 1))

    return levels

def name_recognition(images, i = 0):
    mosaic = name_mosaic(images=images)
    if DEBUG:
        mosaic.save(f'results/name_mosaic_{i}.jpg')
    return mosaic_to_names(mosaic=mosaic)

def level_recognition(images, i = 0):
    mosaic = level_mosaic(images=images)
    if DEBUG:
        mosaic.save(f'results/level_mosaic_{i}.jpg')
    return mosaic_to_levels(mosaic=mosaic)