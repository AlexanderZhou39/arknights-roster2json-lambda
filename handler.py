from PIL import Image
from cgi import FieldStorage
from io import BytesIO
import base64
import json

import logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

from util import (
    resize, 
    get_potential,
    get_module,
    get_promotion,
    get_favorite,
    get_operator_name,
    get_operator_level,
    get_original_rect,
    name_recognition,
    level_recognition
)
from model import detect
from settings import (
    ACCEPTED_EXTENSIONS, 
    PROMOTION_CLASSES,
    POTENTIAL_CLASS,
    OPERATOR_CLASS,
    FAVORITE_CLASS,
    NAME_CLASS,
    MOD_CLASS,
    LEVEL_CLASS,
    INPUT_WIDTH,
    INPUT_HEIGHT,
    CONF_THRESHOLD
)
def parse_into_field_storage(fp, ctype, clength):
    fs = FieldStorage(
        fp=fp,
        environ={'REQUEST_METHOD': 'POST'},
        headers={
            'content-type': ctype,
            'content-length': clength
        },
        keep_blank_values=True
    )
    form = {}
    files = {}
    for f in fs.list:
        if f.filename:
            files.setdefault(f.name, []).append(f)
        else:
            form.setdefault(f.name, []).append(f.value)
    return form, files

def get_file_from_request_body(headers, body):
    fp = BytesIO(base64.b64decode(body)) # decode
    environ = {"REQUEST_METHOD": "POST"}
    headers = {
        "content-type": headers["content-type"],
        "content-length": headers["content-length"],
    }

    fs = FieldStorage(fp=fp, environ=environ, headers=headers) 
    return fs

def main(event, context):
    try: 
        files = get_file_from_request_body(
            headers=event["headers"], body=event["body"]
        )
        image = files.getvalue("image")
        pil_image_raw = Image.open(BytesIO(image))
    except:
        return {
            "cookies" : [],
            "isBase64Encoded": False,
            "statusCode": 400,
            "headers": { "content-type": "application/json" },
            "body": json.dumps({
                "message": "Unable to process image"
            })
        }

    pil_image, offset, resize_w, resize_h = resize(pil_image_raw, INPUT_WIDTH, INPUT_HEIGHT)

    # make prediction
    detect_res = detect(pil_image)
    # set 0.4 threshold
    detect_res = detect_res[detect_res['confidence'] > CONF_THRESHOLD]
    # calculate coordinate centers
    detect_res['xcenter'] = (detect_res['xmax'] + detect_res['xmin']) / 2
    detect_res['ycenter'] = (detect_res['ymax'] + detect_res['ymin']) / 2

    # separate predictions intro attributes
    potentials = detect_res[detect_res['class'] == POTENTIAL_CLASS]
    promotions = detect_res[detect_res['class'].isin(PROMOTION_CLASSES)]
    levels = detect_res[detect_res['class'] == LEVEL_CLASS]
    names = detect_res[detect_res['class'] == NAME_CLASS]
    operators = detect_res[detect_res['class'] == OPERATOR_CLASS]
    favorites = detect_res[detect_res['class'] == FAVORITE_CLASS]
    modules = detect_res[detect_res['class'] == MOD_CLASS]


    name_images = []
    level_images = []

    final_result = []
    # iterate through operator predictions
    for index, operator in operators.iterrows():
        o_xmin = operator.xmin
        o_xmax = operator.xmax
        o_ymin = operator.ymin
        o_ymax = operator.ymax

        # get potential info
        potential = get_potential(
            df=potentials, 
            img=pil_image,
            xmin=o_xmin, xmax=o_xmax, ymin=o_ymin, ymax=o_ymax,
            index=index
        )

        # get module info
        module = get_module(
            df=modules,
            img=pil_image,
            xmin=o_xmin, xmax=o_xmax, ymin=o_ymin, ymax=o_ymax
        )

        # get promotion info
        promotion = get_promotion(
            df=promotions, xmin=o_xmin, xmax=o_xmax, ymin=o_ymin, ymax=o_ymax
        )
        # get favorite
        favorite = get_favorite(
            df=favorites, xmin=o_xmin, xmax=o_xmax, ymin=o_ymin, ymax=o_ymax
        )

        # get name
        name = get_operator_name(
            df=names, 
            xmin=o_xmin, xmax=o_xmax, ymin=o_ymin, ymax=o_ymax,
            resize_w=resize_w, resize_h=resize_h,
            offset=offset,
            resize_full_height=pil_image.height,
            image_pil_raw=pil_image_raw,
            index=index
        )
        name_images.append(name)
        # get level
        level = get_operator_level(
            df=levels, 
            xmin=o_xmin, xmax=o_xmax, ymin=o_ymin, ymax=o_ymax,
            resize_w=resize_w, resize_h=resize_h,
            offset=offset,
            image_pil_raw=pil_image_raw,
            index=index
        )
        level_images.append(level)

        xyxy = (
            o_xmin,
            o_ymin,
            o_xmax,
            o_ymax
        )

        original_coords = get_original_rect(
            offset=offset, 
            resize_w=resize_w, 
            resize_h=resize_h, 
            xyxy=xyxy,
            width=pil_image_raw.width,
            height=pil_image_raw.height
        )

        data = {
            'potential': potential,
            'promotion': promotion,
            'favorite': favorite,
            'module': module,
            'xyxy': {
                'xmin': original_coords[0],
                'ymin': original_coords[1],
                'xmax': original_coords[2],
                'ymax': original_coords[3]
            }
        }
        final_result.append(data)


    names = name_recognition(name_images[:9], i=0)
    levels = level_recognition(level_images[:9], i=0)
    if len(final_result) > 9:
        names_last = name_recognition(name_images[9:], i=1)
        levels_last = level_recognition(level_images[9:], i=1)

        names = names + names_last
        levels = levels + levels_last

    for i, operator in enumerate(final_result):
        if i < len(names):
            operator['name'] = names[i]
        else:
            operator['name'] = ""
        
        if i < len(levels):
            operator['level'] = levels[i]
        else:
            operator['level'] = 1

    return {
        'operators': final_result,
        'length': len(final_result)
    }
