from PIL import Image
from cgi import FieldStorage
from io import BytesIO

import logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

from util import (
    resize, 
    get_potential,
    get_promotion,
    get_favorite,
    get_operator_name,
    get_operator_level,
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

def main(event, context):
    raise Exception(event)
    try:
        print(event)
        logger.info(event)
        logger.info(event["body"])
        body_file = BytesIO(bytes(event["body"], "utf-8"))
        form, files = parse_into_field_storage(
            body_file,
            event['headers']['content_type'],
            body_file.getbuffer().nbytes
        )
        logger.info(files)
    except:
        return {
            "statusCode": 400,
            "body": {
                'message': 'Could not parse image'
            }
        }
    # image = event['body']['image']
    # # only accept png and jpgs
    # if image.content_type not in ACCEPTED_EXTENSIONS:
    #     return {
    #         "statusCode": 415,
    #         "body": {
    #             'message': 'Unsupported image type'
    #         }
    #     }
    

    try:
        image = None
        for v in files.values():
            if len(v) > 0:
                image = v[0]
                break
        pil_image_raw = Image.open(image)
    except:
        return {
            "statusCode": 400,
            "body": {
                'message': 'Could not parse image'
            }
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

        data = {
            'potential': potential,
            'promotion': promotion,
            'favorite': favorite
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
        'statusCode': 200,
        'body': {
            'operators': final_result,
            'length': len(final_result)
        }
    }
