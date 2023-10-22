"""
This 'utils' is the real engine of the server,
which handle all data processing and model call.

"""
import copy
import sys
import base64
import os
import tempfile
import uuid
from pathlib import Path
import json
from flasgger import Swagger
from flask import Flask
import time
from flask_cors import CORS
import exception_handlers
import rapidoc
from config import settings
from exceptions import InternalServerError
from exceptions import SomethingWrongHappened
from exceptions import Validation
import numpy as np
import cv2
from cm_engine import Image_Pixel,Rectangle,Point, detect_multi_cards , deinitialize_model, recognize_card,  get_blur_percentage, Image, PointVector, initialize_model ,FieldVector, CardVector , detect_card , recognize_text_fields

ENGINE_CODES = {
    "1": 	(SomethingWrongHappened, "4008"),
    "2":	(SomethingWrongHappened, "4008"),
    "1215":	(SomethingWrongHappened, "4444"),
    "61300":	(SomethingWrongHappened, "4007"),
    # "61302":	"Encoding Image buffer fails.",
    # "61303":	"Saving the image failed.",
    # "3700":	"The input card corners are invalid.",
    # "3701":	"The given card side conflict with the detected card side when no card corners are given.",
    # "1800":	"The input fields must be of type text fields."
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_app() -> Flask:
    """
    This function creates the main flask app, register the needed blueprints.

    Returns
    ========
        flask.Flask
            The main flask app instance.

    """
    app = Flask(__name__)
    CORS(app, origins=settings.ALLOWED_CORS_ORIGINS)
    app.config["JSON_SORT_KEYS"] = False
    for key in settings.dict():
        app.config[key] = settings.dict().get(key)
    app.register_blueprint(exception_handlers.blueprint)
    app.register_blueprint(rapidoc.blueprint)
    return app


def register_swagger(app: Flask, docs_file: str) -> Flask:
    """
    This function registers swagger documentations from a yml file.

    Parameters
    ===========
        app
            The Flask app instance.
        docs_file
            The path of the yml documentations file.

    Returns
    ========
        flask.Flask
            The main flask app instance after registering swagger.

    """
    app.config["SWAGGER"] = {
        "uiversion": 3,
        "openapi": "3.0.2",
        "termsOfService": None,
        "favicon": "https://rdi-eg.ai/wp-content/uploads/2020/10/cropped-icon-32x32.png",
    }
    swagger_config = Swagger.DEFAULT_CONFIG
    swagger_config[
        "swagger_ui_bundle_js"
    ] = "//cdnjs.cloudflare.com/ajax/libs/swagger-ui/3.52.4/swagger-ui-bundle.min.js"
    swagger_config[
        "swagger_ui_standalone_preset_js"
    ] = "//cdnjs.cloudflare.com/ajax/libs/swagger-ui/3.52.4/swagger-ui-standalone-preset.min.js"
    swagger_config["jquery_js"] = "//unpkg.com/jquery@2.2.4/dist/jquery.min.js"
    swagger_config["swagger_ui_css"] = "//cdnjs.cloudflare.com/ajax/libs/swagger-ui/3.52.4/swagger-ui.min.css"
    Swagger(app, template_file=docs_file, config=swagger_config)
    return app


def call_count_units():
    """
    call count units is a function;
        which calls the main engine function to calculate the number of units in the input data.
        which the user needs to process this task.
    Inputs:
        file:
            The audio file path which the user needs to know how much this file will cost.
    Outputs:
        estimated_units:
            This is an integer value of units.

    """
    # TODO: populate this function
    ...


def base64_to_bytes(base64_img: str, error_code: str) -> bytes:
    """
    This function converts base64 images to an bytes file.

    Parameters
    ===========
        base64_img: str
            The base64 string representing the image.
        error_code: int
            The error code of the raised ValidationError if the string couldn't be converted.

    Returns
    ========
        bytes
            The image bytes after converting

    Raises
    =======
        Validation
            if the base64 string cannot be converted, our custom Validation error will be raised.

    """
    try:
        return base64.b64decode(base64_img)
    except (ValueError, TypeError):
        raise Validation(error_code)


def get_temp_file_name(ext):
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, f"{uuid.uuid4()}.{ext}")

def construct_color(r=0, g=0, b=0):
    color = Image_Pixel()
    color.red = r
    color.blue = g
    color.green = b
    return color
    
def fill_detected_card(detected_card_points, image, thickness=-1):
    x1,y1,x2,y2 = detected_card_points
    
    rectangle_upper = Point(x1,y1)
    rectangle_bottom = Point(x2,y2)
    image.draw_rectangle(Rectangle(rectangle_upper,rectangle_bottom), construct_color(255,255 ,255), thickness)
    return image

def poly_to_bbox(polygon):
    x_coords = []
    y_coords = []
    for p in polygon:
        x_coords.append(p.x)
        y_coords.append(p.y)
    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)
    bbox = [x1,y1,x2,y2]
    return bbox

def prepare_crop_pts(pts,w,h):
    #handling negative values
    for i in range(len(pts)):
        if pts[i] < 0:
            pts[i] = 0
    x1,y1,x2,y2 = pts
    x1 = x1-20 if x1-20 > 0 else 0
    y1 = y1-20 if y1-20 > 0 else 0
    x2 = x2+20 if x2+20 < w else w
    y2 = y2+20 if y2+20 < h else h
    return x1,y1,x2,y2


def check_cards_positions(x1,y1,x2,y2,w,h,img):
    ''' 
    this func checks detected card position in original image to 
    know how multi cards exists in original image , then we crop
    the detected card and pass the image to detect func again
    args : 
        x1,y1,x2,y2 : detcted card poly
        w ,h : img dimensions
        img : original image
    return : img ( after cropping detected card from original )  
    '''
    
    x_diff = x2-x1
    y_diff = y2-y1
    x_ratio = x_diff/w
    y_ratio = y_diff/h
    if x_ratio > y_ratio:
        if y1 < h-y2:
            img.crop(0,y2,w,h) #img[y2:h,:]
        else:
            img.crop(0,0,w,y1) #img[0:y1,:]
    else:
        if x1 < w-x2:
            img.crop(x2,0,w,h) #img[:,x2:w]
        else:
            img.crop(0,0,x1,h) #img[:,0:x1]
    return img

def get_base64_from_image(image, bgr=False):
    image.convert_color(Image.BGR if bgr else Image.RGB)
    temp_image_path = get_temp_file_name("png")
    saved = image.save(temp_image_path)
    if saved.code != 0:
        return ""
    with open(temp_image_path, "rb") as img_file:
        image_str = base64.b64encode(img_file.read())
        image_str = image_str.decode("utf-8")
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    return image_str


def prepare_result(result, return_cropped_images):
    cropped_images = {}
    if return_cropped_images:
        cropped_images = {
            "cropped_image": get_base64_from_image(result.warped_card, True)
        }
    return {
        **{
            field.type: field.value for field in result.text_fields
        },
        **{
            field.type: get_base64_from_image(field.image, True) for field in result.image_fields
        },
        **{
            "blur_percentage": result.blur_percent
        },
        **cropped_images
    }
def prepare_multi_result(card, image_fields , field_output , return_cropped_images,blur_percentage,doc_type):
    cropped_images = {}
    if return_cropped_images:
        cropped_images = {
            "cropped_image": get_base64_from_image(card.warped_card, True)
        }
    return {
        **{
            field.type: field.value for field in field_output.result
        },
        **{
            field.label: get_base64_from_image(field.image, True) for field in image_fields
        },
        **{
            "blur_percentage": blur_percentage
        },
        "doc_type":doc_type,
        **cropped_images
    }

def get_model_path(document_type, model_type):
    return os.path.join(BASE_DIR, settings.MODELS_DIR, document_type, model_type) + "/"


def call_process_document(image, return_cropped_images, document_type, ):
    image_data = base64_to_bytes(image, "4006")
    image_result = Image.construct(image_data, len(image_data))
    if image_result.error.code != 0:
        print(image_result.error.what, flush=True)
        raise SomethingWrongHappened("4007")
    recognition_output = recognize_card(
        image_result.result,
        get_model_path(document_type, 'layout'),
        get_model_path(document_type, 'recognition'),
        settings.LAYOUT_THREADS,
        "Non",
        PointVector([]),
        False,
        settings.RECOGNITION_THREADS
    )

    if recognition_output.error.code != 0:
        print(recognition_output.error.code, recognition_output.error.what, flush=True)
        error, error_code = ENGINE_CODES.get(f"{recognition_output.error.code}", (InternalServerError, "4500"))
        raise error(error_code)

    return prepare_result(recognition_output.result, return_cropped_images)



def call_get_cards_and_recognize(image, return_cropped_images, document_type, ):
    image_data = base64_to_bytes(image, "4006")
    image_output = Image.construct(image_data, len(image_data))
    if image_output.error.code != 0:
        print(image_output.error.what, flush=True)
        raise SomethingWrongHappened("4007")   

    blur_result = get_blur_percentage(image_output.result, get_model_path(document_type, 'layout'))

    cards = CardVector()
    card = PointVector()
    card.type = "intitated_type" # any intialization value ( not Non ) to start the for loop
    i = 0 # counting number of detected cards to resize CardVector
    img = image_output.result
    while card.type != "Non":
        output = detect_card(img, get_model_path(document_type, 'layout'),settings.LAYOUT_THREADS, 
        "Non",PointVector(), False , False)
        card, error = output.result, output.error
        if error.code != 0:
            print(error.code, error.what, flush=True)
            error, error_code = ENGINE_CODES.get(f"{error.code}", (InternalServerError, "4500"))
            raise error(error_code)
        if card.type != "Non":
            w , h = img.width() , img.height()
            card_bbox = poly_to_bbox(card.polygon) 
            x1,y1,x2,y2 = card_bbox
            cards.resize(i+1)
            cards[i] = card
            img = check_cards_positions(x1,y1,x2,y2,w,h,img)
            i += 1
        else:
            break
        
    with open("model_name2_eng_fields.json") as f :
        modelname_2_engfields =  json.load(f)
    
    # recognize the card's fields.
    cards_dicts = {}
    for i in range(cards.size()):
        card = cards[i]
        fields = card.fields
        # text_fields = []
        eng_fields = {}
        image_fields = []
        if fields.size() > 0 :
            recog_fields = FieldVector()
            for field in fields:
                lbl = field.label
                if "image" in lbl or "Image" in lbl:
                    image_fields.append(field)
                elif lbl in modelname_2_engfields[document_type]:
                    eng_fields[lbl] = get_base64_from_image(field.image,True)
                else:
                    recog_fields.append(field)

            field_output = recognize_text_fields(recog_fields, get_model_path(document_type, 'recognition'), use_double_inference=True)
            if field_output.error.code != 0:
                print(field_output.error.code, field_output.error.what, flush=True)
                error, error_code = ENGINE_CODES.get(f"{field_output.error.code}", (InternalServerError, "4500"))
                raise error(error_code)
        
        if eng_fields != {} :
            with open("fields_type_2_base64img.json","w") as h:
                h.write(json.dumps(eng_fields,indent=4))

            os.system("export LD_LIBRARY_PATH=/opt/python/lib; python trocr_infer.py")

            with open("fields_type_2_text.json") as h:
                eng_fields= json.load(h)

            os.system("rm fields_type_2_text.json")

            cards_dicts["card_" + str(i)] = prepare_multi_result(card, image_fields , field_output ,return_cropped_images,blur_result.result,document_type)
            
            for eng_f in eng_fields:
                cards_dicts["card_" + str(i)][eng_f] = eng_fields[eng_f]

    return cards_dicts

def get_cv2img_from_base64(base64_string):
    img = base64.b64decode(base64_string.encode('utf-8'))
    #reading image from bytes
    img = np.frombuffer(img, dtype=np.uint8)
    #converting to np data structure
    img = cv2.imdecode(img, flags=1)
    return img

def k2_recognize(recog_fields):
    for field in recog_fields:
        image = field.image
        lbl = field.label
        
    return

def inference_mrz(user_id):
    cards_dir = "../../Data/cards/"+user_id
    os.system("cd mrz-server; export LD_LIBRARY_PATH=$(pwd); python e-kyc.py "+cards_dir)
    with open("../Data/Processing_results/"+user_id+".json") as f:
        results = json.load(f)
    return results

def process_docs_in_image(user_id,image, return_cropped_images,k2_recognition):

    request_time = time.time()
    request_time = str(request_time)

    #reading image
    image_data = base64_to_bytes(image, "4006")
    image_output = Image.construct(image_data, len(image_data))
    # output = Image.construct(path, Image.BGR)
    if image_output.error.code != 0:
        print(image_output.error.what, flush=True)
        raise SomethingWrongHappened("4007")   

    #localizing and classifying cards
    img = image_output.result
    output = detect_multi_cards(img, "../Data/Models/"+user_id ,settings.LAYOUT_THREADS,False)
    cards = output.result 
    if output.error.code not in [0,3703]:
        print(output.error.code, output.error.what, flush=True)
        error, error_code = ENGINE_CODES.get(f"{output.error.code}", (InternalServerError, "4500"))
        raise error(error_code)

    #removing unsupported detected cards
    documents_types = []
    for i in range(cards.size()-1,-1,-1):
        if cards[i].type != "Non":
            documents_types.append(cards[i].type)
        else:
            cards.__delitem__(i)
            
    if cards.size() == 0:
        return {"Cards":"There is no supported docs in the image"}
    
    os.makedirs("../Data/cards/"+user_id,exist_ok=True)
    
    Mrz = False
    i=0
    for card in cards:
        if card.type == 'Mrz':
            Mrz = True
            cards[i].warped_card.convert_color(Image.BGR)
            cards[i].warped_card.save("../Data/cards/"+user_id+"/card_"+str(i+1)+".jpg")
            i+=1

    if Mrz:
        mrz_results = inference_mrz(user_id)
        return mrz_results

    #json file containing all english fields that will be passed to tr_ocr 
    with open("model_name2_eng_fields_newmdlsnaming.json") as f :
        modelname_2_engfields =  json.load(f)
    
    #processing classified cards
    #intializing all needed models
    for mdl in set(documents_types):
        initialize_model(get_model_path(mdl, 'layout'),get_model_path(mdl, 'recognition'))

    cards_dicts ={}
    eng_fields = {} #getting eng fields ready to trocr
    arabic_fields = {}
    # cards_image_fields ={}

    for i in range(cards.size()):
    
    #     recognition_output = recognize_card(
    #     cards[i].warped_card,
    #     get_model_path(documents_types[i], 'layout'),
    #     get_model_path(documents_types[i], 'recognition'),
    #     settings.LAYOUT_THREADS,
    #     "Non",
    #     PointVector([]),
    #     False,
    #     settings.RECOGNITION_THREADS
    # )
    #     if recognition_output.error.code != 0:
    #         print(recognition_output.error.code, recognition_output.error.what, flush=True)
    #         error, error_code = ENGINE_CODES.get(f"{recognition_output.error.code}", (InternalServerError, "4500"))
    #         raise error(error_code)
        
        blur_result = get_blur_percentage(cards[i].warped_card, get_model_path(documents_types[i], 'layout'))
        card_output = detect_card(cards[i].warped_card, get_model_path(documents_types[i], 'layout'),settings.LAYOUT_THREADS,"Non",PointVector(), False , True)
        error = card_output.error
        if error.code != 0:
            print(error.code, error.what, flush=True)
            error, error_code = ENGINE_CODES.get(f"{error.code}", (InternalServerError, "4500"))
            raise error(error_code)

        fields = card_output.result.fields
        image_fields = []
        if fields.size() > 0 :
            recog_fields = FieldVector()
            for field in fields:
                lbl = field.label
                if "image" in lbl or "Image" in lbl:
                    image_fields.append(field)
                    # cards_image_fields["card_"+str(i+1)+"_"+lbl] = get_base64_from_image(field.image,True)
                    cards_dicts ["card_"+str(i+1)]= {}
                    cards_dicts["card_"+str(i+1)][lbl] = get_base64_from_image(field.image,True)
                elif lbl in modelname_2_engfields[documents_types[i]]:
                    eng_fields["card_"+str(i+1)+"_"+lbl] = get_base64_from_image(field.image,True) #here i+1 just for not returning card_0 , starting from card_1
                    recog_fields.append(field)
                else:
                    recog_fields.append(field)
                    arabic_fields["card_"+str(i+1)+"_"+lbl] = get_base64_from_image(field.image,True)
                    
            if not k2_recognition:
                field_output = recognize_text_fields(recog_fields, get_model_path(documents_types[i], 'recognition'), use_double_inference=False)
                if field_output.error.code != 0:
                    print(field_output.error.code, field_output.error.what, flush=True)
                    error, error_code = ENGINE_CODES.get(f"{field_output.error.code}", (InternalServerError, "4500"))
                    raise error(error_code)

                cards_dicts["card_" + str(i+1)] = prepare_multi_result(cards[i], image_fields , field_output ,return_cropped_images,blur_result.result,documents_types[i])
        cards[i].warped_card.convert_color(Image.BGR)
        cards[i].warped_card.save("../Data/cards/"+user_id+"/card_"+str(i+1)+".jpg")
    #getting the correct path of the needed libs in both conda or docker
    lib_path = [ x for x in sys.path if "-packages" in x ]
    lib_path_lens = [ len(x) for x in lib_path ]
    torch_lib_path = lib_path[lib_path_lens.index(max(lib_path_lens))]

    if k2_recognition and arabic_fields != {}:
        with open("k2/V1_0/ar_fields.json","w") as f:
            f.write(json.dumps(arabic_fields,indent=4)) 
        os.system("cd k2/V1_0; export LD_LIBRARY_PATH="+torch_lib_path+"/torch/lib; python e-kyc.py -J ar_fields.json -O Output/ -M Models/Ar/")    
        for field_lbl_img in arabic_fields:
            with open("k2/V1_0/Output/"+field_lbl_img+".json") as f:
                result_json = json.load(f)                   
            text = result_json["zones"][0]['paragraphs'][0]['lines'][0]['text']
            card_no = "_".join(field_lbl_img.split("_")[:2])
            field_label = "_".join(field_lbl_img.split("_")[2:])
            cards_dicts[card_no][field_label] = text

    if k2_recognition and eng_fields != {}:
        with open("k2/V1_0/en_fields.json","w") as f:
            f.write(json.dumps(eng_fields,indent=4)) 
        os.system("cd k2/V1_0; export LD_LIBRARY_PATH="+torch_lib_path+"/torch/lib; python e-kyc.py -J en_fields.json -O Output/ -M Models/En/")    
        for field_lbl_img in eng_fields:
            with open("k2/V1_0/Output/"+field_lbl_img+".json") as f:
                result_json = json.load(f)                   
            text = result_json["zones"][0]['paragraphs'][0]['lines'][0]['text']
            card_no = "_".join(field_lbl_img.split("_")[:2])
            field_label = "_".join(field_lbl_img.split("_")[2:])
            cards_dicts[card_no][field_label] = text

    # if eng_fields != {} :
    #     with open("fields_type_2_base64img"+request_time+".json","w") as h:
    #         h.write(json.dumps(eng_fields,indent=4))
    #     #getting the correct path of the needed libs in both conda or docker
    #     lib_path = [ x for x in sys.path if "site-packages" in x ]
    #     lib_path_lens = [ len(x) for x in lib_path ]
    #     torch_lib_path = lib_path[lib_path_lens.index(max(lib_path_lens))]

    #     os.system("export LD_LIBRARY_PATH="+torch_lib_path+"/torch/lib; python trocr_infer.py "+request_time)

    #     with open("fields_type_2_text"+request_time+".json") as h:
    #         eng_fields= json.load(h)

    #     os.system("rm fields_type_2_text"+request_time+".json")

    #     for eng_f in eng_fields:
    #         card_no = "_".join(eng_f.split("_")[:2])
    #         field_label = "_".join(eng_f.split("_")[2:])
    #         cards_dicts[card_no][field_label] = eng_fields[eng_f]
    
    #deintializing all needed models
    for mdl in set(documents_types):
        deinitialize_model(get_model_path(mdl, 'layout'),get_model_path(mdl, 'recognition'))

    # cards_dicts['Card_' + str(i)] = prepare_result(recognition_output.result, return_cropped_images) #uncomment to work with reconize_card func
    return cards_dicts
           
def get_documents_list():
    return [
        directory.name for directory in Path(settings.MODELS_DIR).glob("*") if directory.is_dir()
    ]
